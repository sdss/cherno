#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-09-13
# @Filename: astrometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from functools import partial

from typing import Any, NamedTuple, Optional, Union

import matplotlib
import numpy
import pandas
import sep
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from coordio.defaults import PLATE_SCALE
from coordio.utils import radec2wokxy

from clu.command import FakeCommand

from cherno import config
from cherno.actor import ChernoCommandType
from cherno.coordinates import gfa_to_radec, gfa_to_wok, umeyama
from cherno.maskbits import GuiderStatus
from cherno.tcc import apply_correction


matplotlib.use("Agg")

PathType = Union[str, pathlib.Path]


warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)


class TimedProcess(NamedTuple):
    """A completed process which includes its elapsed time."""

    process: asyncio.subprocess.Process
    elapsed: float


class AstrometryNet:
    """A wrapper for the astrometry.net ``solve-field`` command.

    Parameters
    ----------
    configure_params
        Parameters to be passed to `.configure`.
    """

    def __init__(self, **configure_params):
        solve_field_cmd = subprocess.run(
            "which solve-field", shell=True, capture_output=True
        )
        solve_field_cmd.check_returncode()

        self.solve_field_cmd = solve_field_cmd.stdout.decode().strip()

        self._options = {}
        self.configure(**configure_params)

    def configure(
        self,
        backend_config: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sort_column: Optional[str] = None,
        sort_ascending: Optional[bool] = None,
        no_plots: Optional[bool] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius: Optional[float] = None,
        scale_low: Optional[float] = None,
        scale_high: Optional[float] = None,
        scale_units: Optional[str] = None,
        dir: Optional[str] = None,
        **kwargs,
    ):
        """Configures how to run of ``solve-field```.

        The parameters this method accepts are identical to those of
        ``solve-field`` and are passed unchanged.

        Parameters
        ----------
        backend_config
            Use this config file for the ``astrometry-engine`` program.
        width
            Specify the field width, in pixels.
        height
            Specify the field height, in pixels.
        sort_column
            The FITS column that should be used to sort the sources.
        sort_ascending
            Sort in ascending order (smallest first);
            default is descending order.
        no_plot
            Do not produce plots.
        ra
            RA of field center for search, in degrees.
        dec
            Dec of field center for search, in degrees.
        radius
            Only search in indexes within ``radius`` degrees of the field
            center given by ``ra`` and ``dec``.
        scale_low
            Lower bound of image scale estimate.
        scale_high
            Upper bound of image scale estimate.
        scale_units
            In what units are the lower and upper bounds? Choices:
            ``'degwidth'``, ``'arcminwidth'``, ``'arcsecperpix'``,
            ``'focalmm'``.
        dir
            Path to the directory where all output files will be saved.
        """

        self._options = {
            "backend-config": backend_config,
            "width": width,
            "height": height,
            "sort-column": sort_column,
            "sort-ascending": sort_ascending,
            "no-plots": no_plots,
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "scale-low": scale_low,
            "scale-high": scale_high,
            "scale-units": scale_units,
            "dir": dir,
            "overwrite": True,
        }
        self._options.update(kwargs)

        return

    def _build_command(self, files, options=None):
        """Builds the ``solve-field`` command to run."""

        if options is None:
            options = self._options

        flags = ["no-plots", "sort-ascending", "overwrite"]

        cmd = [self.solve_field_cmd]

        for option in options:
            if options[option] is None:
                continue
            if option in flags:
                if options[option] is True:
                    cmd.append("--" + option)
            else:
                cmd.append("--" + option)
                cmd.append(str(options[option]))

        cmd += list(files)

        return cmd

    async def run(
        self,
        files: list[PathType],
        shell: bool = True,
        stdout: Optional[PathType] = None,
        stderr: Optional[PathType] = None,
        **kwargs,
    ) -> TimedProcess:
        """Runs astrometry.net.

        Parameters
        ----------
        files
            List of files to be processed.
        shell
            Whether to call `subprocess.run` with ``shell=True``.
        stdout
            Path where to save the stdout output.
        stderr
            Path where to save the stderr output.
        kwargs
            Configuration parameters (see `.configure`) to override. The
            configuration applies only to this run of ``solve-field`` and it
            is not saved.

        Returns
        -------
        `subprocess.CompletedProcess`
            The completed process.

        """

        options = self._options.copy()
        options.update(kwargs)

        if not isinstance(files, (tuple, list)):
            files = [files]

        t0 = time.time()

        args = self._build_command(files, options=options)

        if shell:
            cmd = await asyncio.create_subprocess_shell(
                " ".join(args),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        else:
            cmd = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        stdout_bytes, stderr_bytes = await cmd.communicate()

        elapsed = time.time() - t0

        if stdout:
            with open(stdout, "wb") as out:
                out.write(" ".join(args).encode() + b"\n")
                out.write(stdout_bytes)

        if stderr:
            with open(stderr, "wb") as err:
                err.write(stderr_bytes)

        return TimedProcess(cmd, elapsed)


@dataclass
class ExtractionData:
    """Data from extraction."""

    image: str
    camera: str
    exposure_no: int
    observatory: str
    field_ra: float
    field_dec: float
    field_pa: float
    nregions: int
    nvalid: int
    background_rms: float
    solved: bool = False
    fwhm: float = numpy.nan
    a: float = numpy.nan
    b: float = numpy.nan
    ellipticity: float = numpy.nan
    nkeep: int = 0
    wcs: WCS = field(default_factory=WCS)
    solve_time: float = 0.0
    camera_racen: float = numpy.nan
    camera_deccen: float = numpy.nan
    xrot: float = numpy.nan
    yrot: float = numpy.nan
    rotation: float = numpy.nan
    proc_image: str | None = None


async def extract_and_run(
    images: pathlib.Path | str | list[pathlib.Path] | list[str],
    astrometry_outdir: PathType = "astrometry",
    proc_image_outdir: PathType | None = None,
    sigma: float = 10.0,
    min_npix: int = 50,
    cpulimit: float = 15.0,
    overwrite: bool = False,
    plot: bool = True,
    command: ChernoCommandType | None = None,
) -> list[ExtractionData]:
    """Extracts sources and runs Astrometry.net.

    Returns the WCS object for each processed image or `None` if failed.

    """

    import matplotlib.pyplot as plt

    plt.ioff()

    if isinstance(images, (list, tuple)):
        results = await asyncio.gather(
            *[
                extract_and_run(
                    image,
                    astrometry_outdir=astrometry_outdir,
                    proc_image_outdir=proc_image_outdir,
                    sigma=sigma,
                    min_npix=min_npix,
                    cpulimit=cpulimit,
                    overwrite=overwrite,
                    plot=plot,
                    command=command,
                )
                for image in images
            ]
        )
        return [result[0] for result in results]

    assert isinstance(images, (pathlib.Path, str))
    image = images

    header = fits.getheader(image, 1)
    camera = header["CAMNAME"][0:-1]  # Remove the n/s at the end of the camera name.

    path = pathlib.Path(image)
    mjd = path.parts[-2]
    dirname = path.parent
    proc_basename = "proc-" + path.parts[-1]

    match = re.match(r".*gimg\-gfa\d[ns]\-(\d+)\.fits", path.parts[-1])
    if match:
        exp_no = int(match.group(1))
    else:
        exp_no = 0

    if os.path.isabs(astrometry_outdir):
        astrometry_outdir = os.path.join(astrometry_outdir, mjd)
    else:
        astrometry_outdir = os.path.join(dirname, astrometry_outdir)

    if not os.path.exists(astrometry_outdir):
        os.makedirs(astrometry_outdir)

    outfile_root = os.path.join(astrometry_outdir, proc_basename)

    data: Any = fits.getdata(image)
    back = await asyncio.get_running_loop().run_in_executor(
        None,
        sep.Background,
        data.astype("int32"),
    )

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(back, origin="lower")
        ax.set_title("Background: " + path.parts[-1])
        ax.set_gid(False)
        fig.savefig(outfile_root + "-background.png", dpi=300)

    extract = partial(
        sep.extract,
        data - back.back(),
        sigma,
        err=back.globalrms,
    )
    regions = pandas.DataFrame(
        await asyncio.get_running_loop().run_in_executor(None, extract)
    )

    if len(regions) > 0:
        regions.loc[regions.npix > min_npix, "valid"] = 1

        # Keep only detectons with flux and sort them by highest flux first.
        # regions.dropna(subset=["flux"], inplace=True)
        # regions.sort_values("flux", inplace=True, ascending=False)

        valid = regions.loc[regions.valid == 1]
        regions.to_hdf(outfile_root + ".hdf", "data")
    else:
        valid = []

    extraction_data = ExtractionData(
        str(image),
        camera,
        exposure_no=exp_no,
        observatory=config["observatory"],
        field_ra=header["RAFIELD"],
        field_dec=header["DECFIELD"],
        field_pa=header["FIELDPA"],
        nregions=len(regions),
        nvalid=len(valid),
        background_rms=back.globalrms,
    )

    if len(valid) < 5:  # Don't even try.
        if command is not None:
            command.warning(f"Camera {camera}: not enough sources.")
        return [extraction_data]

    pixel_scale = config["cameras"]["pixel_scale"]

    fwhm, a, b, ell, nkeep = calculate_fwhm_camera(valid, rej_low=1, rej_high=3)
    fwhm = numpy.round(fwhm * pixel_scale if fwhm != -999.0 else fwhm, 2)
    a = numpy.round(a * pixel_scale if a != -999.0 else a, 2)
    b = numpy.round(b * pixel_scale if b != -999.0 else b, 2)
    ell = numpy.round(ell, 2)

    if command is not None:
        command.debug(fwhm_camera=[camera, exp_no, fwhm, a, b, ell, nkeep])

    extraction_data.fwhm = fwhm
    extraction_data.a = a
    extraction_data.b = b
    extraction_data.ellipticity = ell
    extraction_data.nkeep = nkeep

    gfa_xyls = Table.from_pandas(regions.loc[:, ["x", "y"]])
    gfa_xyls_file = outfile_root + ".xyls"
    gfa_xyls.write(gfa_xyls_file, format="fits", overwrite=True)

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(path.parts[-1] + r" $(\sigma={})$".format(sigma))
        ax.set_gid(False)

        data_back = data - back.back()
        ax.imshow(
            data_back,
            origin="lower",
            cmap="gray",
            vmin=data_back.mean() - back.globalrms,
            vmax=data_back.mean() + back.globalrms,
        )
        fig.savefig(outfile_root + "-original.png", dpi=300)
        ax.scatter(
            regions.loc[regions.valid == 1].x,
            regions.loc[regions.valid == 1].y,
            marker="x",  # type: ignore
            s=3,
            c="r",
        )
        fig.savefig(outfile_root + "-centroids.png", dpi=300)

    backend_config = os.path.join(os.path.dirname(__file__), "etc/astrometrynet.cfg")
    astrometry_net = AstrometryNet()
    astrometry_net.configure(
        backend_config=backend_config,
        width=2048,
        height=2048,
        no_plots=True,
        scale_low=pixel_scale * 0.9,
        scale_high=pixel_scale * 1.1,
        scale_units="arcsecperpix",
        # sort_column="flux",
        radius=0.5,
        dir=astrometry_outdir,
        cpulimit=cpulimit,
    )

    wcs_output = outfile_root + ".wcs"
    if os.path.exists(wcs_output):
        os.remove(wcs_output)

    camera_id = int(camera[-1])
    radec_centre = gfa_to_radec(
        1024,
        1024,
        camera_id,
        header["RA"],
        header["DEC"],
        position_angle=header["FIELDPA"],
        site_name=config["observatory"],
    )

    proc = await astrometry_net.run(
        [gfa_xyls_file],
        stdout=outfile_root + ".stdout",
        stderr=outfile_root + ".stderr",
        ra=radec_centre[0],
        dec=radec_centre[1],
    )

    proc_hdu = fits.open(image).copy()

    rec = Table.from_pandas(regions).as_array()
    proc_hdu.append(fits.BinTableHDU(rec, name="CENTROIDS"))

    extraction_data.solve_time = proc.elapsed

    if os.path.exists(wcs_output):
        extraction_data.solved = True
        wcs = WCS(open(wcs_output).read())
        extraction_data.wcs = wcs

        racen, deccen = wcs.pixel_to_world_values([[1024, 1024]])[0]
        extraction_data.camera_racen = numpy.round(racen, 6)
        extraction_data.camera_deccen = numpy.round(deccen, 6)

        proc_hdu[1].header.update(wcs.to_header())

        # TODO: consider parallactic angle here.
        cd = wcs.wcs.cd
        rot_rad = numpy.arctan2([cd[0, 1], cd[0, 0]], [cd[1, 1], cd[1, 0]])

        # Rotation is from N to E to the x and y axes of the GFA.
        yrot, xrot = numpy.rad2deg(rot_rad)
        extraction_data.xrot = numpy.round(xrot % 360.0, 3)
        extraction_data.yrot = numpy.round(yrot % 360.0, 3)

        # Calculate field rotation.
        camera_rot = config["cameras"]["rotation"][camera]
        rotation = numpy.array([xrot - camera_rot - 90, yrot - camera_rot]) % 360
        rotation[rotation > 180.0] -= 360.0  # type: ignore
        rotation = numpy.mean(rotation)
        extraction_data.rotation = numpy.round(rotation, 3)

    proc_hdu[1].header["SOLVED"] = extraction_data.solved
    proc_hdu[1].header["SOLVTIME"] = (proc.elapsed, "Time to solve the field or fail")
    proc_hdu[1].header["FWHM"] = (fwhm, "Average FWHM in arcsec")

    if command is not None and command.actor is not None:
        offsets = command.actor.state.offset
        proc_hdu[1].header["OFFRA"] = (offsets[0], "Offset in RA [arcsec]")
        proc_hdu[1].header["OFFDEC"] = (offsets[1], "Offset in Dec [arcsec]")
        proc_hdu[1].header["OFFPA"] = (offsets[2], "Offset in PA [arcsec]")

    if command is not None:
        if extraction_data.solved is False:
            command.info(
                camera_solution=[
                    camera,
                    exp_no,
                    False,
                    -999.0,
                    -999.0,
                    -999.0,
                    -999.0,
                    -999.0,
                ]
            )
        else:
            command.info(
                camera_solution=[
                    camera,
                    exp_no,
                    True,
                    extraction_data.camera_racen,
                    extraction_data.camera_deccen,
                    extraction_data.xrot,
                    extraction_data.yrot,
                    extraction_data.rotation,
                ]
            )

    procpath = os.path.join(str(proc_image_outdir or dirname), "proc-" + path.parts[-1])
    loop = asyncio.get_running_loop()
    func = partial(
        proc_hdu.writeto,
        procpath,
        overwrite=overwrite,
        output_verify="silentfix",
    )
    await loop.run_in_executor(None, func)

    plt.close("all")

    extraction_data.proc_image = procpath

    return [extraction_data]


def calculate_fwhm_camera(
    regions,
    rej_low: int = 1,
    rej_high: int = 3,
):
    """Calcualtes the FWHM from a list of detections with outlier rejection.

    Parameters
    ----------
    regions
        The pandas data frame with the list of regions. Usually an output from
        ``sep``. Must include columns ``valid``, ``a``, and ``b``.
    rej_low
        How many of the lowest ranked FWHM measurements to remove for the
        average.
    rej_high
        How many of the highest ranked FWHM measurements to remove for the
        average.

    Returns
    -------
    fwhm,a,b,ell,nkeep
        The FWHM measured as the average of the circle that envelops the minor
        and major axis after outlier rejection, the averaged semi-major and
        smi-minor axes, the ellipticity, and the number of data points kept.

    """

    if len(regions) == 0:
        return -999, -999, -999, -999, 0

    if len(regions) < 10:
        rej_low = rej_high = 0

    fwhm = numpy.max([regions.a * 2, regions.b * 2], axis=0)
    fwhm_argsort = numpy.argsort(fwhm)

    if len(fwhm) - (rej_low + rej_high) <= 0:
        nkeep = len(fwhm)
    else:
        fwhm_argsort = fwhm_argsort.tolist()[rej_low : len(fwhm_argsort) - rej_high]
        nkeep = len(fwhm_argsort)

    fwhm = numpy.mean(fwhm[fwhm_argsort])
    a = numpy.mean(regions.a.iloc[fwhm_argsort])
    b = numpy.mean(regions.b.iloc[fwhm_argsort])
    ell = 1 - b / a

    return fwhm, a, b, ell, nkeep


def astrometry_fit(
    data: list[ExtractionData],
    grid=(10, 10),
    offset: tuple = (0.0, 0.0, 0.0),
):
    """Fits translation, rotation, and scale from a WCS solution."""

    offset_ra, offset_dec, offset_pa = offset

    xwok_gfa: list[float] = []
    ywok_gfa: list[float] = []
    xwok_astro: list[float] = []
    ywok_astro: list[float] = []

    for d in data:

        camera_id = int(d.camera[-1])
        xidx = numpy.arange(2048)[:: 2048 // grid[0]]
        yidx = numpy.arange(2048)[:: 2048 // grid[1]]

        coords: Any = d.wcs.pixel_to_world(xidx, yidx)
        ra = coords.ra.value
        dec = coords.dec.value

        for x, y in zip(xidx, yidx):
            xw, yw, _ = gfa_to_wok(x, y, camera_id)
            xwok_gfa.append(xw)
            ywok_gfa.append(yw)

        offset_ra_corr = offset_ra * numpy.cos(numpy.deg2rad(d.field_dec)) / 3600.0

        _xwok_astro, _ywok_astro, *_ = radec2wokxy(
            ra,
            dec,
            None,
            "GFA",
            d.field_ra - offset_ra_corr,
            d.field_dec - offset_dec / 3600.0,
            d.field_pa - offset_pa / 3600.0,
            "APO",
            None,
            pmra=None,
            pmdec=None,
            parallax=None,
        )

        xwok_astro += _xwok_astro.tolist()
        ywok_astro += _ywok_astro.tolist()

    X = numpy.array([xwok_gfa, ywok_gfa])
    Y = numpy.array([xwok_astro, ywok_astro])
    try:
        c, R, t = umeyama(X, Y)
    except ValueError:
        return False

    plate_scale = PLATE_SCALE[data[0].observatory]

    delta_x = numpy.round(t[0] / plate_scale * 3600.0, 3)
    delta_y = numpy.round(t[1] / plate_scale * 3600.0, 3)

    # delta_x and delta_y only align with RA/Dec if PA=0. Otherwise we need to
    # project using the PA.
    pa_rad = numpy.deg2rad(data[0].field_pa)
    delta_ra = delta_x * numpy.cos(pa_rad) + delta_y * numpy.sin(pa_rad)
    delta_dec = -delta_x * numpy.sin(pa_rad) + delta_y * numpy.cos(pa_rad)

    # Round up.
    delta_ra = numpy.round(delta_ra, 3)
    delta_dec = numpy.round(delta_dec, 3)

    delta_rot = numpy.round(-numpy.rad2deg(numpy.arctan2(R[1, 0], R[0, 0])) * 3600.0, 1)
    delta_scale = numpy.round(c, 6)

    delta_x = (numpy.array(xwok_gfa) - numpy.array(xwok_astro)) ** 2  # type: ignore
    delta_y = (numpy.array(ywok_gfa) - numpy.array(ywok_astro)) ** 2  # type: ignore

    xrms = numpy.round(numpy.sqrt(numpy.sum(delta_x) / len(delta_x)), 3)
    yrms = numpy.round(numpy.sqrt(numpy.sum(delta_y) / len(delta_y)), 3)
    rms = numpy.round(numpy.sqrt(numpy.sum(delta_x + delta_y) / len(delta_x)), 3)

    return (delta_ra, delta_dec, delta_rot, delta_scale, xrms, yrms, rms)


async def process_and_correct(
    command: ChernoCommandType | FakeCommand,
    filenames: list[str],
    apply: bool = True,
    full: bool = False,
):
    """Processes a series of files for the same pointing and applies a correction."""

    assert command and command.actor

    min_npix = command.actor.state.acquisition["min_npix"]
    cpulimit = command.actor.state.acquisition["cpulimit"]
    sigma = command.actor.state.acquisition["sigma"]

    data = await extract_and_run(
        filenames,
        plot=False,
        sigma=sigma,
        cpulimit=cpulimit,
        min_npix=min_npix,
        command=command,
    )

    solved = [d for d in data if d.solved is True]
    nkeep = [d.nkeep for d in solved]

    if len(solved) == 0:
        command.error(acquisition_valid=False, did_correct=False)
        update_proc_headers(data, False, command.actor.state.guide_loop)
        return False

    fwhm = numpy.average([d.fwhm for d in solved], weights=nkeep)
    ellipticity = numpy.average([d.ellipticity for d in solved], weights=nkeep)
    camera_rotation = numpy.average([d.rotation for d in solved], weights=nkeep)

    if solved[0].field_ra == "NaN" or isinstance(solved[0].field_ra, str):
        command.error(acquisition_valid=False, did_correct=False)
        command.error("Field not defined. Cannot run astrometric fit.")
        update_proc_headers(data, False, command.actor.state.guide_loop)
        return False

    if (offset := command.actor.state.offset) == (0.0, 0.0, 0.0):
        command.debug(offset=list(offset))
    else:
        command.warning(offset=list(offset))

    fit = astrometry_fit(solved, offset=command.actor.state.offset)

    exp_no = solved[0].exposure_no  # Should be the same for all.

    if fit is False:
        delta_ra = delta_dec = delta_rot = delta_scale = -999.0
        rms = -999.0
    else:
        delta_ra = fit[0]
        delta_dec = fit[1]
        delta_rot = fit[2]
        delta_scale = fit[3]

        xrms = fit[4]
        yrms = fit[5]
        rms = fit[6]

        command.debug(guide_rms=[exp_no, xrms, yrms, rms])

    command.info(
        astrometry_fit=[
            exp_no,
            len(solved),
            -999.0,
            -999.0,
            numpy.round(fwhm, 2),
            numpy.round(ellipticity, 2),
            numpy.round(camera_rotation, 3),
            delta_ra,
            delta_dec,
            delta_rot,
            delta_scale,
        ]
    )

    actor_state = command.actor.state
    guider_status = actor_state.status

    stopping = (guider_status & (GuiderStatus.STOPPING | GuiderStatus.IDLE)).value > 0
    will_apply = apply is True and stopping is False

    correction_applied: list[float] = [0.0, 0.0, 0.0, 0.0]

    if will_apply is True:
        command.info("Applying corrections.")

        min_isolated = actor_state.guide_loop["rot"]["min_isolated_correction"]
        if abs(delta_rot) >= min_isolated:
            command.debug("Applying only large rotator correction.")
            correction_applied = await apply_correction(
                command,
                rot=-delta_rot,
                k_rot=None if full is False else 1.0,
            )

        else:

            correction_applied = await apply_correction(
                command,
                rot=-delta_rot,
                radec=(-delta_ra, -delta_dec),
                k_radec=None if full is False else 1.0,
                k_rot=None if full is False else 1.0,
            )

        command.info(
            acquisition_valid=True,
            did_correct=True,
            correction_applied=correction_applied,
        )

    else:
        command.info(acquisition_valid=True, did_correct=False)

    update_proc_headers(
        data,
        will_apply,
        command.actor.state.guide_loop,
        rms=rms,
        delta_ra=delta_ra,
        delta_dec=delta_dec,
        delta_rot=delta_rot,
        delta_scale=delta_scale,
        correction_applied=correction_applied,
    )

    return True


def update_proc_headers(
    data: list[ExtractionData],
    applied: bool,
    guide_loop: dict,
    rms: float = -999.0,
    delta_ra: float = -999.0,
    delta_dec: float = -999.0,
    delta_rot: float = -999.0,
    delta_scale: float = -999.0,
    correction_applied: list[float] = [0.0, 0.0, 0.0, 0.0],
):

    cra, cdec, crot, cscl = correction_applied

    radec_pid_k = guide_loop["radec"]["pid"]["k"]
    radec_pid_td = guide_loop["radec"]["pid"].get("Td", 0.0)
    radec_pid_ti = guide_loop["radec"]["pid"].get("Ti", 0.0)

    rot_pid_k = guide_loop["rot"]["pid"]["k"]
    rot_pid_td = guide_loop["rot"]["pid"].get("Td", 0.0)
    rot_pid_ti = guide_loop["rot"]["pid"].get("Ti", 0.0)

    if "scale" in guide_loop:
        scale_pid_k = guide_loop["scale"]["pid"]["k"]
        scale_pid_td = guide_loop["scale"]["pid"].get("Td", 0.0)
        scale_pid_ti = guide_loop["scale"]["pid"].get("Ti", 0.0)
    else:
        scale_pid_k = 0.0
        scale_pid_td = 0.0
        scale_pid_ti = 0.0

    # Update headers of proc images with deltas.
    for img in data:
        if img.proc_image is not None:
            hdus = fits.open(img.proc_image, mode="update")

            hdus[1].header["RADECK"] = (radec_pid_k, "PID K term for RA/Dec")
            hdus[1].header["RADECTD"] = (radec_pid_td, "PID Td term for RA/Dec")
            hdus[1].header["RADECTI"] = (radec_pid_ti, "PID Ti term for RA/Dec")

            hdus[1].header["ROTK"] = (rot_pid_k, "PID K term for Rot.")
            hdus[1].header["ROTTD"] = (rot_pid_td, "PID Td term for Rot.")
            hdus[1].header["ROTTI"] = (rot_pid_ti, "PID Ti term for Rot.")

            hdus[1].header["SCLK"] = (scale_pid_k, "PID K term for Scale")
            hdus[1].header["SCLTD"] = (scale_pid_td, "PID Td term for Scale")
            hdus[1].header["SCLTI"] = (scale_pid_ti, "PID Ti term for Scale")

            hdus[1].header["RMS"] = (rms, "Guide RMS [arcsec]")
            hdus[1].header["CAPPLIED"] = (applied, "Guide correction applied?")

            hdus[1].header["DELTARA"] = (delta_ra, "RA measured delta [arcsec]")
            hdus[1].header["DELTADEC"] = (delta_dec, "Dec measured delta [arcsec]")
            hdus[1].header["DELTAROT"] = (delta_rot, "Rotator measured delta [arcsec]")
            hdus[1].header["DELTASCL"] = (delta_scale, "Scale measured factor")

            hdus[1].header["CORR_RA"] = (cra, "RA applied correction [arcsec]")
            hdus[1].header["CORR_DEC"] = (cdec, "Dec applied correction [arcsec]")
            hdus[1].header["CORR_ROT"] = (crot, "Rotator applied correction [arcsec]")
            hdus[1].header["CORR_SCL"] = (cscl, "Scale applied correction")

            hdus.close()
