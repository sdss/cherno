#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-07
# @Filename: acquisition.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from functools import partial

from typing import TYPE_CHECKING

import numpy
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning

from clu.command import FakeCommand

from cherno import config, log
from cherno.astrometry import AstrometryNet
from cherno.coordinates import gfa_to_radec
from cherno.exceptions import ChernoError
from cherno.extraction import Extraction, ExtractionData, PathLike
from cherno.maskbits import GuiderStatus
from cherno.tcc import apply_correction
from cherno.utils import astrometry_fit, run_in_executor


if TYPE_CHECKING:
    from cherno.actor import ChernoCommandType
    from cherno.actor.actor import ChernoState


warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)


__all__ = ["Acquisition"]


@dataclass
class AcquisitionData:
    """Data from the acquisition process."""

    extraction_data: ExtractionData
    solved: bool = False
    solve_time: float = -999.0
    wcs: WCS = field(default_factory=WCS)
    camera_racen: float = -999.0
    camera_deccen: float = -999.0
    xrot: float = -999.0
    yrot: float = -999.0
    rotation: float = -999.0
    proc_image: pathlib.Path | None = None

    def __post_init__(self):

        self.path = self.extraction_data.path
        self.camera = self.extraction_data.camera
        self.exposure_no = self.extraction_data.exposure_no
        self.obstime = self.extraction_data.obstime
        self.observatory = self.extraction_data.observatory
        self.field_ra = self.extraction_data.field_ra
        self.field_dec = self.extraction_data.field_dec
        self.field_pa = self.extraction_data.field_pa

    @property
    def e_data(self):
        """Shortcut for ``extraction_data``."""

        return self.extraction_data


@dataclass
class AstrometricSolution:
    """Astrometric solution data."""

    valid_solution: bool
    acquisition_data: list[AcquisitionData]
    delta_ra: float = -999.0
    delta_dec: float = -999.0
    delta_rot: float = -999.0
    delta_scale: float = -999.0
    rms: float = -999.0
    fwhm: float = -999.0
    camera_rotation: float = -999.0
    correction_applied: list[float] = field(default_factory=list)


class Acquisition:
    """Runs the steps for field acquisition."""

    def __init__(
        self,
        extractor: Extraction | None = None,
        astrometry: AstrometryNet | None = None,
        command: ChernoCommandType | None = None,
        astrometry_params: dict = {},
        extraction_params: dict = {},
    ):

        self.extractor = extractor or Extraction(**extraction_params)

        if astrometry is not None:
            self.astrometry = astrometry
        else:
            pixel_scale = config["cameras"]["pixel_scale"]
            backend_config = pathlib.Path(__file__).parent / "etc/astrometrynet.cfg"
            self.astrometry = AstrometryNet(
                backend_config=str(backend_config),
                width=2048,
                height=2048,
                no_plots=True,
                scale_low=pixel_scale * 0.9,
                scale_high=pixel_scale * 1.1,
                scale_units="arcsecperpix",
                sort_column="flux",
                radius=0.5,
                cpulimit=config["acquisition"]["cpulimit"],
                **astrometry_params,
            )

        self.command = command or FakeCommand(log)

    def set_command(self, command: ChernoCommandType):
        """Sets the command."""

        self.command = command

    async def process(
        self,
        command: ChernoCommandType | None,
        images: PathLike | list[PathLike],
        write_proc: bool = True,
        overwrite: bool = False,
        correct: bool = True,
        full_correction: bool = False,
        offset: list[float] | None = None,
    ):
        """Performs extraction and astrometry."""

        if command is not None:
            self.set_command(command)

        if not isinstance(images, (list, tuple)):
            images = [images]

        ext_data: list[ExtractionData] = await run_in_executor(
            self.extractor.multiprocess,
            images,
        )

        for d in ext_data:
            if d.nvalid < 5:
                self.command.warning(f"Camera {d.camera}: not enough sources.")
            else:
                self.command.info(
                    fwhm_camera=[
                        d.camera,
                        d.exposure_no,
                        d.fwhm_median,
                        d.nregions,
                        d.nvalid,
                    ]
                )

        acq_data = await asyncio.gather(*[self._astrometry_one(d) for d in ext_data])

        if write_proc:
            await asyncio.gather(
                *[self.write_proc_image(d, overwrite=overwrite) for d in acq_data]
            )

        ast_solution = await self.fit(list(acq_data), offset=offset)

        if correct and ast_solution.valid_solution is True:
            await self.correct(ast_solution, full=full_correction)

        if self.command.actor:
            update_proc_headers(ast_solution, self.command.actor.state)

        return ast_solution

    async def fit(self, data: list[AcquisitionData], offset: list[float] | None = None):
        """Calculate the astrometric solution."""

        ast_solution = AstrometricSolution(
            False,
            data,
            correction_applied=[0.0, 0.0, 0.0],
        )

        solved = sorted([d for d in data if d.solved is True], key=lambda x: x.camera)
        weights = [s.extraction_data.nvalid for s in solved]

        if len(solved) == 0:
            self.command.error(acquisition_valid=False, did_correct=False)
            return ast_solution

        fwhm = numpy.average([d.e_data.fwhm_median for d in solved], weights=weights)
        camera_rotation = numpy.average([d.rotation for d in solved], weights=weights)

        ast_solution.fwhm = float(fwhm)
        ast_solution.camera_rotation = float(camera_rotation)

        if solved[0].field_ra == "NaN" or isinstance(solved[0].field_ra, str):
            self.command.error(acquisition_valid=False, did_correct=False)
            self.command.error("Field not defined. Cannot run astrometric fit.")
            return ast_solution

        if offset is None:
            if self.command.actor:
                offset = list(self.command.actor.state.offset)
            else:
                offset = list(config.get("offset", [0.0, 0.0, 0.0]))

        if any(offset):
            self.command.warning(offset=offset)
        else:
            self.command.debug(offset=offset)

        fit = astrometry_fit(solved, offset=offset, obstime=solved[0].obstime.jd)

        exp_no = solved[0].exposure_no  # Should be the same for all.

        if fit is False:
            rms = delta_ra = delta_dec = delta_rot = delta_scale = -999.0
        else:
            delta_ra = fit[0]
            delta_dec = fit[1]
            delta_rot = fit[2]
            delta_scale = fit[3]

            xrms = fit[4]
            yrms = fit[5]
            rms = fit[6]

            self.command.info(guide_rms=[exp_no, xrms, yrms, rms])

        self.command.info(
            astrometry_fit=[
                exp_no,
                len(solved),
                -999.0,
                -999.0,
                numpy.round(fwhm, 2),
                -999,
                numpy.round(camera_rotation, 3),
                delta_ra,
                delta_dec,
                delta_rot,
                delta_scale,
            ]
        )

        if delta_scale > 0 and self.command.actor:
            # If we measured the scale, add it to the actor state. This is later
            # used to compute the average scale over a period. We also add the time
            # because we'll want to reject measurements that are too old.
            self.command.actor.state.scale_history.append((time.time(), delta_scale))

        ast_solution.delta_ra = delta_ra
        ast_solution.delta_dec = delta_dec
        ast_solution.delta_rot = delta_rot
        ast_solution.delta_scale = delta_scale
        ast_solution.rms = rms

        return ast_solution

    async def correct(self, data: AstrometricSolution, full: bool = False):
        """Runs the astrometric fit"""

        if not self.command.actor or self.command.status.is_done:
            raise ChernoError("Cannot run correct without a valid running command.")

        actor_state = self.command.actor.state
        guider_status = actor_state.status

        stopping = (
            guider_status & (GuiderStatus.STOPPING | GuiderStatus.IDLE)
        ).value > 0
        will_apply = stopping is False

        correction_applied: list[float] = [0.0, 0.0, 0.0, 0.0]

        if will_apply is True:
            self.command.info("Applying corrections.")

            min_isolated = actor_state.guide_loop["rot"]["min_isolated_correction"]
            if abs(data.delta_rot) >= min_isolated:
                self.command.debug("Applying only large rotator correction.")
                correction_applied = await apply_correction(
                    self.command,
                    rot=-data.delta_rot,
                    k_rot=None if full is False else 1.0,
                )

            else:

                correction_applied = await apply_correction(
                    self.command,
                    rot=-data.delta_rot,
                    radec=(-data.delta_ra, -data.delta_dec),
                    k_radec=None if full is False else 1.0,
                    k_rot=None if full is False else 1.0,
                )

            self.command.info(
                acquisition_valid=True,
                did_correct=any(correction_applied),
                correction_applied=correction_applied,
            )

        else:
            self.command.info(acquisition_valid=True, did_correct=False)

        data.correction_applied = correction_applied

        return True

    async def _astrometry_one(self, ext_data: ExtractionData):

        regions = ext_data.regions

        if self.astrometry._options.get("dir", None) is None:
            astrometry_dir = pathlib.Path(config["acquisition"]["astrometry_dir"])
            if astrometry_dir.is_absolute():
                pass
            else:
                astrometry_dir = ext_data.path.parent / astrometry_dir

            astrometry_dir.mkdir(exist_ok=True, parents=True)
        else:
            astrometry_dir = pathlib.Path(self.astrometry._options["dir"])

        outfile_root = astrometry_dir / ext_data.path.stem

        if "x_0" in regions:
            xyls_df = regions.loc[regions.valid == 1, ["x_0", "y_0", "flux_0"]].copy()
            xyls_df = xyls_df.rename(columns={"x_0": "x", "y_0": "y", "flux_0": "flux"})
        else:
            xyls_df = regions.loc[regions.valid == 1, ["x", "y", "flux"]]

        gfa_xyls = Table.from_pandas(xyls_df)
        gfa_xyls_path = outfile_root.with_suffix(".xyls")
        gfa_xyls.write(str(gfa_xyls_path), format="fits", overwrite=True)

        wcs_output = outfile_root.with_suffix(".wcs")
        wcs_output.unlink(missing_ok=True)

        camera_id = int(ext_data.camera[-1])
        radec_centre = gfa_to_radec(
            1024,
            1024,
            camera_id,
            ext_data.field_ra,
            ext_data.field_dec,
            position_angle=ext_data.field_pa,
            site_name=ext_data.observatory,
        )

        proc = await self.astrometry.run(
            [gfa_xyls_path],
            stdout=outfile_root.with_suffix(".stdout"),
            stderr=outfile_root.with_suffix(".stderr"),
            ra=radec_centre[0],
            dec=radec_centre[1],
        )

        acq_data = AcquisitionData(ext_data, solve_time=proc.elapsed)

        if wcs_output.exists():
            acq_data.solved = True
            wcs = WCS(open(wcs_output).read())
            acq_data.wcs = wcs

            racen, deccen = wcs.pixel_to_world_values([[1024, 1024]])[0]
            acq_data.camera_racen = numpy.round(racen, 6)
            acq_data.camera_deccen = numpy.round(deccen, 6)

            # TODO: consider parallactic angle here.
            cd: numpy.ndarray = wcs.wcs.cd
            rot_rad = numpy.arctan2(
                numpy.array([cd[0, 1], cd[0, 0]]),
                numpy.array([cd[1, 1], cd[1, 0]]),
            )

            # Rotation is from N to E to the x and y axes of the GFA.
            yrot, xrot = numpy.rad2deg(rot_rad)
            acq_data.xrot = numpy.round(xrot % 360.0, 3)
            acq_data.yrot = numpy.round(yrot % 360.0, 3)

            # Calculate field rotation.
            camera_rot = config["cameras"]["rotation"][acq_data.camera]
            rotation = numpy.array([xrot - camera_rot - 90, yrot - camera_rot]) % 360
            rotation[rotation > 180.0] -= 360.0  # type: ignore
            rotation = numpy.mean(rotation)
            acq_data.rotation = numpy.round(rotation, 3)

        self.command.info(
            camera_solution=[
                acq_data.camera,
                acq_data.exposure_no,
                acq_data.solved,
                acq_data.camera_racen,
                acq_data.camera_deccen,
                acq_data.xrot,
                acq_data.yrot,
                acq_data.rotation,
            ]
        )

        return acq_data

    async def write_proc_image(
        self,
        acq_data: AcquisitionData,
        outpath: PathLike = None,
        overwrite: bool = False,
    ):
        """Writes the proc-gimg image."""

        ext_data = acq_data.extraction_data

        proc_hdu = fits.open(str(acq_data.path)).copy()

        rec = Table.from_pandas(acq_data.extraction_data.regions).as_array()
        proc_hdu.append(fits.BinTableHDU(rec, name="CENTROIDS"))

        proc_hdu[1].header["SOLVED"] = acq_data.solved
        proc_hdu[1].header["SOLVTIME"] = (
            acq_data.solve_time,
            "Time to solve the field or fail",
        )
        proc_hdu[1].header["FWHM"] = (ext_data.fwhm_median, "Average FWHM in arcsec")

        if self.command.actor is not None:
            offsets = self.command.actor.state.offset
        else:
            offsets = [-999.0] * 3

        proc_hdu[1].header["OFFRA"] = (offsets[0], "Offset in RA [arcsec]")
        proc_hdu[1].header["OFFDEC"] = (offsets[1], "Offset in Dec [arcsec]")
        proc_hdu[1].header["OFFPA"] = (offsets[2], "Offset in PA [arcsec]")

        proc_hdu[1].header.update(acq_data.wcs.to_header())

        proc_path: pathlib.Path
        if outpath is not None:
            proc_path = pathlib.Path(outpath)
            if proc_path.is_dir():
                proc_path = proc_path / ("proc-" + ext_data.path.name)
        else:
            proc_path = ext_data.path.parent / ("proc-" + ext_data.path.name)

        loop = asyncio.get_running_loop()
        func = partial(
            proc_hdu.writeto,
            proc_path,
            overwrite=overwrite,
            output_verify="silentfix",
        )
        await loop.run_in_executor(None, func)

        acq_data.proc_image = proc_path


def update_proc_headers(data: AstrometricSolution, guider_state: ChernoState):

    guide_loop = guider_state.guide_loop

    enabled_axes = guider_state.enabled_axes
    enabled_radec = "radec" in enabled_axes
    enabled_rot = "rot" in enabled_axes

    cra, cdec, crot, cscl = data.correction_applied

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

    rms = data.rms
    delta_ra = data.delta_ra
    delta_dec = data.delta_dec
    delta_rot = data.delta_rot
    delta_scale = data.delta_scale

    # Update headers of proc images with deltas.
    for a_data in data.acquisition_data:
        if a_data.proc_image is not None:
            hdus = fits.open(str(a_data.proc_image), mode="update")

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

            hdus[1].header["E_RADEC"] = (enabled_radec, "RA/Dec corrections enabled?")
            hdus[1].header["E_ROT"] = (enabled_rot, "Rotator corrections enabled?")
            hdus[1].header["E_FOCUS"] = (False, "Focus corrections enabled?")
            hdus[1].header["E_SCL"] = (False, "Scale corrections enabled?")

            hdus[1].header["DELTARA"] = (delta_ra, "RA measured delta [arcsec]")
            hdus[1].header["DELTADEC"] = (delta_dec, "Dec measured delta [arcsec]")
            hdus[1].header["DELTAROT"] = (delta_rot, "Rotator measured delta [arcsec]")
            hdus[1].header["DELTASCL"] = (delta_scale, "Scale measured factor")

            hdus[1].header["CORR_RA"] = (cra, "RA applied correction [arcsec]")
            hdus[1].header["CORR_DEC"] = (cdec, "Dec applied correction [arcsec]")
            hdus[1].header["CORR_ROT"] = (crot, "Rotator applied correction [arcsec]")
            hdus[1].header["CORR_SCL"] = (cscl, "Scale applied correction")

            hdus.close()
