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
import subprocess
import time
from functools import partial

from typing import Any, NamedTuple, Optional, Union, cast

import pandas
import sep
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from cherno.actor import ChernoCommandType


PathType = Union[str, pathlib.Path]


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
            cmd_str = args[0]

        else:
            cmd = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            cmd_str = " ".join(args)

        stdout_bytes, stderr_bytes = await cmd.communicate()
        if cmd.returncode and cmd.returncode > 0:
            raise subprocess.CalledProcessError(
                cmd.returncode,
                cmd=cmd_str,
                output=stdout_bytes,
                stderr=stderr_bytes,
            )

        elapsed = time.time() - t0

        if stdout:
            with open(stdout, "wb") as out:
                out.write(" ".join(args).encode() + b"\n")
                out.write(stdout_bytes)

        if stderr:
            with open(stderr, "wb") as err:
                err.write(stderr_bytes)

        return TimedProcess(cmd, elapsed)


async def extract_and_run(
    images: pathlib.Path | str | list[pathlib.Path] | list[str],
    astrometry_outdir: PathType = "astrometry",
    proc_image_outdir: PathType | None = None,
    sigma: float = 10.0,
    min_npix: int = 50,
) -> list[fits.Header | None]:
    """Extracts sources and runs Astrometry.net.

    Returns the WCS object for each processed image or `None` if failed.

    """

    if isinstance(images, (list, tuple)):
        results = await asyncio.gather(
            *[
                extract_and_run(
                    image,
                    astrometry_outdir,
                    sigma=sigma,
                    min_npix=min_npix,
                )
                for image in images
            ]
        )
        return [result[0] for result in results]

    assert isinstance(images, (pathlib.Path, str))
    image = images

    path = pathlib.Path(image)
    mjd = path.parts[-2]
    dirname = path.parent
    proc_basename = "proc-" + path.parts[-1]

    if os.path.isabs(astrometry_outdir):
        astrometry_outdir = os.path.join(astrometry_outdir, mjd)
    else:
        astrometry_outdir = os.path.join(dirname, astrometry_outdir)

    if not os.path.exists(astrometry_outdir):
        os.makedirs(astrometry_outdir)

    data: Any = fits.getdata(image)
    back = sep.Background(data.astype("int32"))

    regions = pandas.DataFrame(
        sep.extract(
            data - back.back(),
            sigma,
            err=back.globalrms,
        )
    )
    regions.loc[regions.npix > min_npix, "valid"] = 1
    regions.to_hdf(os.path.join(astrometry_outdir, proc_basename + ".hdf"), "data")

    if len(regions) < 5:  # Don't even try.
        return [None]

    gfa_xyls = Table.from_pandas(regions.loc[:, ["x", "y"]])
    gfa_xyls_file = os.path.join(astrometry_outdir, proc_basename + ".xyls")
    gfa_xyls.write(gfa_xyls_file, format="fits", overwrite=True)

    header = fits.getheader(image, 1)

    pixel_scale = 0.216

    backend_config = os.path.join(os.path.dirname(__file__), "../etc/astrometrynet.cfg")
    astrometry_net = AstrometryNet()
    astrometry_net.configure(
        backend_config=backend_config,
        width=2048,
        height=2048,
        no_plots=True,
        scale_low=pixel_scale * 0.9,
        scale_high=pixel_scale * 1.1,
        scale_units="arcsecperpix",
        radius=2.0,
        dir=astrometry_outdir,
    )

    wcs_output = os.path.join(astrometry_outdir, proc_basename + ".wcs")
    if os.path.exists(wcs_output):
        os.remove(wcs_output)

    proc = await astrometry_net.run(
        [gfa_xyls_file],
        stdout=os.path.join(astrometry_outdir, proc_basename + ".stdout"),
        stderr=os.path.join(astrometry_outdir, proc_basename + ".stderr"),
        ra=header["RA"],
        dec=header["DEC"],
    )

    proc_hdu = fits.open(image).copy()

    if not os.path.exists(wcs_output):
        proc_hdu[1].header["SOLVED"] = False
        proc_hdu[1].header["SOLVTIME"] = proc.elapsed
        wcs = None
    else:
        proc_hdu[1].header["SOLVED"] = True
        proc_hdu[1].header["SOLVTIME"] = proc.elapsed
        wcs = WCS(open(wcs_output).read())
        proc_hdu[1].header.update(wcs.to_header())

    loop = asyncio.get_running_loop()
    func = partial(
        proc_hdu.writeto,
        os.path.join(proc_image_outdir or dirname, "proc-" + proc_basename),
        overwrite=True,
    )
    await loop.run_in_executor(None, func)

    return [proc_hdu[1].header]


async def process_and_correct(command: ChernoCommandType, filenames: list[str]):
    """Processes a series of files for the same pointing and applies a correction."""

    # Create instance of AstrometryNet
    headers = await extract_and_run(filenames)

    if not any(headers):
        return command.fail(acquisition_valid=0)

    for header in headers:
        if header is None:
            continue

        camera = header["CAMNAME"]

        if header["SOLVED"] is False:
            command.info(acquisition_data=[camera, False, -999.0, -999.0])
        else:
            wcs = WCS(header)
            ra, dec = wcs.pixel_to_world_values([[1024, 1024]])[0]
            command.info(acquisition_data=[camera, True, ra, dec])
