#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-09-13
# @Filename: astrometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import subprocess
import time

from typing import NamedTuple, Optional

from cherno.extraction import PathLike


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
            "which solve-field",
            shell=True,
            capture_output=True,
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
        files: PathLike | list[PathLike],
        shell: bool = True,
        stdout: Optional[PathLike] = None,
        stderr: Optional[PathLike] = None,
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

        args = self._build_command(list(map(str, files)), options=options)

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
