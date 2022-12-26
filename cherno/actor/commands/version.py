#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-09-11
# @Filename: version.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from coordio import __version__ as coordio_version

from cherno import __version__ as cherno_version

from .. import ChernoCommandType, cherno_parser


__all__ = ["version"]


@cherno_parser.command()
async def version(command: ChernoCommandType):
    """Prints the current version of cherno and dependencies."""

    command.info(version=cherno_version)
    command.info(coordio_version=coordio_version)

    try:
        from astrometry import __version__ as astrometrynet_version  # type: ignore

        command.info(astrometrynet_version=astrometrynet_version)
    except ImportError:
        command.warning("Cannot find astrometry.net version.")

    try:
        from fps_calibrations import get_version

        command.info(fps_calibrations_version=get_version())
    except ImportError:
        command.warning("Cannot find fps_calibrations version.")

    return command.finish()
