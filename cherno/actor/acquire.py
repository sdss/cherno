#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-06
# @Filename: acquire.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import click
from astropy.wcs import WCS

from cherno.astrometry import extract_and_run

from . import cherno_parser


if TYPE_CHECKING:
    from . import ChernoCommandType


@cherno_parser.command()
@click.option("-t", "--exposure-time", type=float, default=15.0)
async def acquire(command: ChernoCommandType, exposure_time: float = 15.0):
    """Runs the acquisition procedure."""

    command.info("Exposing cameras.")

    cmd = await command.send_command("fliswarm", f"talk -c gfa expose {exposure_time}")
    if cmd.status.did_fail:
        return command.fail("Failed exposing camera.")

    filename_bundle = []
    for reply in cmd.replies:
        for keyword in reply.keywords:
            if keyword.name == "filename_bundle":
                filename_bundle = [value.native for value in keyword.values]

    if len(filename_bundle) == 0:
        return command.fail("filename_bundle not output.")

    # Create instance of AstrometryNet
    headers = await extract_and_run(filename_bundle, "/data/astrometrynet/")

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

    return command.finish()
