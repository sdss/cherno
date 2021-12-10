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

from cherno.actor.exposer import Exposer
from cherno.astrometry import process_and_correct
from cherno.exceptions import ExposerError

from . import cherno_parser


if TYPE_CHECKING:
    from . import ChernoCommandType


@cherno_parser.command()
@click.option(
    "-t",
    "--exposure-time",
    type=float,
    default=15.0,
    help="Cameras exposure time.",
)
@click.option(
    "-c",
    "--continuous",
    is_flag=True,
    help="Run acquisition in continuous mode.",
)
async def acquire(
    command: ChernoCommandType,
    exposure_time: float = 15.0,
    continuous: bool = False,
):
    """Runs the acquisition procedure."""

    exposer = Exposer(command, callback=process_and_correct)

    try:
        await exposer.loop(
            exposure_time,
            count=1 if continuous is False else None,
            timeout=25,
        )
    except ExposerError as err:
        return command.fail(f"Acquisition failed: {err}")

    return command.finish()
