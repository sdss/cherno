#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-06
# @Filename: acquire.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from functools import partial

from typing import TYPE_CHECKING

import click

from cherno.actor.exposer import Exposer
from cherno.astrometry import process_and_correct
from cherno.exceptions import ExposerError

from . import cherno_parser


if TYPE_CHECKING:
    from . import ChernoCommandType


__all__ = ["acquire"]


@cherno_parser.command()
@click.option(
    "-t",
    "--exposure-time",
    type=float,
    default=None,
    help="Cameras exposure time.",
)
@click.option(
    "-c",
    "--continuous",
    is_flag=True,
    help="Run acquisition in continuous mode.",
)
@click.option(
    "--apply/--no-apply",
    default=True,
    help="Whether to apply the correction.",
)
@click.option(
    "--plot",
    is_flag=True,
    help="Whether to plot results of astrometry.net.",
)
@click.option(
    "-f",
    "--full",
    is_flag=True,
    help="Applies the full correction once. Cannot be used with --continuous.",
)
async def acquire(
    command: ChernoCommandType,
    exposure_time: float | None = None,
    continuous: bool = False,
    apply: bool = True,
    plot: bool = False,
    full: bool = False,
):
    """Runs the acquisition procedure."""

    assert command.actor

    if exposure_time is not None:
        if exposure_time < 1.0:
            return command.fail("Exposure time not set or too small.")
        else:
            command.actor.state.exposure_time = exposure_time

    callback = partial(process_and_correct, apply=apply, full=full)
    exposer = Exposer(command, callback=callback)

    try:
        await exposer.loop(
            None,
            count=1 if continuous is False else None,
            timeout=25,
        )
    except ExposerError as err:
        return command.fail(f"Acquisition failed: {err}")

    return command.finish()
