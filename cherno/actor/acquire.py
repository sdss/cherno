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
    default=15.0,
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
    "--cpulimit",
    type=float,
    default=15.0,
    help="Maximum runtime for astrometry.net.",
)
async def acquire(
    command: ChernoCommandType,
    exposure_time: float = 15.0,
    continuous: bool = False,
    apply: bool = True,
    plot: bool = False,
    cpulimit: float = 15.0,
):
    """Runs the acquisition procedure."""

    callback = partial(
        process_and_correct,
        run_options={"plot": plot, "cpulimit": cpulimit},
        apply=apply,
    )
    exposer = Exposer(command, callback=callback)

    try:
        await exposer.loop(
            exposure_time,
            count=1 if continuous is False else None,
            timeout=25,
        )
    except ExposerError as err:
        return command.fail(f"Acquisition failed: {err}")

    return command.finish()
