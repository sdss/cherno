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

from cherno import config
from cherno.acquisition import Acquisition
from cherno.actor.exposer import Exposer
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
    "-m",
    "--cameras",
    type=str,
    help="Comma-separated cameras to expose.",
)
@click.option(
    "-c",
    "--continuous",
    is_flag=True,
    help="Run acquisition in continuous mode.",
)
@click.option(
    "-n",
    "--count",
    type=int,
    help="Number of acquisition iterations. Incompatible with --continuous.",
)
@click.option(
    "--apply/--no-apply",
    default=True,
    help="Whether to apply the correction.",
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
    count: int | None = None,
    apply: bool = True,
    full: bool = False,
    cameras: str | None = None,
):
    """Runs the acquisition procedure."""

    assert command.actor

    if count is not None and continuous is True:
        return command.fail("--count and --continuous are incompatible.")

    count = count or 1

    if cameras is None:
        names = None
    else:
        names = list(cameras.split(","))

    if exposure_time is not None:
        if exposure_time < 1.0:
            return command.fail("Exposure time not set or too small.")
        else:
            command.actor.state.exposure_time = exposure_time

    acquisition = Acquisition(config["observatory"])

    callback = partial(
        acquisition.process,
        correct=apply,
        full_correction=full,
        scale_rms=True,
    )
    exposer = Exposer(command, callback=callback)

    try:
        await exposer.loop(
            None,
            count=count if continuous is False else None,
            timeout=25,
            names=names,
        )
    except ExposerError as err:
        return command.fail(f"Acquisition failed: {err}")

    return command.finish()
