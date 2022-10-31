#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-06
# @Filename: acquire.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
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
@click.option(
    "--only-radec",
    is_flag=True,
    help="Only fits and corrects RA/Dec. Rotation and scale values are informative.",
)
@click.option(
    "--auto-radec-min",
    type=int,
    default=config["acquisition"]["auto_radec_min"],
    help="Number of cameras solving below which only RA/Dec will be fit. "
    "-1 disables this feature and a full fit is allways performed.",
)
@click.option(
    "-w",
    "--wait",
    type=float,
    help="Time to wait between iterations.",
)
@click.option(
    "--no-block",
    is_flag=True,
    help="Returns immediatly after starting the acquisition loop.",
)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "astrometrynet", "gaia"], case_sensitive=False),
    default="hybrid",
    help="Solving mode. Hybrid uses astrometry.net first and Gaia for the cameras "
    "not solved.",
)
@click.option(
    "--gaia-max-mag",
    type=float,
    help="Maximum Gaia magnitude to query.",
)
@click.option(
    "--cross-match-blur",
    type=float,
    help="Blur sigma for cross-correlation",
)
@click.option(
    "--fit-all-detections/--no-fit-all-detections",
    default=config["acquisition"]["fit_all_detections"],
    help="Perform fit using all detected sources. Otherwise uses only "
    "the centre of each solved camera.",
)
async def acquire(
    command: ChernoCommandType,
    exposure_time: float | None = None,
    continuous: bool = False,
    count: int | None = None,
    apply: bool = True,
    full: bool = False,
    only_radec: bool = False,
    auto_radec_min: int = config["acquisition"]["auto_radec_min"],
    cameras: str | None = None,
    wait: float | None = None,
    no_block: bool = False,
    mode: str | None = None,
    gaia_max_mag: float | None = None,
    cross_match_blur: float | None = None,
    fit_all_detections: bool = True,
):
    """Runs the acquisition procedure."""

    assert command.actor

    if count is not None and continuous is True:
        return command.fail("--count and --continuous are incompatible.")

    _exposure_loop = command.actor.state._exposure_loop
    if _exposure_loop is not None and not _exposure_loop.done():
        command.warning("An active Exposer loop was found. Cancelling it.")
        _exposure_loop.cancel()

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
    command.actor.state._acquisition_obj = acquisition  # To update PID coeffs.

    if mode == "hybrid":
        mode_kwargs = {"use_astrometry_net": True, "use_gaia": True}
    elif mode == "astrometrynet":
        mode_kwargs = {"use_astrometry_net": True, "use_gaia": False}
    elif mode == "gaia":
        mode_kwargs = {"use_astrometry_net": False, "use_gaia": True}
    else:
        mode_kwargs = {}

    callback = partial(
        acquisition.process,
        correct=apply,
        full_correction=full,
        wait_for_correction=(wait is None),
        only_radec=only_radec,
        auto_radec_min=auto_radec_min,
        gaia_phot_g_mean_mag_max=gaia_max_mag,
        gaia_cross_correlation_blur=cross_match_blur,
        fit_all_detections=fit_all_detections,
        **mode_kwargs,
    )
    exposer = Exposer(command, callback=callback)

    command.actor.state._exposure_loop = asyncio.create_task(
        exposer.loop(
            None,
            count=count if continuous is False else None,
            timeout=25,
            names=names,
            delay=wait or 0.0,
        )
    )

    if not no_block:
        try:
            await command.actor.state._exposure_loop
        except ExposerError as err:
            return command.fail(f"Acquisition failed: {err}")
    else:
        return command.finish("Finishing due to --no-block. The guide loop is running.")

    return command.finish()
