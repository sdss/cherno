#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: guide.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace

import click

from cherno import config
from cherno.acquisition import Acquisition
from cherno.actor.exposer import Exposer
from cherno.exceptions import ExposerError
from cherno.maskbits import GuiderStatus

from .. import ChernoCommandType, cherno_parser


__all__ = ["guide", "get_callback", "get_guide_common_params", "Params", "check_params"]


def get_guide_common_params(continuous: bool = True, full: bool = False):
    """Returns a customised list of parameters."""

    options = [
        click.Option(
            ["--exposure-time", "-t"],
            type=float,
            default=None,
            help="Cameras exposure time.",
        ),
        click.Option(
            ["--cameras", "-m"],
            type=str,
            help="Comma-separated cameras to expose.",
        ),
        click.Option(
            ["--count", "-n"],
            type=int,
            help="Number of acquisition iterations. Incompatible with --continuous.",
        ),
        click.Option(
            ["--apply/--no-apply"],
            default=True,
            help="Whether to apply the correction.",
        ),
        click.Option(
            ["--full/--no-full"],
            default=full,
            help="Applies the full correction. Otherwise corrections are weighted "
            "using the PID loop.",
        ),
        click.Option(
            ["--only-radec"],
            is_flag=True,
            help="Only fits and corrects RA/Dec. Rotation and scale "
            "corrections are calculated but not applied.",
        ),
        click.Option(
            ["--auto-radec-min"],
            type=int,
            default=config["acquisition"]["auto_radec_min"],
            help="Number of cameras solving below which only RA/Dec will be fit. "
            "-1 disables this feature and a full fit is allways performed.",
        ),
        click.Option(
            ["--wait", "-w"],
            type=float,
            help="Time to wait between iterations.",
        ),
        click.Option(
            ["--mode"],
            type=click.Choice(
                ["hybrid", "astrometrynet", "gaia"],
                case_sensitive=False,
            ),
            default="hybrid",
            help="Solving mode. Hybrid uses astrometry.net first and "
            "Gaia for the cameras not solved.",
        ),
        click.Option(
            ["--gaia-max-mag"],
            type=float,
            help="Maximum Gaia magnitude to query.",
        ),
        click.Option(
            ["--cross-match-blur"],
            type=float,
            help="Blur sigma for cross-correlation",
        ),
        click.Option(
            ["--fit-all-detections/--no-fit-all-detections"],
            default=config["acquisition"]["fit_all_detections"],
            help="Perform fit using all detected sources. Otherwise uses only "
            "the centre of each solved camera.",
        ),
        click.Option(
            ["--continuous/--no-continuous"],
            default=continuous,
            help="Run acquisition in continuous mode.",
        ),
    ]

    return sorted(options, key=lambda opt: opt.name or "")


class Params(SimpleNamespace):
    command: ChernoCommandType
    exposure_time: float | None = None
    continuous: bool = False
    count: int | None = None
    apply: bool = True
    full: bool = False
    only_radec: bool = False
    auto_radec_min: int = config["acquisition"]["auto_radec_min"]
    cameras: str | None = None
    wait: float | None = None
    mode: str | None = None
    gaia_max_mag: float | None = None
    cross_match_blur: float | None = None
    fit_all_detections: bool = True
    mode_kwargs: dict[str, bool] = {}
    names: list[str] | None = None


def check_params(params: Params):
    """Checks the parameters. Fails the command if an error is found."""

    command = params.command

    if params.count is not None and params.continuous is True:
        command.fail("--count and --continuous are incompatible.")
        return False

    _exposure_loop = command.actor.state._exposure_loop
    if _exposure_loop is not None and not _exposure_loop.done():
        command.warning("An active Exposer loop was found. Cancelling it.")
        _exposure_loop.cancel()

    # Inplace update parameters.

    params.count = params.count or 1

    if params.mode == "hybrid":
        params.mode_kwargs = {"use_astrometry_net": True, "use_gaia": True}
    elif params.mode == "astrometrynet":
        params.mode_kwargs = {"use_astrometry_net": True, "use_gaia": False}
    elif params.mode == "gaia":
        params.mode_kwargs = {"use_astrometry_net": False, "use_gaia": True}
    else:
        params.mode_kwargs = {}

    if params.cameras is not None:
        params.names = list(params.cameras.split(","))

    if params.exposure_time is not None:
        if params.exposure_time < 1.0:
            command.fail("Exposure time not set or too small.")
            return False
        else:
            command.actor.state.exposure_time = params.exposure_time

    return True


def get_callback(params: Params):
    """Returns the Acquisition.process() callback."""

    acquisition = Acquisition(config["observatory"])
    params.command.actor.state._acquisition_obj = acquisition  # To update PID coeffs.

    return partial(
        acquisition.process,
        correct=params.apply,
        full_correction=params.full,
        wait_for_correction=(params.wait is None),
        only_radec=params.only_radec,
        auto_radec_min=params.auto_radec_min,
        gaia_phot_g_mean_mag_max=params.gaia_max_mag,
        gaia_cross_correlation_blur=params.cross_match_blur,
        fit_all_detections=params.fit_all_detections,
        **params.mode_kwargs,
    )


async def _guide(params: Params):
    """Actually run the guide loop."""

    command = params.command

    if not check_params(params):
        # Already failed
        return

    callback = get_callback(params)
    exposer = Exposer(command, callback=callback)

    command.actor.state._exposure_loop = asyncio.create_task(
        exposer.loop(
            None,
            count=params.count if params.continuous is False else None,
            timeout=25,
            names=params.names,
            delay=params.wait or 0.0,
        )
    )

    try:
        await command.actor.state._exposure_loop
    except ExposerError as err:
        return command.fail(f"Acquisition failed: {err}")

    return command.finish()


@cherno_parser.command()
async def guide(**kwargs):
    """Runs the guiding loop."""

    params = Params(**kwargs)

    return await _guide(params)