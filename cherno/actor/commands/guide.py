#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: guide.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

from typing import Callable

import click

from cherno import config
from cherno.actor.exposer import Exposer
from cherno.exceptions import ExposerError
from cherno.guider import Guider
from cherno.maskbits import GuiderStatus

from .. import ChernoCommandType, cherno_parser


__all__ = [
    "guide",
    "get_callback",
    "get_guide_common_params",
    "GuideParams",
    "check_params",
]


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
            ["--max-exposure-time", "-T"],
            type=float,
            default=30,
            help="Maximum exposure time.",
        ),
        click.Option(
            ["--dynamic-exposure-time", "-d"],
            is_flag=True,
            help="Increases the exposure time dynamically if failed to solve.",
        ),
        click.Option(
            ["--cameras", "-m"],
            type=str,
            help="Comma-separated cameras to expose.",
        ),
        click.Option(
            ["--count", "-n"],
            type=int,
            help="Number of guider iterations. Incompatible with --continuous.",
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
            default=config["guider"]["auto_radec_min"],
            help="Number of cameras solving below which only RA/Dec will be fit. "
            "-1 disables this feature and a full fit is allways performed.",
        ),
        click.Option(
            ["--wait", "-w"],
            type=float,
            default=15 if config["observatory"] == "LCO" else None,
            show_default=True,
            help="Time to wait between iterations. Only applies to LCO.",
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
            default=config["guider"]["fit_all_detections"],
            help="Perform fit using all detected sources. Otherwise uses only "
            "the centre of each solved camera.",
        ),
        click.Option(
            ["--continuous/--no-continuous"],
            default=continuous,
            help="Run acquisition in continuous mode.",
        ),
        click.Option(
            ["--plot/--no-plot"],
            default=None,
            help="Produce plots during acquisition/guiding.",
        ),
    ]

    return sorted(options, key=lambda opt: opt.name or "")


class GuideParams(SimpleNamespace):
    command: ChernoCommandType
    exposure_time: float | None = None
    max_exposure_time: float = 30.0
    dynamic_exposure_time: bool = False
    continuous: bool = False
    count: int | None = None
    apply: bool = True
    full: bool = False
    only_radec: bool = False
    auto_radec_min: int = config["guider"]["auto_radec_min"]
    cameras: str | None = None
    wait: float | None = None
    mode: str | None = None
    gaia_max_mag: float | None = None
    cross_match_blur: float | None = None
    fit_all_detections: bool = True
    plot: bool | None = None
    mode_kwargs: dict[str, bool] = {}
    names: list[str] | None = None


async def check_params(params: GuideParams):
    """Checks the parameters. Fails the command if an error is found."""

    command = params.command

    if params.count is not None and params.continuous is True:
        command.fail("--count and --continuous are incompatible.")
        return False

    _exposure_loop = command.actor.state._exposure_loop
    if _exposure_loop is not None and not _exposure_loop.done():
        command.warning("An active Exposer loop was found. Cancelling it.")
        _exposure_loop.cancel()
        with suppress(asyncio.CancelledError):
            await _exposure_loop
        command.actor.state.set_status(GuiderStatus.IDLE, mode="override")

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


def get_callback(
    params: GuideParams,
    target_rms: float | None = None,
):
    """Returns the Guide.process() callback."""

    guider = Guider(
        config["observatory"],
        target_rms=target_rms,
        command=params.command,
    )
    params.command.actor.state._guider_obj = guider  # To update PID coeffs.

    async def process_callback(command: ChernoCommandType, images: list[str]):
        ast_solution = await guider.process(
            command,
            images,
            correct=params.apply,
            full_correction=params.full,
            offset=None,
            wait_for_correction=False,  # This only applies to LCO.
            only_radec=params.only_radec,
            auto_radec_min=params.auto_radec_min,
            gaia_phot_g_mean_mag_max=params.gaia_max_mag,
            gaia_cross_correlation_blur=params.cross_match_blur,
            fit_all_detections=params.fit_all_detections,
            plot=params.plot,
            fit_focus=config["guider"].get("fit_focus", True),
            **params.mode_kwargs,
        )

        if params.dynamic_exposure_time:
            max_exposure_time = params.max_exposure_time or 30.0
            current_exposure_time = command.actor.state.exposure_time
            if current_exposure_time >= max_exposure_time:
                # This should only happen if the observer manually set the
                # exposure time, in which case we accept it.
                new_exposure_time = current_exposure_time
            elif not ast_solution.valid_solution:
                new_exposure_time = min(current_exposure_time * 2, max_exposure_time)
            elif ast_solution.n_solved < 4:
                new_exposure_time = min(current_exposure_time * 1.5, max_exposure_time)
            else:
                new_exposure_time = current_exposure_time

            if new_exposure_time != round(current_exposure_time, 1):
                command.actor.state.exposure_time = new_exposure_time
                command.info(f"Exposure time updated to {new_exposure_time:.1f} s.")

        if command.actor.observatory == "LCO":
            rot_correction = ast_solution.correction_applied[2]
            if abs(rot_correction) > 0:
                # Do not add a delay if we applied a rotator correction. The
                # rotator blocks until done, which means the RA/Dec corrections
                # will probably have converged as well.
                pass
            elif (
                any(ast_solution.correction_applied)
                and params.wait is not None
                and params.wait > 0.0
            ):
                command.debug(f"Waiting {params.wait:.1f} s for corrections.")
                await asyncio.sleep(params.wait)

        return ast_solution

    return process_callback


async def _guide(
    params: GuideParams,
    stop_condition: Callable[[], bool] | None = None,
    target_rms: float | None = None,
    max_iterations: int | None = None,
):
    """Actually run the guide loop."""

    command = params.command

    if not await check_params(params):
        # Already failed
        return

    callback = get_callback(params, target_rms=target_rms)
    exposer = Exposer(command, callback=callback)

    command.actor.state._exposure_loop = asyncio.create_task(
        exposer.loop(
            None,
            count=params.count if params.continuous is False else None,
            max_iterations=max_iterations,
            timeout=25,
            names=params.names,
            delay=0.0,
            stop_condition=stop_condition,
        )
    )

    try:
        await command.actor.state._exposure_loop
    except ExposerError as err:
        command.actor.log.exception("Guider loop failed with error:")
        return command.fail(f"Guider failed: {err}")

    if stop_condition is not None and exposer.stop_reached:
        command.info("Stop condition has been reached.")

    return command.finish()


guide_params = get_guide_common_params()


@cherno_parser.command(params=guide_params)
async def guide(command: ChernoCommandType, **kwargs):
    """Runs the guiding loop."""

    params = GuideParams(command=command, **kwargs)

    return await _guide(params)
