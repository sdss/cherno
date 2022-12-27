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

from cherno.actor.commands.guide import GuideParams, _guide, get_guide_common_params

from .. import cherno_parser


if TYPE_CHECKING:
    from .. import ChernoCommandType


__all__ = ["acquire"]


def stop_acquisition(command: ChernoCommandType, target: float):
    """Determines whether the acquisition should be stopped."""

    rms_history = command.actor.state.rms_history

    if len(rms_history) == 0:
        return False

    return rms_history[-1] <= target


acquire_params = get_guide_common_params(continuous=False, full=True)


@cherno_parser.command(params=acquire_params)
@click.option(
    "--target-rms",
    "-r",
    type=float,
    help="RMS at which to stop the acquisition process.",
)
@click.option(
    "--max-iterations",
    "-x",
    type=int,
    help="Maximum number of iterations before failing.",
)
async def acquire(
    command: ChernoCommandType,
    target_rms: float | None = None,
    max_iterations: int | None = None,
    **kwargs,
):
    """Runs the acquisition procedure."""

    params = GuideParams(command=command, **kwargs)

    if target_rms is not None:
        params.continuous = True
        stop_condition = partial(stop_acquisition, params.command, target_rms)
    else:
        stop_condition = None

    return await _guide(
        params,
        stop_condition=stop_condition,
        target_rms=target_rms,
        max_iterations=max_iterations,
    )
