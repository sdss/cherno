#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-28
# @Filename: scale.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import time

import click
import numpy

from . import ChernoCommandType, cherno_parser


__all__ = ["get_scale"]


@cherno_parser.command(name="get-scale")
@click.option(
    "--max-age",
    type=int,
    default=1200,
    help="Maximum age of the scale data points to consider, in seconds.",
)
@click.option(
    "--min-points",
    type=int,
    default=10,
    help="Minimum number of points to use.",
)
@click.option(
    "--sigma",
    type=float,
    default=3,
    help="Sigma-clip rejection.",
)
async def get_scale(
    command: ChernoCommandType,
    max_age: int = 1200,
    min_points: int = 10,
    sigma: float = 3.0,
):
    """Outputs the median, sigma-clipped scale from existing measurements.

    This command is mainly called by jaeger when determining what scale correction
    to apply to a new configuration.

    """

    data = numpy.array(command.actor.state.scale_history)

    # Require at least
    if len(data) == 0:
        command.warning("Not enough points to calculate median scale.")
        return command.finish(scale_median=-999.0)

    # Make first column the delat wrt now.
    data[:, 0] = time.time() - data[:, 0]

    valid = data[data[:, 0] < max_age]
    if len(valid) < min_points:
        command.warning("Not enough points to calculate median scale.")
        return command.finish(scale_median=-999.0)

    if len(valid[valid[:, 1] < 900]) > 10:
        valid = data[data[:, 0] < 900]

    scales = valid[:, 1]

    median0 = numpy.median(scales)
    std = numpy.std(scales)

    clipped = scales[numpy.abs(scales - median0) <= sigma * std]

    if len(clipped) < 3:
        command.warning("Too few data points after sigma clipping.")
        return command.finish(scale_median=-999.0)

    return command.finish(scale_median=numpy.round(numpy.median(clipped), 7))
