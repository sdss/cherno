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
from astropy.stats.sigma_clipping import SigmaClip

from .. import ChernoCommandType, cherno_parser


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

    if len(data) == 0:
        command.warning("Not enough points to calculate median scale.")
        return command.finish(scale_median=-999.0)

    # Make first column the delta wrt now.
    data[:, 0] = time.time() - data[:, 0]

    valid = data[data[:, 0] < max_age]
    if len(valid) < min_points:
        command.warning("Not enough points to calculate median scale.")
        return command.finish(scale_median=-999.0)

    # Generally prefer data from the last exposure.
    if len(valid[valid[:, 0] < 900]) > 10:
        valid = data[data[:, 0] < 900]

    # Sigma-clip entries with scale outliers.
    sc = SigmaClip(sigma)
    mask = ~sc(valid[:, 1]).mask
    valid = valid[mask]

    wmean = numpy.average(valid[:, 1], weights=1.0 / valid[:, 0])

    if wmean < 0.999 or wmean > 1.001:
        command.warning(
            f"Unexpectedly large/small scale factor {wmean:.6f}. "
            "This is unexpected. If you are sure, use this value manually."
        )
        return command.finish(scale_median=-999.0)

    return command.finish(scale_median=numpy.round(wmean, 7))
