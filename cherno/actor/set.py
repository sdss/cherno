#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-14
# @Filename: set.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import click

from . import ChernoCommandType, cherno_parser


__all__ = ["set"]


@cherno_parser.command()
@click.argument("OPTIONS", type=str, nargs=-1)
async def set(command: ChernoCommandType, options: tuple[str, ...]):
    """Sets guiding parameters.

    Valid parameters are:
        exposure-time [EXPTIME]
        pid radec|rot k [VALUE]
        axes [radec rot off]
        cameras CAMERAS

    """

    assert command.actor

    if len(options) == 0:
        return command.fail("No options passed")

    elif options[0] == "exposure-time":
        if len(options) < 2:
            return command.fail("Invalid number of parameters")

        exp_time = float(options[1])
        command.actor.state.exposure_time = exp_time

    elif options[0] == "pid":
        if len(options) < 4:
            return command.fail("Invalid number of parameters")

        axis = options[1]
        component = options[2]
        value = float(options[3])

        if axis not in ["radec", "rot", "focus"]:
            return command.fail(f"Invalid axis {axis}.")

        if value <= 0 or value > 1:
            return command.fail("Invalid value. Must be between 0 and 1.")

        command.info(message={f"pid_{axis}": [value]})
        command.actor.state.guide_loop[axis]["pid"][component] = value

    elif options[0] == "axes":

        if len(options) == 1 or (len(options) == 2 and options[1] in ["off", "none"]):
            axes = []
        else:
            axes = options[1:]

        command.actor.state.enabled_axes = list(axes)
        command.info(enabled_axes=command.actor.state.enabled_axes)

    elif options[0] == "cameras":
        if len(options) < 2:
            return command.fail("Invalid number of parameters")

        cameras = options[1:]
        command.actor.state.enabled_cameras = list(cameras)

    else:
        return command.fail("Invalid parameter.")

    return command.finish()
