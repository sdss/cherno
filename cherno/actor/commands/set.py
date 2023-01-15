#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-14
# @Filename: set.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import click

from .. import ChernoCommandType, cherno_parser


__all__ = ["set"]


@cherno_parser.command()
@click.argument("OPTIONS", type=str, nargs=-1)
async def set(command: ChernoCommandType, options: tuple[str, ...]):
    """Sets guiding parameters.

    Valid parameters are:
        exposure-time [EXPTIME]
        pid ra|dec|rot|focus k|ti|td [VALUE]
        axes [ra dec rot off]
        cameras CAMERAS
        odds 1-10

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
        term = options[2].lower()
        value = float(options[3])

        if axis not in ["ra", "dec", "rot", "focus"]:
            return command.fail(f"Invalid axis {axis}.")

        if term not in ["k", "ti", "td"]:
            return command.fail(f"Invalid term {term}.")

        command.actor.state.guide_loop[axis]["pid"][term] = value

        command.info(
            message={
                f"pid_{axis}": [
                    command.actor.state.guide_loop[axis]["pid"][t]
                    for t in ["k", "ti", "td"]
                ]
            }
        )

        if command.actor.state._guider_obj:
            # This is what actually changes the PID loop during an exposure.
            pid_attr = getattr(command.actor.state._guider_obj.pids, axis)
            if term == "k":
                pid_attr.Kp = value
            elif term == "ti":
                pid_attr.Ki = value
            elif term == "td":
                pid_attr.Kd = value
            pid_attr.reset()

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

    elif options[0] == "odds":
        if len(options) != 2:
            return command.fail("Invalid number of parameters")

        odds = int(options[1])
        if odds < 1 or odds > 10:
            return command.fail("Invalid odds value. Valid range is 1-10.")

        command.actor.state.astrometry_net_odds = odds

    else:
        return command.fail("Invalid parameter.")

    return command.finish()
