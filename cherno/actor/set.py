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

    """

    assert command.actor

    if len(options) == 0:
        return command.fail("No options passed")

    elif options[0] == "exposure-time":
        if len(options) < 2:
            return command.fail("Invalid number of parameters")

        exp_time = float(options[1])
        command.actor.state.exposure_time = exp_time

    else:
        return command.fail("Invalid parameter.")

    return command.finish()
