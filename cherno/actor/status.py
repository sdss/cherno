#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-19
# @Filename: status.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from cherno import config

from . import ChernoCommandType, cherno_parser


__all__ = ["status"]


@cherno_parser.command()
async def status(command: ChernoCommandType):
    """Reports the status of the system."""

    command.info(guider_status=hex(command.actor.state.status.value))
    command.info(enabled_axes=command.actor.state.enabled_axes)

    default_offset = config.get("default_offset", (0.0, 0.0, 0.0))
    command.info(default_offset=default_offset)
    command.info(offset=command.actor.state.offset)

    for axis in ["ra", "dec", "rot", "focus"]:
        command.info(
            message={
                f"pid_{axis}": list(
                    command.actor.state.guide_loop[axis]["pid"].values()
                )
            }
        )

    return command.finish()
