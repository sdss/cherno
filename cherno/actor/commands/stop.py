#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-12
# @Filename: stop.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from cherno.maskbits import GuiderStatus

from .. import ChernoCommandType, cherno_parser


__all__ = ["stop"]


@cherno_parser.command()
async def stop(command: ChernoCommandType):
    """Stops any running guide loops."""

    assert command.actor is not None

    if command.actor.state.status & GuiderStatus.IDLE:
        command.warning("The guider is idle.")
        return command.finish()

    command.actor.state.set_status(GuiderStatus.STOPPING, mode="add")

    return command.finish("The guide loop is stopping.")
