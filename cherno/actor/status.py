#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-19
# @Filename: status.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from . import ChernoCommandType, cherno_parser


__all__ = ["status"]


@cherno_parser.command()
async def status(command: ChernoCommandType):
    """Reports the status of the system."""

    command.info(guider_status=hex(command.actor.state.status.value))
    command.info(enabled_axes=command.actor.state.enabled_axes)

    return command.finish()
