#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: guide.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import click

from cherno.maskbits import GuiderStatus

from . import ChernoCommandType, cherno_parser


__all__ = ["guide"]


@cherno_parser.command()
@click.option(
    "-o",
    "--off",
    is_flag=True,
    help="Turns guiding off.",
)
@click.option(
    "-t",
    "--timeout",
    type=float,
    default=5.0,
    help="Extra time to wait for an exposure to be done before failiing.",
)
async def guide(command: ChernoCommandType, off: bool = False, timeout: float = 5.0):
    """Starts the guiding loop."""

    assert command.actor

    state = command.actor.state

    if off:
        if state.status & GuiderStatus.STOPPING:
            return command.fail(error="Guider is already stopping.")
        elif not (state.status & GuiderStatus.EXPOSING):
            return command.fail(error="Guider is not exposing.")

    command.info("Starting guider loop.")

    while True:
        if state.status & (GuiderStatus.STOPPING | GuiderStatus.FAILED):
            return command.finish("Finishing guider loop.")
