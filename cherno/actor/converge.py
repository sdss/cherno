#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-09-01
# @Filename: converge.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
from datetime import datetime

import click
import numpy

from clu.parsers.click import (
    cancel_command,
    get_current_command_name,
    get_running_tasks,
    timeout,
)

from . import ChernoCommandType, cherno_parser


__all__ = ["converge"]


@cherno_parser.command()
@click.argument("RMS", type=float)
@click.option(
    "-n",
    "--iterations",
    type=int,
    default=2,
    help="Number of guide loop iterations the RMS must be below the required level.",
)
@click.option(
    "--max-age",
    type=float,
    default=300,
    help="Maximum age, in seconds, of the RMS mesurements.",
)
@click.option(
    "--require-new",
    is_flag=True,
    help="Requires that the last RMS measurement is less than 5 seconds old.",
)
@timeout(600)
async def converge(
    command: ChernoCommandType,
    rms: float | None = None,
    iterations: int = 2,
    max_age: float = 300,
    require_new: bool = True,
):
    """Runs until the guide RMS has been below a given threshold for N iterations."""

    tasks = get_running_tasks(get_current_command_name())
    if tasks and len(tasks) > 1:
        command.warning("Another version of this command was running. Cancelling it.")
        cancel_command(keep_last=True)

    assert command.actor.store is not None

    while True:
        kws: list = command.actor.store.tail("guide_rms", n=iterations)

        kws = [k for k in kws if (datetime.now() - k.date).total_seconds() < max_age]
        values = numpy.array([k.value[-1] for k in kws if len(k.value) > 0])

        last_age = (datetime.now() - kws[-1].date).total_seconds()
        if (not require_new) or (require_new and last_age < 5):
            if len(values) >= iterations and numpy.all(values <= rms):
                return command.finish("RMS has been reached.")

        await asyncio.sleep(1)
