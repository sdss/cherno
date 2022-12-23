#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-06
# @Filename: acquire.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

import click

from cherno.actor.commands.guide import (
    Params,
    check_params,
    get_callback,
    get_guide_common_params,
)
from cherno.actor.exposer import Exposer
from cherno.exceptions import ExposerError

from .. import cherno_parser


__all__ = ["acquire"]


acquire_params = get_guide_common_params(continuous=False, full=True)


@cherno_parser.command(params=acquire_params)
@click.option("--test", is_flag=True)
async def acquire(**kwargs):
    """Runs the acquisition procedure."""

    params = Params(**kwargs)
    command = params.command

    if not check_params(params):
        # Already failed
        return

    callback = get_callback(params)
    exposer = Exposer(command, callback=callback)

    command.actor.state._exposure_loop = asyncio.create_task(
        exposer.loop(
            None,
            count=params.count if params.continuous is False else None,
            timeout=25,
            names=params.names,
            delay=params.wait or 0.0,
        )
    )

    try:
        await command.actor.state._exposure_loop
    except ExposerError as err:
        return command.fail(f"Acquisition failed: {err}")

    return command.finish()
