#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-01-29
# @Filename: __main__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress

import click
from click_default_group import DefaultGroup

from clu.tools import cli_coro
from sdsstools.daemonizer import DaemonGroup

from cherno.actor.actor import ChernoActor


async def shutdown(signal, loop, actor):
    """Cancel tasks, including run_forever()."""

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    with suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks)


@click.group(cls=DefaultGroup, default="actor", default_if_no_args=True)
@click.option(
    "-o",
    "--observatory",
    type=str,
    help="Observatory configuration to use. Defaults to using $OBSERVATORY.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Debug mode. Use additional v for more details.",
)
@click.pass_context
def cherno(ctx, verbose: bool = False, observatory: str | None = None):
    """Cherno CLI."""

    ctx.obj = {"verbose": verbose, "observatory": observatory}


@cherno.group(cls=DaemonGroup, prog="cherno_actor", workdir=os.getcwd())
@click.pass_context
@cli_coro
async def actor(ctx):
    """Runs the actor."""

    observatory = ctx.obj["observatory"] or os.environ["OBSERVATORY"]

    config_file = os.path.join(
        os.path.dirname(__file__),
        f"etc/cherno_{observatory}.yml",
    )

    cherno_actor = ChernoActor.from_config(config_file)

    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s,
            lambda s=s: asyncio.create_task(shutdown(s, loop, cherno_actor)),
        )

    try:
        await cherno_actor.start()
        await cherno_actor.run_forever()
    except asyncio.CancelledError:
        pass
    finally:
        await cherno_actor.stop()
        loop.stop()


if __name__ == "__main__":
    cherno()
