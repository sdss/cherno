#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-01-29
# @Filename: __main__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio
import os
import pathlib
import signal
from contextlib import suppress

import click
from click_default_group import DefaultGroup

from clu.tools import cli_coro
from sdsstools.daemonizer import DaemonGroup

from cherno.actor.actor import ChernoActor
from cherno.astrometry import extract_and_run


async def shutdown(signal, loop, actor):
    """Cancel tasks, including run_forever()."""

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    with suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks)


@click.group(cls=DefaultGroup, default="actor", default_if_no_args=True)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Debug mode. Use additional v for more details.",
)
@click.pass_context
def cherno(ctx, verbose):
    """Cherno CLI."""

    ctx.obj = {"verbose": verbose}


@cherno.group(cls=DaemonGroup, prog="cherno_actor", workdir=os.getcwd())
@click.pass_context
@cli_coro
async def actor(ctx):
    """Runs the actor."""

    config_file = os.path.join(os.path.dirname(__file__), "etc/cherno.yml")

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


@cherno.command()
@click.argument("IMAGE", type=click.Path(exists=True, dir_okay=False))
@click.argument("OUTDIR", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--cpulimit", default=15, type=float, help="astrometry.net time limit.")
@click.option("--npix", default=50, type=int, help="Minimum number of pixels.")
@click.option("--sigma", default=10.0, type=float, help="Minimum SNR.")
@cli_coro
async def reprocess(
    image: str,
    outdir: str,
    cpulimit: float = 30.0,
    npix: int = 50,
    sigma: float = 10.0,
):
    """Reprocess an image."""

    path = pathlib.Path(str(outdir))
    if path.is_relative_to("/data/gcam"):
        raise ValueError("Output directory cannot be in /data/gcam.")

    if not path.exists():
        path.mkdir(parents=True)

    data = await extract_and_run(
        [image],
        proc_image_outdir=outdir,
        sigma=sigma,
        min_npix=npix,
        cpulimit=cpulimit,
        overwrite=True,
        plot=False,
    )

    print(data)


if __name__ == "__main__":
    cherno()
