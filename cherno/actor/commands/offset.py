#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-13
# @Filename: offset.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import click

from cherno import config

from .. import ChernoCommandType, cherno_parser


__all__ = ["offset"]


@cherno_parser.command()
@click.argument("RA", type=float, required=False)
@click.argument("DEC", type=float, required=False)
@click.argument("PA", type=float, required=False, default=0.0)
@click.option(
    "--reset/--no-reset",
    default=False,
    help="Resets the PID loop if a guider loop is active.",
)
async def offset(
    command: ChernoCommandType,
    ra: float | None = None,
    dec: float | None = None,
    pa: float = 0.0,
    reset: bool = False,
):
    """Offsets the field boresight by RA/DEC/PA arcsec."""

    if ra is None and dec is None:
        command.warning("Resetting offset.")
        ra = 0.0
        dec = 0.0
        pa = 0.0
    elif ra is not None and dec is None:
        return command.fail(error="ra and dec are required.")

    assert ra is not None and dec is not None and pa is not None

    assert command.actor
    command.actor.state.offset = (ra, dec, pa)

    if reset:
        if command.actor.state._guider_obj is not None:
            pids = command.actor.state._guider_obj.pids

            pids.ra.reset()
            pids.dec.reset()
            pids.rot.reset()
            pids.focus.reset()

        command.warning("PID loop internals have been reset.")

    default_offset = config.get("default_offset", (0.0, 0.0, 0.0))

    return command.finish(offset=[ra, dec, pa], default_offset=default_offset)
