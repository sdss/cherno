#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-13
# @Filename: offset.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import click

from . import ChernoCommandType, cherno_parser


__all__ = ["offset"]


@cherno_parser.command()
@click.argument("RA", type=float)
@click.argument("DEC", type=float)
async def offset(command: ChernoCommandType, ra: float, dec: float):
    """Offsets the field boresight by RA/DEC arcsec."""

    assert command.actor
    command.actor.state.offset = (ra, dec)

    return command.finish()
