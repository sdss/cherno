#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-12-30
# @Filename: reprocess.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import click

from cherno import config
from cherno.guider import Guider

from .. import ChernoCommandType, cherno_parser


__all__ = ["reprocess"]


@cherno_parser.command()
@click.argument("MJD", type=int)
@click.argument("FRAME", type=int)
async def reprocess(command: ChernoCommandType, mjd: int, frame: int):
    """Reprocesses a frame (does not apply corrections."""

    paths = list(pathlib.Path(f"/data/gcam/{mjd}/").glob(f"gimg-*{frame:04d}.fits"))
    images = [str(pp) for pp in paths if pp.exists()]

    if len(images) == 0:
        return command.fail("No images found.")

    guider = Guider(config["observatory"])
    await guider.process(command, images, overwrite=True, correct=False)

    return command.finish()
