#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-16
# @Filename: show.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from pprint import pformat

from . import ChernoCommandType, cherno_parser


__all__ = ["show"]


@cherno_parser.command()
async def show(command: ChernoCommandType):
    """Shows current configuration options."""

    assert command.actor

    for field in command.actor.state.__dataclass_fields__:
        if field in ["actor", "camera_state"]:
            continue
        lines = pformat(getattr(command.actor.state, field)).splitlines()
        for nline, line in enumerate(lines):
            if nline == 0:
                command.info(text=f"{field}: {line}")
            else:
                spaces = " " * len(field)
                command.info(text=f"{spaces}  {line}")

    return command.finish()
