#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-19
# @Filename: status.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import warnings

from cherno import config
from cherno.exceptions import ChernoUserWarning

from .. import ChernoCommandType, cherno_parser


__all__ = ["status"]


def get_astrometrynet_paths():
    """Returns a list of astrometry.net index paths that cherno will use."""

    astrometry_net_config = config["guider"]["astrometry_net_config"]
    backend_config = pathlib.Path(__file__).parents[2] / astrometry_net_config

    if not backend_config.exists():
        warnings.warn(
            "Cannot find astrometry.net backend configuration. "
            "This is a critical issue.",
            ChernoUserWarning,
        )
        return []

    index_paths: list[str] = []
    for line in open(backend_config, "r").readlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith("add_path"):
            index_paths.append(line.split()[1])

    return index_paths


@cherno_parser.command()
async def status(command: ChernoCommandType):
    """Reports the status of the system."""

    command.info(guider_status=hex(command.actor.state.status.value))
    command.info(enabled_axes=command.actor.state.enabled_axes)

    command.info(astrometrynet_index_paths=get_astrometrynet_paths())

    default_offset = config.get("default_offset", (0.0, 0.0, 0.0))
    command.info(default_offset=default_offset)
    command.info(offset=command.actor.state.offset)

    for axis in ["ra", "dec", "rot", "focus"]:
        command.info(
            message={
                f"pid_{axis}": list(
                    command.actor.state.guide_loop[axis]["pid"].values()
                )
            }
        )

    return command.finish()
