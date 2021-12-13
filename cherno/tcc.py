#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-12
# @Filename: tcc.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy


if TYPE_CHECKING:
    from cherno.actor import ChernoCommandType


__all__ = ["apply_correction"]


async def apply_correction(
    command: ChernoCommandType,
    radec: tuple[float, float] | numpy.ndarray | None = None,
    rot: float | None = None,
    k_radec=0.6,
    k_rot=0.5,
):
    """Send corrections to the"""

    if radec is not None:
        corr_radec = numpy.array(radec) / 3600.0  # In degrees!
        corr_radec *= k_radec

        if numpy.any(corr_radec > 30 / 3600.0):
            command.error("RA/Dec correction > 30 arcsec. Not applying correction.")
            return False

        tcc_offset_cmd = await command.send_command(
            "tcc",
            f"offset arc {corr_radec[0]},{corr_radec[1]} /computed",
        )

        if tcc_offset_cmd.status.did_fail:
            command.error("Failed applying RA/Dec correction.")
            return False

    if rot is not None:
        corr_rot = numpy.array(rot)
        corr_rot *= k_rot

        if numpy.any(corr_rot > 60 / 3600.0):
            command.error("Rotator correction > 60 arcsec. Not applying correction.")
            return False

        tcc_offset_cmd = await command.send_command(
            "tcc",
            f"offset rotator {corr_rot} /computed",
        )

        if tcc_offset_cmd.status.did_fail:
            command.error("Failed applying rotator correction.")
            return False

    return True