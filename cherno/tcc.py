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

        if numpy.any(corr_radec > 60 / 3600):
            command.warning("RA/Dec correction > 1 arcmin. Not applying correction.")

        else:
            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset arc {corr_radec[0]},{corr_radec[1]} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying RA/Dec correction.")
                return False

    if rot is not None:
        corr_rot = numpy.array(rot) / 3600.0
        corr_rot *= k_rot

        if numpy.any(corr_rot > 1):
            command.warning("Rotator correction > 1 degree. Not applying correction.")

        else:
            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset rotator {corr_rot} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying rotator correction.")
                return False

    return True
