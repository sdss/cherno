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

from cherno import config


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

        min_corr_arcsec = config["guide_loop"]["radec"]["min_correction"]
        max_corr_arcsec = config["guide_loop"]["radec"]["max_correction"]

        if numpy.all(corr_radec < min_corr_arcsec / 3600.0):
            # Small correction. Do not apply.
            command.debug("Ignoring small ra/dec correction.")

        elif numpy.any(corr_radec > max_corr_arcsec / 3600):
            command.warning("RA/Dec correction too large. Not applying correction.")

        else:
            corr_radec *= k_radec

            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset arc {corr_radec[0]}, {corr_radec[1]} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying RA/Dec correction.")
                return False

    if rot is not None:
        corr_rot = numpy.array(rot) / 3600.0  # In degrees!

        min_corr_arcsec = config["guide_loop"]["rotation"]["min_correction"]
        max_corr_arcsec = config["guide_loop"]["rotation"]["max_correction"]

        if numpy.all(corr_rot < min_corr_arcsec / 3600.0):
            # Small correction. Do not apply.
            command.debug("Ignoring small rotator correction.")

        elif numpy.any(corr_rot > max_corr_arcsec / 3600):
            command.warning("Rotator correction too large. Not applying correction.")

        else:
            corr_rot *= k_rot

            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset guide 0.0, 0.0, {corr_rot} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying rotator correction.")
                return False

    return True
