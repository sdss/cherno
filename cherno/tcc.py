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
    k_radec: float | None = None,
    k_rot: float | None = None,
):
    """Send corrections to the TCC. Corrections here are in arcsec."""

    if radec is not None:
        corr_radec = numpy.array(radec) / 3600.0  # In degrees!

        default_k_radec: float = config["guide_loop"]["radec"]["pid"]["k"]
        k_radec = k_radec or default_k_radec

        min_corr_arcsec = config["guide_loop"]["radec"]["min_correction"]
        max_corr_arcsec = config["guide_loop"]["radec"]["max_correction"]

        if numpy.all(numpy.abs(corr_radec) < (min_corr_arcsec / 3600.0)):
            # Small correction. Do not apply.
            command.debug("Ignoring small ra/dec correction.")

        elif numpy.any(numpy.abs(corr_radec) > (max_corr_arcsec / 3600)):
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

        default_k_rot: float = config["guide_loop"]["rotation"]["pid"]["k"]
        k_rot = k_rot or default_k_rot

        min_corr_arcsec = config["guide_loop"]["rotation"]["min_correction"]
        max_corr_arcsec = config["guide_loop"]["rotation"]["max_correction"]

        if numpy.all(numpy.abs(corr_rot) < (min_corr_arcsec / 3600.0)):
            # Small correction. Do not apply.
            command.debug("Ignoring small rotator correction.")

        elif numpy.any(numpy.abs(corr_rot) > (max_corr_arcsec / 3600)):
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
