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

from cherno.exceptions import ChernoError


if TYPE_CHECKING:
    from cherno.actor import ChernoCommandType


__all__ = ["apply_axes_correction"]


async def apply_axes_correction(
    command: ChernoCommandType,
    radec: tuple[float, float] | numpy.ndarray | None = None,
    rot: float | None = None,
    k_radec: float | None = None,
    k_rot: float | None = None,
):
    """Send corrections to the TCC. Corrections here are in arcsec."""

    assert command.actor

    state = command.actor.state

    guide_loop = state.guide_loop
    enabled_axes = state.enabled_axes

    # Correction applied in ra, dec, rot, scale, in arcsec.
    correction_applied = [0.0, 0.0, 0.0, 0.0]

    if radec is not None and "radec" in enabled_axes:
        corr_radec = numpy.array(radec) / 3600.0  # In degrees!

        default_k_radec: float = guide_loop["radec"]["pid"]["k"]
        k_radec = k_radec or default_k_radec

        min_corr_arcsec = guide_loop["radec"]["min_correction"]
        max_corr_arcsec = guide_loop["radec"]["max_correction"]

        if numpy.all(numpy.abs(corr_radec) < (min_corr_arcsec / 3600.0)):
            # Small correction. Do not apply.
            command.debug("Ignoring small ra/dec correction.")

        elif numpy.any(numpy.abs(corr_radec) > (max_corr_arcsec / 3600)):
            raise ChernoError("RA/Dec correction too large. Not applying correction.")

        else:
            corr_radec *= k_radec

            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset arc {corr_radec[0]}, {corr_radec[1]} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying RA/Dec correction.")
                return correction_applied

            correction_applied[0] = numpy.round(corr_radec[0] * 3600.0, 3)
            correction_applied[1] = numpy.round(corr_radec[1] * 3600.0, 3)

    if rot is not None and "rot" in enabled_axes:
        corr_rot = numpy.array(rot) / 3600.0  # In degrees!

        default_k_rot: float = guide_loop["rot"]["pid"]["k"]
        k_rot = k_rot or default_k_rot

        min_corr_arcsec = guide_loop["rot"]["min_correction"]
        max_corr_arcsec = guide_loop["rot"]["max_correction"]

        if numpy.all(numpy.abs(corr_rot) < (min_corr_arcsec / 3600.0)):
            # Small correction. Do not apply.
            command.debug("Ignoring small rotator correction.")

        elif numpy.any(numpy.abs(corr_rot) > (max_corr_arcsec / 3600)):
            raise ChernoError("Rotator correction too large. Not applying correction.")

        else:
            corr_rot *= k_rot

            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset guide 0.0, 0.0, {corr_rot} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying rotator correction.")
                return correction_applied

            correction_applied[2] = numpy.round(corr_rot * 3600.0, 3)

    return correction_applied


async def apply_focus_correction(
    command: ChernoCommandType,
    focus_corr: float,
    k_focus: float | None = None,
):
    """Send focus corrections to the TCC. Corrections are in microns."""

    assert command.actor

    state = command.actor.state

    guide_loop = state.guide_loop
    enabled_axes = state.enabled_axes

    min_corr = guide_loop["focus"]["min_correction"]
    max_corr = guide_loop["focus"]["max_correction"]

    if "focus" not in enabled_axes:
        return

    if numpy.abs(focus_corr) < min_corr:
        command.debug("Ignoring small focus correction.")
        return

    if numpy.abs(focus_corr) > max_corr:
        command.warning("Ignoring large focus correction.")
        return

    default_k_focus: float = guide_loop["focus"]["pid"]["k"]
    k_focus = k_focus or default_k_focus

    focus_corr = focus_corr * k_focus

    tcc_offset_cmd = await command.send_command(
        "tcc",
        f"set focus={focus_corr} /incremental",
    )

    if tcc_offset_cmd.status.did_fail:
        command.error("Failed applying focus correction.")
        return

    return numpy.round(focus_corr, 1)
