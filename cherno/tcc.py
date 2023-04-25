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
    from clu.command import FakeCommand

    from cherno.actor import ChernoCommandType
    from cherno.guider import AxesPID


__all__ = ["apply_axes_correction"]


async def apply_axes_correction(
    command: ChernoCommandType | FakeCommand,
    pids: AxesPID,
    delta_radec: tuple[float, float] | numpy.ndarray | None = None,
    delta_rot: float | None = None,
    full: bool = False,
):
    """Send corrections to the TCC. Corrections here are in arcsec."""

    assert command.actor

    state = command.actor.state

    guide_loop = state.guide_loop
    enabled_axes = state.enabled_axes

    # Correction applied in ra, dec, rot, scale, in arcsec.
    correction_applied = [0.0, 0.0, 0.0, 0.0]

    if delta_radec is not None:
        corr_radec = [0.0, 0.0]

        for ax_idx, ax in enumerate(["ra", "dec"]):
            if ax not in enabled_axes or float(delta_radec[ax_idx]) == -999.0:
                continue

            if full:
                corr_ax = -float(delta_radec[ax_idx])
            else:
                # This returns the correction (i.e., opposite sign).
                corr_ax = getattr(pids, ax)(float(delta_radec[ax_idx])) or 0.0

            min_corr_arcsec = guide_loop[ax]["min_correction"]
            max_corr_arcsec = guide_loop[ax]["max_correction"]

            if numpy.abs(corr_ax) < min_corr_arcsec:
                command.debug(f"Skipping small {ax.lower()} correction.")
                corr_ax = 0.0
            elif numpy.abs(corr_ax) > max_corr_arcsec:
                raise ChernoError(f"{ax.upper()} correction too large: {corr_ax:.2f}.")

            corr_radec[ax_idx] = corr_ax / 3600

        if numpy.any(numpy.abs(corr_radec) > 0):
            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset arc {corr_radec[0]}, {corr_radec[1]} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying RA/Dec correction.")
                return correction_applied

            correction_applied[0] = float(numpy.round(corr_radec[0] * 3600.0, 3))
            correction_applied[1] = float(numpy.round(corr_radec[1] * 3600.0, 3))

    if delta_rot is not None and "rot" in enabled_axes and delta_rot != -999.0:
        if full:
            corr_rot = -delta_rot
        else:
            corr_rot = pids.rot(delta_rot) or 0.0

        min_corr_arcsec = guide_loop["rot"]["min_correction"]
        max_corr_arcsec = guide_loop["rot"]["max_correction"]

        if numpy.abs(corr_rot) < min_corr_arcsec:
            command.debug("Skipping small rotator correction.")
            corr_rot = 0.0
        elif numpy.abs(corr_rot) > max_corr_arcsec:
            raise ChernoError(f"Rotator correction too large: {corr_rot:.1f}.")

        corr_rot /= 3600

        if numpy.abs(corr_rot) > 0:
            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset guide 0.0, 0.0, {corr_rot} /computed",
            )

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying rotator correction.")
                return correction_applied

            correction_applied[2] = float(numpy.round(corr_rot * 3600.0, 3))

    return correction_applied


async def apply_focus_correction(
    command: ChernoCommandType | FakeCommand,
    pids: AxesPID,
    delta_focus: float,
):
    """Send focus corrections to the TCC. Corrections are in microns."""

    assert command.actor

    state = command.actor.state

    guide_loop = state.guide_loop
    enabled_axes = state.enabled_axes

    min_corr = guide_loop["focus"]["min_correction"]
    max_corr = guide_loop["focus"]["max_correction"]

    if "focus" not in enabled_axes or delta_focus == -999.0:
        return 0.0

    corr_focus = pids.focus(delta_focus) or 0.0

    if numpy.abs(corr_focus) < min_corr:
        command.debug("Skipping small focus correction.")
        return 0.0

    if numpy.abs(corr_focus) > max_corr:
        command.warning(f"Ignoring large focus correction: {corr_focus:.1f}.")
        return 0.0

    tcc_offset_cmd = await command.send_command(
        "tcc",
        f"set focus={corr_focus} /incremental",
    )

    if tcc_offset_cmd.status.did_fail:
        command.error("Failed applying focus correction.")
        return 0.0

    return numpy.round(corr_focus, 1)
