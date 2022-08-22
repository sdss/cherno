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


__all__ = ["apply_correction_lco"]


async def apply_correction_lco(
    command: ChernoCommandType,
    radec: tuple[float, float] | numpy.ndarray | None = None,
    rot: float | None = None,
    focus: float | None = None,
    k_ra: float | None = None,
    k_dec: float | None = None,
    k_rot: float | None = None,
    k_focus: float | None = None,
):
    """Send corrections to the LCOTCC. Corrections here are in arcsec."""

    assert command.actor

    state = command.actor.state

    guide_loop = state.guide_loop
    enabled_axes = state.enabled_axes

    # Correction applied in ra, dec, rot, scale, and focus.
    no_correction = correction_applied = [0.0, 0.0, 0.0, 0.0, 0.0]

    corr_radec = numpy.array([0.0, 0.0])
    corr_rot: float = 0.0
    corr_focus: float = 0.0

    if radec is not None:

        for ax_idx, ax in enumerate(["ra", "dec"]):
            if ax in enabled_axes:
                corr_ax = float(numpy.array(radec[ax_idx]) / 3600.0)  # In degrees!

                default_k_ax: float = guide_loop[ax]["pid"]["k"]

                if ax == "ra":
                    k_ax = k_ra or default_k_ax
                else:
                    k_ax = k_dec or default_k_ax

                min_corr_arcsec = guide_loop[ax]["min_correction"]
                max_corr_arcsec = guide_loop[ax]["max_correction"]

                if numpy.all(numpy.abs(corr_ax) < (min_corr_arcsec / 3600.0)):
                    # Small correction. Do not apply.
                    command.debug(f"Ignoring small {ax.upper()} correction.")

                elif numpy.any(numpy.abs(corr_ax) > (max_corr_arcsec / 3600)):
                    raise ChernoError(
                        f"{ax.upper()} correction too large. "
                        "Not applying correction."
                    )

                else:
                    corr_ax *= k_ax
                    corr_radec[ax_idx] = corr_ax

    if rot is not None and "rot" in enabled_axes:
        corr_rot = float(numpy.array(rot) / 3600.0)  # In degrees!

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

            correction_applied[2] = numpy.round(corr_rot * 3600.0, 3)

    if focus is not None and "focus" in enabled_axes:

        min_corr = guide_loop["focus"]["min_correction"]
        max_corr = guide_loop["focus"]["max_correction"]

        if numpy.abs(focus) < min_corr:
            command.debug("Ignoring small focus correction.")
            focus_corr = 0.0

        elif numpy.abs(focus) > max_corr:
            command.warning("Ignoring large focus correction.")
            focus_corr = 0.0

        else:
            default_k_focus: float = guide_loop["focus"]["pid"]["k"]
            k_focus = k_focus or default_k_focus

            focus_corr = focus * k_focus

        correction_applied[4] = numpy.round(focus_corr, 1)

    if (
        numpy.any(numpy.abs(corr_radec) > 0)
        or numpy.abs(corr_rot) > 0
        or numpy.abs(corr_focus) > 0
    ):
        tcc_offset_cmd = await command.send_command(
            "lcotcc",
            f"guideoffset {corr_radec[0]},{corr_radec[1]},{corr_rot},{corr_focus}",
        )

        if tcc_offset_cmd.status.did_fail:
            command.error("Failed applying RA/Dec correction.")
            return no_correction

    return no_correction
