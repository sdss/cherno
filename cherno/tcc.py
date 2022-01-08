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

from cherno.maskbits import GuiderStatus


if TYPE_CHECKING:
    from cherno.actor import ChernoCommandType


__all__ = ["apply_correction"]


def _set_correcting(command: ChernoCommandType, mode: bool):
    """Sets/unsets the CORRECTING flag."""

    current = command.actor.state.status

    if mode is True:
        new = current | GuiderStatus.CORRECTING
    else:
        new = current & ~GuiderStatus.CORRECTING

    command.actor.state.set_status(new)


async def apply_correction(
    command: ChernoCommandType,
    radec: tuple[float, float] | numpy.ndarray | None = None,
    rot: float | None = None,
    k_radec: float | None = None,
    k_rot: float | None = None,
):
    """Send corrections to the TCC. Corrections here are in arcsec."""

    assert command.actor

    guide_loop = command.actor.state.guide_loop
    enabled_axes = command.actor.state.enabled_axes

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
            command.warning("RA/Dec correction too large. Not applying correction.")

        else:
            corr_radec *= k_radec

            _set_correcting(command, True)

            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset arc {corr_radec[0]}, {corr_radec[1]} /computed",
            )

            _set_correcting(command, False)

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
            command.warning("Rotator correction too large. Not applying correction.")

        else:
            corr_rot *= k_rot

            _set_correcting(command, True)

            tcc_offset_cmd = await command.send_command(
                "tcc",
                f"offset guide 0.0, 0.0, {corr_rot} /computed",
            )

            _set_correcting(command, False)

            if tcc_offset_cmd.status.did_fail:
                command.error("Failed applying rotator correction.")
                return correction_applied

            correction_applied[2] = numpy.round(corr_rot * 3600.0, 3)

    return correction_applied
