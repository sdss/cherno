#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-18
# @Filename: mock.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from . import ChernoCommandType, cherno_parser


__all__ = ["mock_command"]


@cherno_parser.command()
async def mock_command(command: ChernoCommandType):
    """Mocks some output. For developer use only."""

    command.info(fwhm_camera=["gfa4", 290, 2.315, 13, 10])
    command.info(fwhm_camera=["gfa2", 290, 2.566, 15, 14])
    command.info(fwhm_camera=["gfa4", 290, 2.315, 13, 10])
    command.info(fwhm_camera=["gfa3", 290, 2.409, 10, 6])
    command.info(fwhm_camera=["gfa5", 290, 2.2, 13, 8])
    command.info(focus_fit=[290, 2.198, 2.298e-06, -0.001441, 6.636, 0.891, -62.7])

    command.info(fwhm_camera=["gfa4", 291, 2.315, 13, 10])
    command.info(fwhm_camera=["gfa2", 291, 2.566, 15, 14])
    command.info(fwhm_camera=["gfa4", 291, 2.315, 13, 10])
    command.info(fwhm_camera=["gfa3", 291, 2.409, 10, 6])
    command.info(fwhm_camera=["gfa5", 291, 2.2, 13, 8])
    command.info(focus_fit=[291, 2.198, 2.298e-06, -0.001441, 6.636, 0.891, -62.7])

    return command.finish()
