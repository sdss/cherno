#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: maskbits.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from enum import Flag

__all__ = ["GuiderStatus"]


class GuiderStatus(Flag):
    """Maskbits with the guider status."""

    IDLE = 1
    EXPOSING = 2 << 0
    PROCESSING = 2 << 1
    CORRECTING = 2 << 2
    STOPPING = 2 << 3
    FAILED = 2 << 4

    def get_names(self):
        """Returns a list of active bit names."""

        return [bit.name for bit in GuiderStatus if self & bit]
