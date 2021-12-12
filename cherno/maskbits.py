#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: maskbits.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from enum import Enum, Flag


__all__ = ["GuiderStatus", "CameraStatus"]


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


class CameraStatus(Enum):
    """Enumeration of camera statuses."""

    EXPOSURE_IDLE = "idle"
    EXPOSURE_FLUSHING = "flushing"
    EXPOSURE_INTEGRATING = "integrating"
    EXPOSURE_READING = "reading"
    EXPOSURE_READ = "read"
    EXPOSURE_DONE = "done"
    EXPOSURE_FAILED = "failed"
    EXPOSURE_WRITING = "writing"
    EXPOSURE_WRITTEN = "written"
    EXPOSURE_POST_PROCESSING = "post_processing"
    EXPOSURE_POST_PROCESS_DONE = "post_process_done"
    EXPOSURE_POST_PROCESS_FAILED = "post_process_failed"
    UNKNWON = 'unknown'