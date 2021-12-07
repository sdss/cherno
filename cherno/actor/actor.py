#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: actor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import logging
from dataclasses import dataclass

import clu
from clu.tools import ActorHandler

from cherno import __version__, log
from cherno.exceptions import ChernoUserWarning
from cherno.maskbits import GuiderStatus


class ChernoActor(clu.LegacyActor):
    """The jaeger SDSS-style actor."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, version=__version__, **kwargs)

        self.state = ChernoState()

        # Add ActorHandler to log and to the warnings logger.
        self.actor_handler = ActorHandler(
            self,
            level=logging.WARNING,
            filter_warnings=[ChernoUserWarning],
        )
        log.addHandler(self.actor_handler)
        if log.warnings_logger:
            log.warnings_logger.addHandler(self.actor_handler)


@dataclass
class ChernoState:
    """Stores the state of the guider."""

    status: GuiderStatus = GuiderStatus.IDLE
