#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: actor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from dataclasses import dataclass

import clu
from clu.legacy import TronKey

from cherno import __version__
from cherno.maskbits import CameraStatus, GuiderStatus


class ChernoActor(clu.LegacyActor):
    """The Cherno SDSS-style actor."""

    def __init__(self, *args, **kwargs):

        models = list(set(kwargs.pop("models", []) + ["fliswarm"]))

        super().__init__(*args, version=__version__, models=models, **kwargs)

        self.state = ChernoState(self)

        self.models["fliswarm"].register_callback(self._process_fliswarm_status)

    async def _process_fliswarm_status(self, model: dict, key: TronKey):
        """Updates FLISwarm messages."""

        camera_state = self.state.camera_state

        if key.name == "exposure_state":
            camera_name, _, status, *_ = key.value
            if camera_name not in camera_state:
                camera_state[camera_name] = CameraState(camera_name)
            camera_state[camera_name].status = CameraStatus(status)
        elif key.name == "status":
            camera_name = key.value[0]
            temperature = key.value[16]
            if camera_name not in camera_state:
                camera_state[camera_name] = CameraState(camera_name)
            camera_state[camera_name].temperature = temperature
        else:
            pass


@dataclass
class ChernoState:
    """Stores the state of the guider."""

    actor: ChernoActor
    status: GuiderStatus = GuiderStatus.IDLE
    camera_state: dict[str, CameraState] = {}

    def set_status(self, status: GuiderStatus, mode="override"):
        """Sets the status and broadcasts it."""

        self.status = status
        self.actor.write("i", message={"guider_status": hex(self.status.value)})


@dataclass
class CameraState:
    """Stores the state of a camera."""

    name: str
    temperature: float | None = None
    status: CameraStatus = CameraStatus("idle")
