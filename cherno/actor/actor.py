#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: actor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
from dataclasses import dataclass, field

from typing import cast

import clu
from clu.legacy import TronKey

import cherno
from cherno import __version__, config
from cherno.maskbits import CameraStatus, GuiderStatus


class ChernoActor(clu.LegacyActor):
    """The Cherno SDSS-style actor."""

    def __init__(self, *args, **kwargs):

        models = list(set(kwargs.pop("models", []) + ["fliswarm"]))

        schema = kwargs.pop("schema", None)
        if schema is not None and os.path.isabs(schema) is False:
            schema = os.path.join(os.path.dirname(cherno.__file__), schema)

        super().__init__(
            *args,
            version=__version__,
            models=models,
            schema=schema,
            **kwargs,
        )

        self.state = ChernoState(self)

        if (offset := config["offset"]) is not None:
            self.state.offset = cast(tuple[float, float, float], tuple(offset))

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
            temperature = float(key.value[16])
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
    camera_state: dict[str, CameraState] = field(default_factory=dict)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    exposure_time: float = 15.0
    guide_loop: dict = field(default_factory=dict)
    acquisition: dict = field(default_factory=dict)
    enabled_cameras: list = field(default_factory=list)
    enabled_axes: list = field(default_factory=list)

    def __post_init__(self):
        self.guide_loop = config["guide_loop"].copy()
        self.acquisition = config["acquisition"].copy()
        self.enabled_cameras = config["cameras"]["names"].copy()
        self.enabled_axes = ["radec", "rot", "focus"]

    def set_status(self, new_status: GuiderStatus, mode="override", report=True):
        """Sets the status and broadcasts it."""

        old_status = self.status

        if mode == "override":
            self.status = new_status
        elif mode == "add":
            self.status |= new_status
        elif mode == "remove":
            self.status &= ~new_status
        else:
            self.actor.write("e", message={"error": f"Invalid mode {mode}."})
            return

        # Some statuses are incompatible with IDLE.
        if (self.status & GuiderStatus.IDLE) and (self.status & GuiderStatus.NON_IDLE):
            self.status &= ~GuiderStatus.IDLE

        # If no status, the guider must be idle.
        if self.status.value == 0:
            self.status = GuiderStatus.IDLE

        if self.status.value != old_status.value and report:
            self.actor.write("i", message={"guider_status": hex(self.status.value)})


@dataclass
class CameraState:
    """Stores the state of a camera."""

    name: str
    temperature: float | None = None
    status: CameraStatus = CameraStatus("unknown")
