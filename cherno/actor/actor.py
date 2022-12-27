#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: actor.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

import clu
from clu.legacy import TronKey

import cherno
from cherno import __version__, config
from cherno.maskbits import CameraStatus, GuiderStatus


if TYPE_CHECKING:
    from cherno.guider import Guider


class ChernoActor(clu.LegacyActor):
    """The Cherno SDSS-style actor."""

    def __init__(self, *args, **kwargs):
        self.observatory: str = config["observatory"].upper()

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
    enabled_cameras: list = field(default_factory=list)
    enabled_axes: list = field(default_factory=list)
    scale_history: list = field(default_factory=list)
    rms_history: deque = field(default_factory=deque)
    astrometry_net_odds: float = 1e9

    _guider_obj: Guider | None = None
    _exposure_loop: asyncio.Task | None = None

    def __post_init__(self):
        self.observatory = self.actor.observatory
        self.enabled_cameras = config["cameras"]["names"].copy()
        self.enabled_axes = config["enabled_axes"].copy()
        self.astrometry_net_odds = config["guider"]["astrometry_net_odds"]
        self.rms_history = deque(maxlen=10)

        self.guide_loop = config["guide_loop"].copy()
        for axis in ["ra", "dec", "rot", "focus"]:
            for term in ["td", "ti"]:
                if term not in self.guide_loop[axis]["pid"]:
                    self.guide_loop[axis]["pid"][term] = 0.0

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
