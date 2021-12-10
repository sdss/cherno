#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-09
# @Filename: exposer.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from typing import Any, Callable, Coroutine, Union

from cherno.exceptions import ChernoError
from cherno.maskbits import GuiderStatus

from . import ChernoCommandType


CallbackType = Union[Callable[[list[str]], Any], Coroutine, None]


class Exposer:
    """Helper to expose the cameras.

    Parameters
    ----------
    command
        Actor command requesting the exposure loop.
    callback
        A callback to invoke every time a group of exposures finishes. The
        callback is invoked with the list of new exposure files. If the
        callback is a coroutine it is run in a task if blocking is disabled
        and awaited otherwise. Use `.set_blocking` to change the blocking
        behaviour. The default behaviour is to block. If the callback is a
        function, it is run in an executor.

    """

    def __init__(self, command: ChernoCommandType, callback: CallbackType = None):

        self.command = command

        assert command.actor
        self.actor = command.actor

        self.actor_state = self.actor.state

        self.callback = callback
        self._blocking: bool = True

    def set_blocking(self, blocking: bool):
        """Whether to block while calling the callback."""

        self._blocking = blocking

    async def loop(
        self,
        exposure_time: float,
        count: int | None = None,
        delay: float = 0.0,
        names: list[str] | None = None,
        timeout: float | None = None,
        callback: CallbackType = None,
    ):
        """Loops the cameras.

        Parameters
        ----------
        exposure_time
            The exposure time to command.
        count
            The number of exposures to take before stopping the loop.
        delay
            How long to wait between the end of an exposure and the beginning
            of the next one.
        names
            FLISwarm camera names to expose. If `None`, uses the cameras specified
            in the configuration.
        timeout
            How long to wait until the end of the exposure time for the FLISwarm
            command to finish before cancelling the loop.
        callback
            The callback to invoke after each exposure completes. If specified,
            overrides the global callback only for this loop.

        """

        if self.actor.tron is None:
            raise ChernoError("Tron is not connected. Cannot expose.")

        n_exp = 0
        while True:

            stopping = (self.actor_state.status & GuiderStatus.STOPPING).value > 0
            if stopping is True or (count is not None and n_exp >= count):
                self.actor_state.set_status(GuiderStatus.IDLE)
                return

            # TODO: check that the cameras we want to expose are connected and idle.
            if names is not None:
                target = "-n " + ",".join(names)
            elif category is not None:
                target = f"-c {category}"
            else:
                target = ""

            try:
                fliswarm_command = await self.actor.tron.send_command(
                    "fliswarm",
                    f"talk {target} expose {exposure_time}",
                )
            except:
                pass
