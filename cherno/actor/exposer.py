#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-09
# @Filename: exposer.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from typing import Any, Callable, Union

from clu import Command

from cherno import config
from cherno.exceptions import ExposerError
from cherno.maskbits import GuiderStatus

from . import ChernoCommandType


CallbackType = Union[Callable[[list[str]], Any], None]


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

    def fail(self, message=""):
        """Sets the guider status to failed and raises an exception."""

        self.actor_state.set_status(GuiderStatus.FAILED)
        raise ExposerError(message)

    async def one(self, exposure_time: float, **kwargs):
        """Exposes the cameras once.

        Parameters
        ----------
        exposure_time
            The exposure time to command.
        kwargs
            Arguments to pass to `.loop`.

        """

        if "count" in kwargs:
            raise ExposerError("Cannot specify count with one().")

        return await loop(exposure_time, count=1, **kwargs)

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
            raise ExposerError("Tron is not connected. Cannot expose.")

        n_exp = 0
        while True:

            stopping = (self.actor_state.status & GuiderStatus.STOPPING).value > 0
            if stopping is True or (count is not None and n_exp >= count):
                self.actor_state.set_status(GuiderStatus.IDLE)
                return

            names = names or config["cameras"]
            if names is None or len(names) == 0:
                self.fail("No cameras defined.")

            names_comma = ",".join(names)

            # Issue a fliswarm talk status to the cameras to update their status.
            try:
                status_command = await asyncio.wait_for(
                    self.actor.tron.send_command(
                        "fliswarm",
                        f"talk -n {names_comma} status",
                    ),
                    3,
                )
            except asyncio.TimeoutError:
                self.fail("Timed out updating camera status.")

            if status_command.status.did_fail:
                self.fail("Failed updating camera status.")

            try:
                expose_command = await asyncio.wait_for(
                    self.actor.tron.send_command(
                        "fliswarm",
                        f"talk {names_comma} expose {exposure_time}",
                    ),
                    exposure_time + timeout if timeout is not None else None,
                )
            except asyncio.TimeoutError:
                self.fail("Timed out waiting for the exposure to finish.")

            if expose_command.status.did_fail:
                self.fail("Expose command failed.")

            filenames = self._get_filename_bundle(expose_command)
            if filenames is None:
                self.fail("The keyword filename_bundle was not output.")
            elif len(filenames) == 0:
                self.fail("The keyword filename_bundle is empty.")
            else:
                await self.invoke_callback(
                    filenames,
                    callback=callback or self.callback,
                )

            n_exp += 1

    def _get_filename_bundle(self, command: Command):
        """Returns the ``filename_bundle`` values from the list of command replies."""

        for reply in command.replies:
            for reply_key in reply.keywords:
                key_name = reply_key.name.lower()
                if key_name != "filename_bundle":
                    continue
                filenames = [value.native for value in reply_key.values]
                return filenames

        return None

    async def invoke_callback(
        self,
        filenames: list[str],
        callback: CallbackType = None,
    ):
        """Invokes the callback with a list of filenames."""

        callback = callback or self.callback
        if callback is None:
            self.fail("No callback defined.")

        if asyncio.iscoroutinefunction(callback):
            task = asyncio.create_task(callback(filenames))
        else:
            task = asyncio.get_running_loop().run_in_executor(None, callback, filenames)

        if self._blocking:
            await task

        return
