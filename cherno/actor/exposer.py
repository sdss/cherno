#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-09
# @Filename: exposer.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import re
from glob import glob

from typing import Any, Callable, NoReturn, Union

from clu import Command
from sdsstools.time import get_sjd

from cherno import config
from cherno.exceptions import ExposerError
from cherno.maskbits import GuiderStatus

from . import ChernoCommandType


CallbackType = Union[Callable[[ChernoCommandType, list[str]], Any], None]


class Exposer:
    """Helper to expose the cameras.

    Parameters
    ----------
    command
        Actor command requesting the exposure loop.
    callback
        A callback to invoke every time a group of exposures finishes. The
        callback is invoked with the command and a list of new exposure files.
        If the callback is a coroutine it is run in a task if blocking is
        disabled and awaited otherwise. Use `.set_blocking` to change the
        blocking behaviour. The default behaviour is to block. If the callback
        is a function, it is run in an executor.

    """

    def __init__(self, command: ChernoCommandType, callback: CallbackType = None):
        self.command = command

        assert command.actor
        self.actor = command.actor

        self.actor_state = self.actor.state
        self.stop_reached: bool = False

        self.callback = callback
        self._blocking: bool = True

    def set_blocking(self, blocking: bool):
        """Whether to block while calling the callback."""

        self._blocking = blocking

    def fail(self, message="") -> NoReturn:
        """Sets the guider status to failed and raises an exception."""

        self.actor_state.set_status(GuiderStatus.FAILED | GuiderStatus.IDLE)
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

        return await self.loop(exposure_time, count=1, **kwargs)

    def is_stopping(self):
        """Is the guider loop stopping?"""

        return (self.actor_state.status & GuiderStatus.STOPPING).value > 0

    async def loop(
        self,
        exposure_time: float | None = None,
        count: int | None = None,
        delay: float = 0.0,
        names: list[str] | None = None,
        timeout: float | None = None,
        callback: CallbackType = None,
        stop_condition: Callable[[], bool] | None = None,
        max_iterations: int | None = None,
    ):
        """Loops the cameras.

        Parameters
        ----------
        exposure_time
            The exposure time to command. If `None`, uses the stored exposure
            time in the actor state.
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
        stop_condition
            A function called at the end of each expose loop, and after the callback
            has been executed, to determine whether to stop the loop. The function
            is called without arguments and must return a boolean, with `True`
            stopping the loop.
        max_iterations
            Maximum number of iterations after which to fail the loop, if
            ``stop_condition`` has not been reached.

        """

        if self.actor.tron is None:
            raise ExposerError("Tron is not connected. Cannot expose.")

        if (self.actor_state.status & GuiderStatus.NON_IDLE).value > 0:
            raise ExposerError("The guider is already exposing.")

        if exposure_time is not None:
            self.actor_state.exposure_time = exposure_time

        n_exp = 0
        while True:
            if self.is_stopping() or (count is not None and n_exp >= count):
                self.actor_state.set_status(GuiderStatus.STOPPING, mode="remove")
                self.actor_state.set_status(GuiderStatus.IDLE, mode="add")
                return

            self._check_ffs()

            # Clear the FAILED flag if it's set
            self.actor_state.set_status(GuiderStatus.FAILED, mode="remove")

            # Set the status of the guider as EXPOSING. This removes IDLE.
            self.actor_state.set_status(GuiderStatus.EXPOSING, mode="add")

            names = names or self.actor.state.enabled_cameras
            if names is None or len(names) == 0:
                self.fail("No cameras defined.")

            num = self._get_num(names)

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
                etime = self.actor_state.exposure_time
                expose_command = await asyncio.wait_for(
                    self.actor.tron.send_command(
                        "fliswarm",
                        f"talk -n {names_comma} -- expose -n {num} {etime}",
                    ),
                    etime + timeout if timeout is not None else None,
                )
            except asyncio.TimeoutError:
                self.fail("Timed out waiting for the exposure to finish.")

            if expose_command.status.did_fail:
                self.fail("Expose command failed.")

            self.actor_state.set_status(GuiderStatus.EXPOSING, mode="remove")

            if self.is_stopping():
                # Continue, the first check in the new iteration will stop the loop.
                continue

            filenames = self._get_filename_bundle(expose_command)
            if filenames is None:
                self.fail("The keyword filename_bundle was not output.")
            elif len(filenames) == 0:
                self.fail("The keyword filename_bundle is empty.")
            else:
                try:
                    await self.invoke_callback(
                        filenames,
                        callback=callback or self.callback,
                    )
                except Exception as err:
                    self.fail(str(err))

            n_exp += 1

            if stop_condition is not None and stop_condition():
                self.actor_state.set_status(GuiderStatus.STOPPING, mode="add")
                self.stop_reached = True
                continue

            if max_iterations and n_exp >= max_iterations:
                self.fail("Maximum number of iterations reached.")

            if delay:
                self.actor_state.set_status(GuiderStatus.WAITING, mode="add")
                await asyncio.sleep(delay)
                self.actor_state.set_status(
                    GuiderStatus.WAITING,
                    mode="remove",
                    report=False,
                )

    def _check_ffs(self):
        """Checks that the FFS are open."""

        if self.actor.observatory == "LCO":
            return

        values = self.actor.models["mcp"]["ffsStatus"].value
        if len(values) == 0 or all([value is None for value in values]):
            self.command.warning("FFS status unknown.")

        if not all([int(ss) == 10 for ss in values]):
            self.command.warning("FFS petals are not open.")

    def _get_num(self, names: list[str]) -> int:
        """Returns the next sequence number."""

        sjd = get_sjd()
        dirpath = os.path.join(config["cameras"]["path"], str(sjd))
        if not os.path.exists(dirpath):
            return 1

        gimgs = glob(os.path.join(dirpath, "*.fits*"))
        matches = [
            int(m.group(1))
            for file_ in gimgs
            if (m := re.search(r"gimg\-.+?\-([0-9]+)", file_))
        ]

        return max(set(matches)) + 1

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

        def unset_processing(*args, **kwargs):
            """Removes the ``PROCESSING`` flag after the callback completes.

            Does not report the status since the loop will already do that almost
            immediately.

            """

            self.actor_state.set_status(
                GuiderStatus.PROCESSING,
                mode="remove",
                report=False,
            )

        self.actor_state.set_status(GuiderStatus.PROCESSING, mode="add")

        callback = callback or self.callback
        if callback is None:
            self.command.warning("Exposer: no callback defined.")
            return

        if asyncio.iscoroutinefunction(callback) or (
            hasattr(callback, "func") and asyncio.iscoroutinefunction(callback.func)
        ):
            task = asyncio.create_task(callback(self.command, filenames))
        else:
            task = asyncio.get_running_loop().run_in_executor(
                None,
                callback,
                self.command,
                filenames,
            )

        task.add_done_callback(unset_processing)

        if self._blocking:
            await task

        return
