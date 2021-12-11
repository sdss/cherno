#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-10
# @Filename: run_astrometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import sys

from clu.command import FakeCommand
from sdsstools import get_logger

from cherno.astrometry import process_and_correct


async def run_astrometry(bundles_file: str):
    """Runs astrometry.net on a list of bundled files."""

    logger = get_logger("run-astrometry-test")
    logger.sh.setLevel(10)

    command = FakeCommand(logger)

    with open(bundles_file, "r") as f:
        for line in f.readlines():
            files = line.strip().split(",")
            if not all([os.path.exists(file_) for file_ in files]):
                raise ValueError(f"Some files not found: {files}")

            logger.info(f"Processing {line.strip()}")

            await process_and_correct(
                command,
                files,
                run_options={"overwrite": True, "plot": False, "cpulimit": 5},
            )


if __name__ == "__main__":
    asyncio.run(run_astrometry(sys.argv[1]))
