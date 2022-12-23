#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-06
# @Filename: acquire.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import click

from cherno.actor.commands.guide import Params, _guide, get_guide_common_params

from .. import cherno_parser


__all__ = ["acquire"]


acquire_params = get_guide_common_params(continuous=False, full=True)


@cherno_parser.command(params=acquire_params)
@click.option("--test", is_flag=True)
async def acquire(**kwargs):
    """Runs the acquisition procedure."""

    params = Params(**kwargs)

    return await _guide(params)
