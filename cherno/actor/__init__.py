#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


from typing import TYPE_CHECKING

from clu.parsers.click import command_parser as cherno_parser

from .actor import ChernoActor

if TYPE_CHECKING:
    from clu import Command

    ChernoCommandType = Command[ChernoActor]

from .guide import *
