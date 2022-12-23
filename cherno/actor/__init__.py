#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-30
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from typing import Union

from clu import Command, FakeCommand
from clu.parsers.click import command_parser as cherno_parser

from .actor import ChernoActor


ChernoCommandType = Union[Command[ChernoActor], FakeCommand]


from .acquire import *
from .commands.config import *
from .commands.converge import *
from .guide import *
from .commands.offset import *
from .commands.scale import *
from .commands.set import *
from .commands.show import *
from .commands.status import *
from .commands.stop import *
from .commands.version import *
