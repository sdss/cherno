#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-10-19
# @Filename: config.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from copy import deepcopy

import click

from cherno import config as cherno_config
from cherno import set_observatory

from .. import ChernoCommandType, cherno_parser


__all__ = ["config_command"]


ORIGINAL_CONFIG = deepcopy(cherno_config)


def all_keys(dict_obj, prefix=""):
    """Returns a prefixed list of all the keys in a nested dictionary."""

    keys = []

    for key, value in dict_obj.items():
        if not isinstance(value, dict):
            keys.append(prefix + key)
        else:
            keys += list(all_keys(value, prefix + key + "."))

    return keys


@cherno_parser.group(name="config")
def config_command(*args, **kwargs):
    """Reads/modifies the internal configuration."""

    pass


@config_command.command()
@click.argument("KEY", type=str)
async def get(command: ChernoCommandType, key: str):
    """Returns the value of a configuration parameter."""

    parts = key.split(".")

    value = None

    subconfig = cherno_config.copy()

    for ii, part in enumerate(parts):
        if not isinstance(subconfig, dict) or part not in subconfig:
            return command.fail(error=f"Cannot read parameter {key}.")

        if ii == len(parts) - 1:
            value = subconfig[part]
        else:
            if not isinstance(subconfig[part], dict):
                return command.fail(error=f"Cannot read parameter {key}.")
            subconfig = subconfig[part]

    if isinstance(value, dict):
        return command.fail(error="Parameter is a dictionary. Try selecting a subitem.")
    elif isinstance(value, list):
        listify = ", ".join(map(str, value))
        return command.finish(text=f'"{key}={listify}"')
    else:
        return command.finish(text=f'"{key}={value}"')


@config_command.command()
@click.argument("KEY", type=str)
@click.argument("VALUE", type=str, required=False)
@click.option(
    "--cast",
    type=click.Choice(["int", "float", "str", "list", "bool"]),
    help="The cast function to use. Without it the type is inferred.",
)
async def set(
    command: ChernoCommandType,
    key: str,
    value: str | None = None,
    cast: str | None = None,
):
    """Sets the value of a configuration parameter.

    If the value is not passed, the default value is reset.

    """

    if key in ["observatory"]:
        return command.fail(error=f"{key} cannot be changed.")

    try:
        if value is None:
            parsed = None
        elif cast is None:
            parsed = eval(value)
        elif cast == "int":
            parsed = int(value)
        elif cast == "float":
            parsed = float(value)
        elif cast == "str":
            parsed = str(value)
        elif cast == "list":
            parsed = list(value)
        elif cast == "bool":
            parsed = bool(value)
        else:
            raise TypeError()
    except Exception:
        return command.fail(error="Failed parsing value.")

    parts = key.split(".")

    subconfig = cherno_config
    original = ORIGINAL_CONFIG

    for ii, part in enumerate(parts):
        if not isinstance(subconfig, dict) or part not in subconfig:
            return command.fail(error=f"Cannot set parameter {key}.")

        if ii == len(parts) - 1:
            if parsed is None:
                subconfig[part] = original[part]
            else:
                subconfig[part] = parsed
            return command.finish(text=f'"{key}={subconfig[part]}"')
        else:
            if not isinstance(subconfig[part], dict):
                return command.fail(error=f"Cannot read parameter {key}.")
            subconfig = subconfig[part]
            original = original[part]


@config_command.command(name="list")
@click.argument("SECTION", type=str, required=False)
async def list_(command: ChernoCommandType, section: str | None):
    """List all the parameters in the configuration or in a section.."""

    section = section or ""
    parts = section.split(".") if section else []

    base = cherno_config.copy()

    for part in parts:
        if (
            not isinstance(base, dict)
            or part not in base
            or not isinstance(base[part], dict)
        ):
            return command.fail(error=f"Cannot list parameters in section {section}.")

        base = base[part]

    prefix = section + "." if section != "" else ""
    keys = all_keys(base, prefix=prefix)

    if len(keys) == 0:
        return command.fail("No parameters found.")

    return command.finish(", ".join(keys))


@config_command.command()
async def reset(command: ChernoCommandType):
    """Resets the configuration to the initial values."""

    set_observatory(cherno_config["observatory"])

    return command.finish()
