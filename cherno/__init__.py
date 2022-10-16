# encoding: utf-8

import os
import pathlib
import warnings

from typing import Any, cast

from sdsstools import get_logger, get_package_version
from sdsstools.configuration import __ENVVARS__, Configuration, get_config


def set_observatory(observatory: str | None):
    """Returns and sets the config for the desired observatory."""

    if config is not None:
        globals()["config"].clear()
    else:
        globals()["config"] = {}

    if observatory is None:
        observatory = "APO"
        warnings.warn("Unknown observatory. Defaulting to APO!", UserWarning)
    else:
        observatory = observatory.upper()

    os.environ["OBSERVATORY"] = observatory
    globals()["config"].update(get_config(f"cherno_{observatory}"))


NAME = "sdss-cherno"

# Inits the logging system. Only shell logging, and exception and warning catching.
# File logging can be started by calling log.start_file_logger(path).
log = get_logger(NAME)


# Sets the config for the observatory defined in $OBSERVATORY.
config: dict[str, Any] | None = None
set_observatory(os.environ.get("OBSERVATORY", None))


__version__ = cast(str, get_package_version(path=__file__, package_name=NAME))
