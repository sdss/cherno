# encoding: utf-8

from typing import cast

from sdsstools import get_logger, get_package_version
from sdsstools.configuration import __ENVVARS__, get_config


NAME = "sdss-cherno"

# Inits the logging system. Only shell logging, and exception and warning catching.
# File logging can be started by calling log.start_file_logger(path).
log = get_logger(NAME)


__ENVVARS__["OBSERVATORY"] = "?"
config = get_config("cherno")


__version__ = cast(str, get_package_version(path=__file__, package_name=NAME))
