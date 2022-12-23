# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32


class ChernoError(Exception):
    """A custom core Cherno exception"""

    def __init__(self, message=None):
        message = "There has been an error" if not message else message

        super(ChernoError, self).__init__(message)


class ExposerError(ChernoError):
    """An error in the `.Exposer` class."""

    pass


class ChernoNotImplemented(ChernoError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):
        message = "This feature is not implemented yet." if not message else message

        super(ChernoNotImplemented, self).__init__(message)


class ChernoMissingDependency(ChernoError):
    """A custom exception for missing dependencies."""

    pass


class ChernoWarning(Warning):
    """Base warning for Cherno."""


class ChernoUserWarning(UserWarning, ChernoWarning):
    """The primary warning class."""

    pass


class ChernoDeprecationWarning(ChernoUserWarning):
    """A warning for deprecated features."""

    pass
