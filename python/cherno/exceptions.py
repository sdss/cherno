# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class ChernoError(Exception):
    """A custom core Cherno exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(ChernoError, self).__init__(message)


class ChernoNotImplemented(ChernoError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(ChernoNotImplemented, self).__init__(message)


class ChernoAPIError(ChernoError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Cherno API'
        else:
            message = 'Http response error from Cherno API. {0}'.format(message)

        super(ChernoAPIError, self).__init__(message)


class ChernoApiAuthError(ChernoAPIError):
    """A custom exception for API authentication errors"""
    pass


class ChernoMissingDependency(ChernoError):
    """A custom exception for missing dependencies."""
    pass


class ChernoWarning(Warning):
    """Base warning for Cherno."""


class ChernoUserWarning(UserWarning, ChernoWarning):
    """The primary warning class."""
    pass


class ChernoSkippedTestWarning(ChernoUserWarning):
    """A warning for when a test is skipped."""
    pass


class ChernoDeprecationWarning(ChernoUserWarning):
    """A warning for deprecated features."""
    pass
