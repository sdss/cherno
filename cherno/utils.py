#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-08
# @Filename: utils.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import concurrent.futures
import warnings
from functools import partial

from typing import TYPE_CHECKING, Any, cast

import numpy
from coordio.defaults import PLATE_SCALE
from coordio.exceptions import CoordIOUserWarning
from coordio.utils import radec2wokxy

from cherno.coordinates import gfa_to_wok


if TYPE_CHECKING:
    from cherno.acquisition import AcquisitionData


warnings.simplefilter("ignore", category=CoordIOUserWarning)


async def run_in_executor(fn, *args, catch_warnings=False, executor="thread", **kwargs):
    """Runs a function in an executor.

    In addition to streamlining the use of the executor, this function
    catches any warning issued during the execution and reissues them
    after the executor is done. This is important when using the
    actor log handler since inside the executor there is no loop that
    CLU can use to output the warnings.

    In general, note that the function must not try to do anything with
    the actor since they run on different loops.

    """

    fn = partial(fn, *args, **kwargs)

    if executor == "thread":
        executor = concurrent.futures.ThreadPoolExecutor
    elif executor == "process":
        executor = concurrent.futures.ProcessPoolExecutor
    else:
        raise ValueError("Invalid executor name.")

    if catch_warnings:
        with warnings.catch_warnings(record=True) as records:
            with executor() as pool:
                result = await asyncio.get_event_loop().run_in_executor(pool, fn)

        for ww in records:
            warnings.warn(ww.message, ww.category)

    else:
        with executor() as pool:
            result = await asyncio.get_running_loop().run_in_executor(pool, fn)

    return result


def umeyama(X, Y):
    """Rigid alignment of two sets of points in k-dimensional Euclidean space.

    Given two sets of points in correspondence, this function computes the
    scaling, rotation, and translation that define the transform TR that
    minimizes the sum of squared errors between TR(X) and its corresponding
    points in Y.  This routine takes O(n k^3)-time.

    Parameters
    ----------
    X
        A ``k x n`` matrix whose columns are points.
    Y
        A ``k x n`` matrix whose columns are points that correspond to the
        points in X

    Returns
    -------
    c,R,t
        The  scaling, rotation matrix, and translation vector defining the
        linear map TR as ::

            TR(x) = c * R * x + t

        such that the average norm of ``TR(X(:, i) - Y(:, i))`` is
        minimized.

    Copyright: Carlo Nicolini, 2013

    Code adapted from the Mark Paskin Matlab version from
    http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m

    See paper by Umeyama (1991)

    """

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)

    Xc = X - numpy.tile(mx, (n, 1)).T
    Yc = Y - numpy.tile(my, (n, 1)).T

    sx = numpy.mean(numpy.sum(Xc * Xc, 0))

    Sxy = numpy.dot(Yc, Xc.T) / n

    U, D, V = numpy.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()

    r = numpy.linalg.matrix_rank(Sxy)
    S = numpy.eye(m)

    if r < (m - 1):
        raise ValueError("Not enough independent measurements")

    if numpy.linalg.det(Sxy) < 0:
        S[-1, -1] = -1
    elif r == m - 1:
        if numpy.linalg.det(U) * numpy.linalg.det(V) < 0:
            S[-1, -1] = -1

    R = numpy.dot(numpy.dot(U, S), V.T)
    c = numpy.trace(numpy.dot(numpy.diag(D), S)) / sx
    t = my - c * numpy.dot(R, mx)

    return c, R, t


def astrometry_fit(
    data: list[AcquisitionData],
    grid=(10, 10),
    offset: tuple | list = (0.0, 0.0, 0.0),
    obstime: float | None = None,
):
    """Fits translation, rotation, and scale from a WCS solution."""

    offset_ra, offset_dec, offset_pa = offset

    xwok_gfa: list[float] = []
    ywok_gfa: list[float] = []
    xwok_astro: list[float] = []
    ywok_astro: list[float] = []

    for d in data:

        camera_id = int(d.camera[-1])
        xidx = numpy.arange(2048)[:: 2048 // grid[0]]
        yidx = numpy.arange(2048)[:: 2048 // grid[1]]

        coords: Any = d.wcs.pixel_to_world(xidx, yidx)
        ra = coords.ra.value
        dec = coords.dec.value

        for x, y in zip(xidx, yidx):
            xw, yw, _ = gfa_to_wok(x, y, camera_id)
            xwok_gfa.append(cast(float, xw))
            ywok_gfa.append(cast(float, yw))

        _xwok_astro, _ywok_astro, *_ = radec2wokxy(
            ra,
            dec,
            None,
            "GFA",
            d.field_ra - offset_ra * numpy.cos(numpy.deg2rad(d.field_dec)) / 3600.0,
            d.field_dec - offset_dec / 3600.0,
            d.field_pa - offset_pa / 3600.0,
            d.observatory.upper(),
            obstime,
        )

        xwok_astro += _xwok_astro.tolist()
        ywok_astro += _ywok_astro.tolist()

    X = numpy.array([xwok_gfa, ywok_gfa])
    Y = numpy.array([xwok_astro, ywok_astro])
    try:
        c, R, t = umeyama(X, Y)
    except ValueError:
        return False

    plate_scale = PLATE_SCALE[data[0].observatory]

    delta_x = numpy.round(t[0] / plate_scale * 3600.0, 3)
    delta_y = numpy.round(t[1] / plate_scale * 3600.0, 3)

    # delta_x and delta_y only align with RA/Dec if PA=0. Otherwise we need to
    # project using the PA.
    pa_rad = numpy.deg2rad(data[0].field_pa)
    delta_ra = delta_x * numpy.cos(pa_rad) + delta_y * numpy.sin(pa_rad)
    delta_dec = -delta_x * numpy.sin(pa_rad) + delta_y * numpy.cos(pa_rad)

    # Round up.
    delta_ra = numpy.round(delta_ra, 3)
    delta_dec = numpy.round(delta_dec, 3)

    delta_rot = numpy.round(-numpy.rad2deg(numpy.arctan2(R[1, 0], R[0, 0])) * 3600.0, 1)
    delta_scale = numpy.round(c, 6)

    delta_x = (numpy.array(xwok_gfa) - numpy.array(xwok_astro)) ** 2  # type: ignore
    delta_y = (numpy.array(ywok_gfa) - numpy.array(ywok_astro)) ** 2  # type: ignore

    xrms = numpy.round(numpy.sqrt(numpy.sum(delta_x) / len(delta_x)), 3)
    yrms = numpy.round(numpy.sqrt(numpy.sum(delta_y) / len(delta_y)), 3)
    rms = numpy.round(numpy.sqrt(numpy.sum(delta_x + delta_y) / len(delta_x)), 3)

    return (delta_ra, delta_dec, delta_rot, delta_scale, xrms, yrms, rms)
