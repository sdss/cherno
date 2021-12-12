#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-10
# @Filename: coordinates.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import numpy
import pandas
from coordio import ICRS, Field, FocalPlane, Observed, Site, Wok
from coordio.conv import guideToTangent, tangentToWok
from coordio.defaults import INST_TO_WAVE, calibration


def gfa_to_wok(xPix: float, yPix: float, gfaID: int):
    """Converts from a GFA pixel to wok coordinates."""

    idx = pandas.IndexSlice

    gfaRow = calibration.gfaCoords.loc[idx[:, gfaID], :]

    b = gfaRow[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
    iHat = gfaRow[["ix", "iy", "iz"]].to_numpy().squeeze()
    jHat = gfaRow[["jx", "jy", "jz"]].to_numpy().squeeze()
    kHat = gfaRow[["kx", "ky", "kz"]].to_numpy().squeeze()

    xt, yt = guideToTangent(xPix, yPix)
    zt = 0

    return tangentToWok(xt, yt, zt, b, iHat, jHat, kHat)  # type: ignore


def gfa_to_radec(
    xPix: float,
    yPix: float,
    gfaID: int,
    bore_ra: float,
    bore_dec: float,
    position_angle: float = 0,
    site_name: str = "APO",
):
    """Converts from a GFA pixel to observed RA/Dec."""

    site = Site(site_name)
    site.set_time()

    wavelength = INST_TO_WAVE["GFA"]

    wok_coords = gfa_to_wok(xPix, yPix, gfaID)

    boresight_icrs = ICRS([[bore_ra, bore_dec]])
    boresight = Observed(
        boresight_icrs,
        site=site,
        wavelength=wavelength,
    )

    wok = Wok([wok_coords], site=site, obsAngle=position_angle)
    focal = FocalPlane(wok, wavelength=wavelength, site=site)
    field = Field(focal, field_center=boresight)
    observed = Observed(field, wavelength=wavelength, site=site)

    return (observed.ra[0], observed.dec[0])


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
