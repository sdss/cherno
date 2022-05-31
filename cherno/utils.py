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
import pathlib
import warnings
from functools import partial

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy
import seaborn
from coordio.defaults import GFA_PIXEL_SIZE, PLATE_SCALE
from coordio.exceptions import CoordIOUserWarning
from coordio.utils import radec2wokxy

from cherno import config
from cherno.coordinates import gfa_to_wok
from cherno.exceptions import ChernoError


if TYPE_CHECKING:
    from cherno.acquisition import AcquisitionData
    from cherno.extraction import ExtractionData


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
    scale_rms: bool = False,
):
    """Fits translation, rotation, and scale from a WCS solution."""

    offset_ra, offset_dec, offset_pa = offset

    xwok_gfa: list[float] = []
    ywok_gfa: list[float] = []
    xwok_astro: list[float] = []
    ywok_astro: list[float] = []

    default_offset = config.get("default_offset", (0.0, 0.0, 0.0))

    for d in data:

        camera_id = int(d.camera[-1])
        xidx, yidx = numpy.meshgrid(
            numpy.linspace(0, 2048, grid[0]),
            numpy.linspace(0, 2048, grid[1]),
        )
        xidx = xidx.flatten()
        yidx = yidx.flatten()

        coords: Any = d.wcs.pixel_to_world(xidx, yidx)
        ra = coords.ra.value
        dec = coords.dec.value

        for x, y in zip(xidx, yidx):
            xw, yw, _ = gfa_to_wok(x, y, camera_id)
            xwok_gfa.append(cast(float, xw))
            ywok_gfa.append(cast(float, yw))

        cos_dec = numpy.cos(numpy.deg2rad(d.field_dec))
        offset_ra_deg = offset_ra / cos_dec / 3600.0
        default_offset_ra_deg = default_offset[0] / cos_dec / 3600.0

        _xwok_astro, _ywok_astro, *_ = radec2wokxy(
            ra,
            dec,
            None,
            "GFA",
            d.field_ra - offset_ra_deg - default_offset_ra_deg,
            d.field_dec - offset_dec / 3600.0 - default_offset[1] / 3600.0,
            d.field_pa - offset_pa / 3600.0 - default_offset[2] / 3600.0,
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

    # delta_x and delta_y only align with RA/Dec if PA=0. Otherwise we need to
    # project using the PA.
    pa_rad = numpy.deg2rad(data[0].field_pa)
    delta_ra = t[0] * numpy.cos(pa_rad) + t[1] * numpy.sin(pa_rad)
    delta_dec = -t[0] * numpy.sin(pa_rad) + t[1] * numpy.cos(pa_rad)

    # Convert to arcsec and round up
    delta_ra = numpy.round(delta_ra / plate_scale * 3600.0, 3)
    delta_dec = numpy.round(delta_dec / plate_scale * 3600.0, 3)

    delta_rot = numpy.round(-numpy.rad2deg(numpy.arctan2(R[1, 0], R[0, 0])) * 3600.0, 1)
    delta_scale = numpy.round(c, 6)

    if scale_rms:
        xwok_astro /= delta_scale
        ywok_astro /= delta_scale

    delta_x = (numpy.array(xwok_gfa) - numpy.array(xwok_astro)) ** 2  # type: ignore
    delta_y = (numpy.array(ywok_gfa) - numpy.array(ywok_astro)) ** 2  # type: ignore

    xrms = numpy.sqrt(numpy.sum(delta_x) / len(delta_x))
    yrms = numpy.sqrt(numpy.sum(delta_y) / len(delta_y))
    rms = numpy.sqrt(numpy.sum(delta_x + delta_y) / len(delta_x))

    # Convert to arcsec and round up
    xrms = numpy.round(xrms / plate_scale * 3600.0, 3)
    yrms = numpy.round(yrms / plate_scale * 3600.0, 3)
    rms = numpy.round(rms / plate_scale * 3600.0, 3)

    return (delta_ra, delta_dec, delta_rot, delta_scale, xrms, yrms, rms)


def focus_fit(
    e_data: list[ExtractionData],
    plot: pathlib.Path | str | bool | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Determines the optimal focus.

    Performs a least-squares polynomial fit to the focus data using
    a quadratic polynomial. Also calculates the r2 coefficient.

    Parameters
    ----------
    e_data
        Extraction data. Must include the observatory, FWHM, focus offset for each
        camera, and camera name.
    plot
        The path where to save the generated plot. If `None`, does not generate
        a plot.

    Returns
    -------
    results
        A tuple with the fitted FWHM, offset to the optimal focus in microns,
        coefficients of the fitted quadratic polynomial, and r2 coefficient.

    """

    def f(x, a, b, c):
        return a * x**2 + b * x + c

    cam = []
    x = []
    y = []
    weights = []

    observatory = e_data[0].observatory
    fwhm_to_microns = config["pixel_scale"][observatory] * GFA_PIXEL_SIZE

    for e_d in e_data:
        valid = e_d.regions.loc[e_d.regions.valid == 1]
        if len(valid) == 0 or e_d.fwhm_median < 0:
            continue

        cam += [int(e_d.camera[-1])] * len(valid)
        x += [e_d.focus_offset] * len(valid)

        # Convert FWHM to microns so that they are the same units as the focus offset.
        fwhm_microns = valid.fwhm * fwhm_to_microns
        y += fwhm_microns.values.tolist()

        # Calculate the weights as the inverse variance of the FWHM measurement.
        # Also add a subjective estimation of how reliable each camera is.
        ivar = 1 / (e_d.regions.loc[e_d.regions.valid == 1, "residual_fit"] ** 2)
        ivar_camera = config["cameras"]["focus_weight"][e_d.camera] * ivar
        weights += ivar_camera.values.tolist()

    cam = numpy.array(cam)
    x = numpy.array(x)
    y = numpy.array(y)
    weights = numpy.array(weights)
    weights = weights / weights.max()

    if len(numpy.unique(x)) < 3:
        raise ChernoError("Not enough data points to fit focus.")

    # Perform polynomial fit.
    a, b, c = numpy.polyfit(x, y, 2, w=weights, full=False)

    # Calculate the focus of the parabola and the associated FWHM.
    x_min = -b / 2 / a
    fwhm_fit = (a * x_min**2 + x_min * b + c) / fwhm_to_microns

    # Determine the r2 coefficient. See https://bit.ly/3LmH76j
    f_i = a * x**2 + b * x + c
    ybar = numpy.sum(y) / len(y)
    ss_res = numpy.sum((y - f_i) ** 2)
    ss_tot = numpy.sum((y - ybar) ** 2)
    r2 = 1 - ss_res / ss_tot

    if plot is not None:

        if isinstance(plot, (str, pathlib.Path)):
            outpath = pathlib.Path(plot)
        else:
            path = e_data[0].path.parent
            mjd = e_data[0].mjd
            seq = e_data[0].exposure_no
            outpath = path / "focus" / f"focus-{mjd}-{seq}.pdf"

        outpath.parent.mkdir(exist_ok=True, parents=True)

        seaborn.set_theme(style="darkgrid", palette="dark")

        with plt.ioff():  # type: ignore
            fig, ax = plt.subplots()

            for icam in sorted(numpy.unique(cam)):
                x_cam = x[cam == icam]
                y_cam = y[cam == icam]
                ax.scatter(
                    x_cam,
                    y_cam / fwhm_to_microns,
                    marker="o",  # type: ignore
                    edgecolor="None",
                    s=5,
                    label=f"GFA{icam}",
                )

            x0 = numpy.min(x)
            x1 = numpy.max(x)
            xs = numpy.linspace(x0 - 0.1 * (x1 - x0), x1 + 0.1 * (x1 - x0))
            ys = (a * xs**2 + b * xs + c) / fwhm_to_microns
            ax.plot(
                xs,
                ys,
                "r-",
                lw=2.0,
                label="Fitted polynomial",
            )

            ax.axvline(
                x=x_min,
                ls="--",
                c="b",
                lw=0.5,
                label="Focus offset (" + str(numpy.round(x_min, 1)) + r"$\rm \,\mu m$)",
            )

            ax.legend()

            ax.set_xlabel("Focus offset [microns]")
            ax.set_ylabel("FWHM [arcsec]")
            ax.set_title(
                f"MJD: {e_data[0].mjd} - Exp No: {e_data[0].exposure_no} - "
                f"FWHM: {numpy.round(fwhm_fit, 3)}  arcsec"
            )

            fig.savefig(str(outpath))

        seaborn.reset_orig()
        plt.close("all")

    return fwhm_fit, x_min, a, b, c, r2
