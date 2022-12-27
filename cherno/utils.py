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

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy
import seaborn

from coordio.exceptions import CoordIOUserWarning

from cherno import config
from cherno.exceptions import ChernoError


if TYPE_CHECKING:
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

    for e_d in e_data:
        valid = e_d.regions.loc[e_d.regions.fwhm_valid == 1]
        if len(valid) == 0 or e_d.fwhm_median < 0:
            continue

        cam += [int(e_d.camera[-1])] * len(valid)
        x += [e_d.focus_offset] * len(valid)

        y += valid.fwhm.values.tolist()

        # Calculate the weights as the inverse variance of the FWHM measurement.
        # Also add a subjective estimation of how reliable each camera is.
        if "residual_fit" in e_d.regions:
            residual_fit = valid["residual_fit"]
            ivar = 1 / (residual_fit**2)
            ivar_camera = config["cameras"]["focus_weight"][e_d.camera] * ivar
            weights += ivar_camera.values.tolist()
        else:
            weights += [1.0] * len(valid)

    cam = numpy.array(cam)
    x = numpy.array(x)
    y = numpy.array(y)
    weights = numpy.array(weights)
    weights = weights / weights.max()

    if len(numpy.unique(x)) < 3:
        raise ChernoError("Not enough data points to fit focus.")

    # Perform polynomial fit.
    a, b, c = numpy.polyfit(x, y, 2, w=weights, full=False)

    # Determine the r2 coefficient. See https://bit.ly/3LmH76j
    f_i = a * x**2 + b * x + c
    ybar = numpy.sum(y) / len(y)
    ss_res = numpy.sum((y - f_i) ** 2)
    ss_tot = numpy.sum((y - ybar) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Calculate the focus of the parabola and the associated FWHM.
    x_min = -b / 2 / a
    fwhm_fit = a * x_min**2 + x_min * b + c

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

        with plt.ioff():
            fig, ax = plt.subplots()

            for icam in sorted(numpy.unique(cam)):
                x_cam = x[cam == icam]
                y_cam = y[cam == icam]
                ax.scatter(
                    x_cam,
                    y_cam,
                    marker="o",  # type: ignore
                    edgecolor="None",
                    s=5,
                    label=f"GFA{icam}",
                )

            x0 = numpy.min(x)
            x1 = numpy.max(x)
            xs = numpy.linspace(x0 - 0.1 * (x1 - x0), x1 + 0.1 * (x1 - x0))
            ys = a * xs**2 + b * xs + c
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
