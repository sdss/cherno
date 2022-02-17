#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-10
# @Filename: coordinates.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

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
