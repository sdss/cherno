#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-07
# @Filename: guider.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import pathlib
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial

from typing import TYPE_CHECKING, Any, Coroutine, Literal

import numpy
import pandas
from astropy.io import fits
from astropy.stats.sigma_clipping import SigmaClip
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from simple_pid.pid import PID

from clu.command import FakeCommand
from coordio import calibration, defaults
from coordio.astrometry import AstrometryNet
from coordio.guide import (
    GuiderFit,
    GuiderFitter,
    SolvePointing,
    cross_match,
    gfa_to_radec,
    radec_to_gfa,
)

from cherno import config, log
from cherno.exceptions import ChernoError
from cherno.extraction import Extraction, ExtractionData, PathLike, apply_calibs
from cherno.lcotcc import apply_correction_lco
from cherno.maskbits import GuiderStatus
from cherno.tcc import apply_axes_correction, apply_focus_correction
from cherno.utils import focus_fit, run_in_executor


if TYPE_CHECKING:
    from cherno.actor import ChernoActor, ChernoCommandType
    from cherno.actor.actor import ChernoState

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings(
    "ignore",
    message="Card is too long, comment will be truncated.",
)
warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = ["Guider", "GuideData", "AstrometricSolution", "AxesPID"]


@dataclass
class GuideData:
    """Data from the acquisition process."""

    camera: str
    extraction_data: ExtractionData
    solved: bool = False
    solve_time: float = -999.0
    solve_method: str = ""
    wcs: WCS = field(default_factory=WCS)
    camera_racen: float = -999.0
    camera_deccen: float = -999.0
    xrot: float = -999.0
    yrot: float = -999.0
    rotation: float = -999.0
    proc_image: pathlib.Path | None = None

    def __post_init__(self):
        self.path = self.extraction_data.path
        self.camera = self.extraction_data.camera
        self.exposure_no = self.extraction_data.exposure_no
        self.obstime = self.extraction_data.obstime
        self.observatory = self.extraction_data.observatory
        self.field_ra = self.extraction_data.field_ra
        self.field_dec = self.extraction_data.field_dec
        self.field_pa = self.extraction_data.field_pa

        match = re.match(r".*([1-6]).*", self.camera)
        if match:
            self.camera_id = int(match.group(1))

    @property
    def e_data(self):
        """Shortcut for ``extraction_data``."""

        return self.extraction_data


@dataclass
class AstrometricSolution:
    """Astrometric solution data."""

    valid_solution: bool
    guide_data: list[GuideData]
    guider_fit: GuiderFit | None = None
    fitter: GuiderFitter | None = None
    delta_ra: float = -999.0
    delta_dec: float = -999.0
    delta_rot: float = -999.0
    delta_scale: float = -999.0
    delta_focus: float = -999.0
    fit_mode: str = "full"
    rms: float = -999.0
    fwhm_median: float = -999.0
    fwhm_fit: float = -999.0
    focus_coeff: list[float] = field(default_factory=lambda: [-999.0] * 3)
    focus_r2: float = -999.0
    camera_rotation: float = -999.0
    correction_applied: list[float] = field(default_factory=lambda: [0.0] * 5)

    @property
    def n_solved(self):
        """Returns the number of solved cameras."""

        return len([gd for gd in self.guide_data if gd.solved])


class AxesPID:
    """Store for the axis PID coefficient."""

    def __init__(self, actor: ChernoActor | None = None):
        self.actor = actor

        self.ra = self.reset("ra")
        self.dec = self.reset("dec")
        self.rot = self.reset("rot")
        self.focus = self.reset("focus")

    def reset(self, axis: str):
        """Restart the PID loop for an axis."""

        if self.actor is None:
            pid_coeffs = config["guide_loop"][axis]["pid"]
        else:
            pid_coeffs = self.actor.state.guide_loop[axis]["pid"]

        return PID(
            Kp=pid_coeffs["k"],
            Ki=pid_coeffs.get("ti", 0),
            Kd=pid_coeffs.get("td", 0),
        )


class Guider:
    """Runs the steps for field acquisition and guiding."""

    def __init__(
        self,
        observatory: str,
        extractor: Extraction | None = None,
        astrometry: AstrometryNet | None = None,
        command: ChernoCommandType | None = None,
        target_rms: float | None = None,
        astrometry_params: dict = {},
        extraction_params: dict = {},
    ):
        self.extractor = extractor or Extraction(observatory, **extraction_params)

        if astrometry is not None:
            self.astrometry = astrometry
        else:
            pixel_scale = config["pixel_scale"]
            astrometry_net_config = config["guider"]["astrometry_net_config"]
            backend_config = pathlib.Path(__file__).parent / astrometry_net_config
            self.astrometry = AstrometryNet(
                backend_config=str(backend_config),
                width=2048,
                height=2048,
                no_plots=True,
                scale_low=pixel_scale * 0.9,
                scale_high=pixel_scale * 1.1,
                scale_units="arcsecperpix",
                sort_column="flux",
                radius=0.5,
                cpulimit=config["guider"]["cpulimit"],
                **astrometry_params,
            )

        self.observatory = observatory.upper()
        self.fitter = GuiderFitter(self.observatory)
        self.command = command or FakeCommand(log)
        self.target_rms = target_rms
        self.pids = AxesPID(command.actor if command is not None else None)

        # reference to solve pointing instance, initialize to None
        self.solve_pointing: SolvePointing | None = None

        # To cache Gaia sources.
        self._gaia_sources: dict = {}

        # Clear RMS history.
        if self.command.actor:
            self.command.actor.state.rms_history.clear()

        # setup task queue for writing proc-*.fits files
        self.write_background_tasks = set()
        self.write_executor = ProcessPoolExecutor(
            max_workers=3,
            mp_context=mp.get_context("spawn"),
        )

        self._database_lock = asyncio.Lock()

    def set_command(self, command: ChernoCommandType):
        """Sets the command."""

        self.command = command

    def check_convergence(
        self,
        ex_data: list[ExtractionData],
        offset: list[float] | None,
    ):
        """Determine whether the guider has converged."""

        if self.solve_pointing is None:
            return False

        nSources = numpy.sum([len(ex.regions) for ex in ex_data])
        if nSources < 6:
            return False

        ex = ex_data[0]

        assert self.solve_pointing and self.solve_pointing.imgNum is not None

        if ex.exposure_no != self.solve_pointing.imgNum + 1:
            return False

        if self.solve_pointing.guide_rms_sky > 1:
            return False

        if self.solve_pointing._raCen != ex.field_ra:
            return False

        if self.solve_pointing._decCen != ex.field_dec:
            return False

        if self.solve_pointing._paCen != ex.field_pa:
            return False

        full_offset = self.get_full_offset(offset)

        if self.solve_pointing.offset_ra != full_offset[0]:
            return False

        if self.solve_pointing.offset_dec != full_offset[1]:
            return False

        if self.solve_pointing.offset_pa != full_offset[2]:
            return False

        return True

    def check_wcs(self, guide_data: list[GuideData]):
        """Returns WCS objects for cameras that solved."""

        guide_cameras = self.command.actor.state.guide_cameras
        wcs_solved = [
            ad.camera
            for ad in guide_data
            if ad.solved is True and ad.camera in guide_cameras
        ]
        if len(wcs_solved) > 1:
            # if 2 or more cameras have wcs solutions, use "coordio"
            for gd in guide_data:
                gd.solved = True
                gd.solve_method = "coordio"

        return wcs_solved

    def try_rapid_solve(
        self,
        guide_data: list[GuideData],
        ext_data: list[ExtractionData],
        offset: list[float] | None,
    ):
        sp_guider_fit = None
        converged = self.check_convergence(ext_data, offset)

        if not self.solve_pointing:
            return None

        if converged:
            dfList = []
            for ex in ext_data:
                regions = ex.regions.copy().reset_index(drop=True)
                if len(regions) == 0:
                    continue
                gfaNum = int(ex.camera.strip("gfa"))
                regions["gfaNum"] = gfaNum
                gain = ex.gain
                regions["gain"] = gain
                dfList.append(regions)

            df = pandas.concat(dfList).reset_index(drop=True)

            # assert self.solve_pointing
            sp_guider_fit = self.solve_pointing.reSolve(
                ex.exposure_no,
                ex.obstime,
                ex.exptime,
                df,
            )

            # assert self.solve_pointing
            if self.solve_pointing.guide_rms_sky > 1:
                # guider probably jumped safer to revert to slow solve
                sp_guider_fit = None
                self.solve_pointing = None
            else:
                for gd in guide_data:
                    gfaNum = int(gd.camera.strip("gfa"))
                    gd.solved = True
                    wcs = self.solve_pointing.fitWCS(gfaNum)
                    gd.wcs = wcs
                    gd.solve_method = "coordio-fast"

        return sp_guider_fit

    def validate_astrometry(self, data: list[GuideData], offset: list[float] | None):
        """Checks if the astrometry.net solution is valid."""

        isOK = False

        ast_solution = AstrometricSolution(False, data)

        solved = sorted([d for d in data if d.solved is True], key=lambda x: x.camera)
        weights = [s.extraction_data.nvalid for s in solved]

        if len(solved) == 0:
            self.command.error(acquisition_valid=False, did_correct=False)
            return ast_solution, isOK

        fwhm = [d.e_data.fwhm_median for d in data if d.e_data.fwhm_median > 0]
        fwhm_weights = [
            1 / numpy.abs(d.e_data.focus_offset or 1.0)
            for d in data
            if d.e_data.fwhm_median > 0
        ]
        fwhm = numpy.average(fwhm, weights=fwhm_weights)

        camera_rotation = numpy.average([d.rotation for d in solved], weights=weights)

        ast_solution.fwhm_median = numpy.round(float(fwhm), 3)
        ast_solution.camera_rotation = numpy.round(float(camera_rotation), 2)

        if solved[0].field_ra == "NaN" or isinstance(solved[0].field_ra, str):
            self.command.error(acquisition_valid=False, did_correct=False)
            self.command.error("Field not defined. Cannot run astrometric fit.")
            return ast_solution, isOK

        isOK = True

        if offset is None:
            if self.command.actor:
                offset = list(self.command.actor.state.offset)
            else:
                offset = list(config.get("offset", [0.0, 0.0, 0.0]))

        if not numpy.allclose(offset, [0.0, 0.0, 0.0]):
            self.command.warning("Using non-zero offsets for astrometric fit.")

        return ast_solution, isOK

    def get_full_offset(self, offset: list[float] | None):
        """Outputs and returns the full offset (absolute plus user-commanded)."""

        if offset is None:
            if self.command.actor:
                offset = list(self.command.actor.state.offset)
            else:
                offset = list(config.get("offset", [0.0, 0.0, 0.0]))

        default_offset = config.get("default_offset", (0.0, 0.0, 0.0))
        full_offset = numpy.array(offset) + numpy.array(default_offset)

        self.command.debug(default_offset=default_offset)
        if any(offset):
            self.command.warning(offset=offset)
        else:
            self.command.debug(offset=offset)

        return full_offset

    def update_ast_solution(
        self,
        ast_solution: AstrometricSolution,
        data: list[GuideData],
        guider_fit: GuiderFit | Literal[False],
        fit_focus: bool,
    ):
        guide_cameras = self.command.actor.state.guide_cameras
        solved = sorted([d for d in data if d.solved is True], key=lambda x: x.camera)
        # fit_cameras1 = [d.camera_id for d in solved if d.camera in guide_cameras]
        fit_cameras = []
        for d in solved:
            if d.camera not in guide_cameras:
                continue
            if d.wcs is None:
                continue
            fit_cameras.append(d.camera_id)

        plate_scale = defaults.PLATE_SCALE[self.observatory]
        mm_to_arcsec = 1 / plate_scale * 3600

        exp_no = solved[0].exposure_no  # Should be the same for all.

        if guider_fit is not False:
            # If we have a guider_fit != False, the fit produced a good astrometric
            # solution.
            ast_solution.valid_solution = True

            # Now unpack fit information. We do this even if we reject the
            # fit below because we want the fit data in the headers, but
            # won't apply the correction.
            delta_ra = guider_fit.delta_ra
            delta_dec = guider_fit.delta_dec
            delta_rot = guider_fit.delta_rot
            delta_scale = guider_fit.delta_scale

            xrms = numpy.round(guider_fit.xrms * mm_to_arcsec, 3)
            yrms = numpy.round(guider_fit.yrms * mm_to_arcsec, 3)
            rms = numpy.round(guider_fit.rms * mm_to_arcsec, 3)

            ast_solution.guider_fit = guider_fit
            ast_solution.fitter = self.fitter

            self.command.info(guide_rms=[exp_no, xrms, yrms, rms])

            # Store RMS. This is used to determine acquisition convergence.
            if self.command.actor and rms > 0 and rms < 1:
                self.command.actor.state.rms_history.append(rms)

            if guider_fit.only_radec:
                ast_solution.fit_mode = "radec"

            # Report fit RMS. First value is the global fit RMS, then one for
            # each camera and a boolean indicating if that camera was used for the
            # global fit. If a camera was rejected the fit RMS is set to -999.
            fit_rms = guider_fit.fit_rms
            fit_rms_global = float(numpy.round(fit_rms.loc[0].rms * mm_to_arcsec, 4))
            fit_rms_camera: list[int | float] = [fit_rms_global]
            for cid in range(1, 7):
                if cid in fit_rms.index:
                    this_fit_rms = numpy.round(fit_rms.loc[cid].rms * mm_to_arcsec, 4)
                    fit_rms_camera.append(float(this_fit_rms))
                else:
                    fit_rms_camera.append(-999.0)
                fit_rms_camera.append(cid in guider_fit.cameras)

            self.command.info(fit_rms_camera=fit_rms_camera)

            # If the fit_rms of all the fit cameras is greater than a certain
            # threshold, reject the fit. This can happen in cases when the
            # telescope is moving during an exposure.
            max_fit_rms = config["guider"]["max_fit_rms"]  # In arcsec
            if guider_fit:
                fit_rms = guider_fit.fit_rms.loc[fit_cameras, "rms"] * mm_to_arcsec
                if numpy.all(fit_rms > max_fit_rms):
                    self.command.warning(
                        "The fit RMS of all the cameras exceeds "
                        "threshold values. Rejecting fit."
                    )
                    ast_solution.valid_solution = False

            # Check the delta_scale. If the change is too large, this is probably
            # a misfit. Reject the fit.
            max_delta_scale_ppm = config["guider"]["max_delta_scale_ppm"]
            delta_scale_ppm = abs(1 - guider_fit.delta_scale) * 1e6  # Parts per million
            if delta_scale_ppm > max_delta_scale_ppm:
                self.command.warning("Scale change exceeds limits. Rejecting fit.")
                ast_solution.valid_solution = False

        else:
            rms = delta_ra = delta_dec = delta_rot = delta_scale = -999.0
            ast_solution.guider_fit = None
            ast_solution.valid_solution = False

        # Update AstrometricSolution object.
        ast_solution.delta_ra = float(delta_ra)
        ast_solution.delta_dec = float(delta_dec)
        ast_solution.delta_rot = float(delta_rot)
        ast_solution.delta_scale = float(delta_scale)
        ast_solution.rms = float(rms)

        if fit_focus:
            focus_cameras = self.command.actor.state.focus_cameras
            try:
                fwhm_fit, x_min, a, b, c, r2 = focus_fit(
                    [d.e_data for d in data if d.camera.lower() in focus_cameras],
                    plot=config["guider"]["plot_focus"],
                )

                # Relationship between M2 move and focal plane. See
                # http://www.loptics.com/ATM/mirror_making/cass_info/cass_info.html
                focus_sensitivity = config["focus_sensitivity"]

                ast_solution.fwhm_fit = round(fwhm_fit, 3)
                ast_solution.delta_focus = round(-x_min / focus_sensitivity, 1)
                ast_solution.focus_coeff = [a, b, c]
                ast_solution.focus_r2 = round(r2, 3)

            except Exception as err:
                self.command.warning(f"Failed fitting focus curve: {err}.")

        ra_bore = dec_bore = -999.0
        if self.solve_pointing:
            ra_bore = self.solve_pointing.raCenMeas
            dec_bore = self.solve_pointing.decCenMeas

        self.command.info(
            astrometry_fit=[
                exp_no,
                len(solved),
                ra_bore,
                dec_bore,
                numpy.round(ast_solution.fwhm_median, 2),
                -999,
                numpy.round(ast_solution.camera_rotation, 3),
                delta_ra,
                delta_dec,
                delta_rot,
                delta_scale,
            ]
        )

        # Output focus data in a single keyword, mostly for Boson's benefit.
        focus_data = [int(exp_no)]
        for d in ast_solution.guide_data:
            if d.extraction_data.fwhm_median > 0 and d.extraction_data.nvalid > 0:
                focus_data += [
                    int(d.camera[-1]),
                    d.extraction_data.fwhm_median,
                    d.extraction_data.focus_offset,
                ]

        self.command.debug(focus_data=focus_data)

        if fit_focus:
            self.command.info(
                focus_fit=[
                    exp_no,
                    ast_solution.fwhm_fit,
                    float(f"{ast_solution.focus_coeff[0]:.3e}"),
                    float(f"{ast_solution.focus_coeff[1]:.3e}"),
                    float(f"{ast_solution.focus_coeff[2]:.3e}"),
                    ast_solution.focus_r2,
                    ast_solution.delta_focus,
                ]
            )

        if (
            delta_scale > 0
            and self.command.actor
            and ast_solution.fwhm_median < 2.5
            and rms > 0
            and rms <= 1
            and guider_fit
            and not guider_fit.only_radec
        ):
            # If we measured the scale, add it to the actor state. This is later
            # used to compute the average scale over a period. We also add the time
            # because we'll want to reject measurements that are too old.
            self.command.actor.state.scale_history.append((time.time(), delta_scale))

        if config["guider"].get("plot_rms", False):
            e_data_test = data[0].e_data
            path = e_data_test.path.parent
            mjd = e_data_test.mjd
            seq = e_data_test.exposure_no
            outpath = path / "fit" / f"fit_rms-{mjd}-{seq}.pdf"
            outpath.parent.mkdir(exist_ok=True)
            self.fitter.plot_fit(outpath)

    async def process(
        self,
        command: ChernoCommandType | None,
        images: list[PathLike],
        write_proc: bool = True,
        overwrite: bool = False,
        correct: bool = True,
        full_correction: bool = False,
        offset: list[float] | None = None,
        scale_rms: bool = True,
        wait_for_correction: bool = True,
        only_radec: bool = False,
        auto_radec_min: int = 2,
        use_astrometry_net: bool | None = None,
        use_gaia: bool = True,
        gaia_phot_g_mean_mag_max: float | None = None,
        gaia_cross_correlation_blur: float | None = None,
        fit_all_detections: bool = True,
        fit_focus: bool = True,
        plot: bool | None = None,
        stop_at_target_rms: bool = False,
    ):
        """Performs extraction and astrometry."""

        if command is not None:
            self.set_command(command)

        # Get the state of the boss spectrograph up front
        bossExposing = False
        bossExpNum = -999

        if self.command.actor is not None:
            if hasattr(self.command.actor, "_process_boss_status"):
                bossExposing, bossExpNum = self.command.actor._process_boss_status()

        self.command.info("Extracting sources.")
        ext_data: list[ExtractionData] = await asyncio.gather(
            *[
                run_in_executor(
                    self.extractor.process,
                    im,
                    plot=plot,
                    executor="process",
                )
                for im in images
            ]
        )

        for d in ext_data:
            if d.nvalid == 0:
                self.command.warning(f"Camera {d.camera}: not enough sources.")
            else:
                self.command.info(
                    fwhm_camera=[
                        d.camera,
                        d.exposure_no,
                        d.fwhm_median,
                        d.nregions,
                        d.nvalid,
                    ]
                )

            # Check CCD temperature
            if d.ccd_temperature is None:
                self.command.warning(f"Camera {d.camera}: CCD temperature not found.")
            elif d.ccd_temperature > config["cameras"]["max_ccd_temperature"]:
                self.command.warning(f"Camera {d.camera}: CCD temperature is too high.")

        guide_data = [GuideData(ed.camera, ed) for ed in ext_data]

        _wcs_solved = []

        sp_guider_fit = self.try_rapid_solve(guide_data, ext_data, offset)

        # try:
        #     sp_guider_fit = self.try_rapid_solve(guide_data, ext_data, offset)
        # except Exception as err:
        #     self.command.warning(f"Rapid solve failed: {err}")
        #     self.command.warning("Reverting to non-converged mode.")
        #     sp_guider_fit = None

        if sp_guider_fit is None:
            # Rapid solve didn't work
            use_astrometry_net = (
                use_astrometry_net
                if use_astrometry_net is not None
                else config["guider"].get("use_astrometry_net", True)
            )

            if use_astrometry_net:
                self.command.info("Running astrometry.net.")
                guide_data = await asyncio.gather(
                    *[self._astrometry_one(d) for d in ext_data]
                )

            # check for astrometry.net solutions for cameras actively used
            # for guide corrections
            # will modify solved attr on guide_data(s)
            # if more than 1 wcs exists

            _wcs_solved = self.check_wcs(guide_data)

            # Use Gaia cross-match for the cameras that did not solve with
            # astrometry.net.
            not_solved = [ad for ad in guide_data if ad.solved is False]
            if use_gaia and len(not_solved) > 0:
                self.command.info("Running Gaia cross-match.")
                res = await asyncio.gather(
                    *[
                        self._gaia_cross_match_one(
                            ad,
                            gaia_phot_g_mean_mag_max=gaia_phot_g_mean_mag_max,
                            gaia_cross_correlation_blur=gaia_cross_correlation_blur,
                            gaia_table_name=config["guider"]["gaia_connection_table"],
                            gaia_connection_string=config["guider"][
                                "gaia_connection_string"
                            ],
                        )
                        for ad in not_solved
                    ],
                    return_exceptions=True,
                )

                for ii, rr in enumerate(res):
                    if isinstance(rr, Exception):
                        cam = not_solved[ii].camera
                        self.command.warning(
                            f"{cam}: Gaia cross-match failed: {str(rr)}"
                        )

                # check for xmatch solutions for cameras actively used
                # for guide corrections
                # will modify solved attr on guide_data(s)
                # if more than 1 wcs exists

                _wcs_solved = self.check_wcs(guide_data)

        # Output all the camera_solution keywords at once.
        for ad in guide_data:
            self.output_camera_solution(ad)

        ast_solution, isOK = self.validate_astrometry(guide_data, offset)

        if isOK:
            # continue with guide fitting
            full_offset: list[float] = self.get_full_offset(offset).tolist()

            if only_radec is True:
                self.command.warning(
                    "Only fitting RA/Dec. The rotation and scale offsets "
                    "are informational-only and not corrected."
                )

            if sp_guider_fit:
                # field was already solved in fast mode
                guider_fit = sp_guider_fit

            elif len(_wcs_solved) > 1:
                guider_fit = await self.fit_SP(
                    list(guide_data),
                    wcs_solved=_wcs_solved,
                    full_offset=full_offset,
                )

            else:
                guider_fit = await self.fit(
                    list(guide_data),
                    full_offset=full_offset,
                    scale_rms=scale_rms,
                    only_radec=only_radec,
                    auto_radec_min=auto_radec_min,
                    fit_all_detections=fit_all_detections,
                )

            self.update_ast_solution(
                ast_solution,
                list(guide_data),
                guider_fit,
                fit_focus,
            )

            if (
                self.target_rms is not None
                and ast_solution.valid_solution
                and ast_solution.rms > 0
                and ast_solution.rms <= self.target_rms
                and stop_at_target_rms
            ):
                self.command.warning("RMS has been reached. Not applying correction.")
                correct = False

        if correct and ast_solution.valid_solution is True:
            await self.correct(
                ast_solution,
                full=full_correction,
                wait_for_correction=wait_for_correction,
                apply_focus=fit_focus,
            )
        else:
            self.command.info(
                acquisition_valid=ast_solution.valid_solution,
                did_correct=any(ast_solution.correction_applied),
                correction_applied=ast_solution.correction_applied,
            )

        # write files in the background and move on

        if write_proc and self.command.actor:
            self.command.debug("Saving proc- files.")

            for d in guide_data:
                header_updates = get_proc_headers(
                    ast_solution, self.command.actor.state, d
                )
                gaia_matches = None
                if self.solve_pointing is not None:
                    header_updates += self.solve_pointing.getMetadata()
                    zp = self.solve_pointing.median_zeropoint(d.camera_id)
                    if numpy.isnan(zp):
                        zp = -999
                    zptTuple = (
                        "R_ZPT",
                        zp,
                        "median measured zeropoint of gaia sources",
                    )
                    header_updates.append(zptTuple)
                    gaia_matches = self.solve_pointing.matchedSources.copy()

                func = partial(
                    write_proc_image,
                    d,
                    overwrite=overwrite,
                    bossExposing=bossExposing,
                    bossExpNum=bossExpNum,
                    header_updates=header_updates,
                    gaia_matches=gaia_matches,
                )

                task = self.write_executor.submit(func)

                # Fire and forget (eg no checking that the fits was actually written ok)
                self.write_background_tasks.add(task)
                task.add_done_callback(self.write_background_tasks.discard)

        return ast_solution

    async def fit(
        self,
        data: list[GuideData],
        full_offset: list[float],
        scale_rms: bool = False,
        only_radec: bool = False,
        auto_radec_min: int = 2,
        fit_all_detections: bool = True,
    ):
        """Calculate the astrometric solution."""

        solved = sorted([d for d in data if d.solved is True], key=lambda x: x.camera)

        if auto_radec_min >= 0 and len(solved) <= auto_radec_min:
            only_radec = True
            self.command.warning(
                f"Only {len(solved)} cameras solved. Only fitting RA/Dec. "
                "The rotation and scale offsets are informational-only "
                "and not corrected."
            )

        self.fitter.reset()
        for d in solved:
            if fit_all_detections:
                regions = d.extraction_data.regions
                pixels = regions.loc[:, ["x", "y"]].copy().values
            else:
                pixels = None
            self.fitter.add_wcs(d.camera, d.wcs, d.obstime.jd, pixels=pixels)

        field_ra = solved[0].field_ra
        field_dec = solved[0].field_dec
        field_pa = solved[0].field_pa

        guide_cameras = self.command.actor.state.guide_cameras
        fit_cameras = [d.camera_id for d in solved if d.camera in guide_cameras]
        fit_rms_sigma = config["guider"].get("fit_rms_sigma", 3)
        guider_fit = False
        while True:
            tmp_guider_fit = self.fitter.fit(
                field_ra,
                field_dec,
                field_pa,
                offset=numpy.array(full_offset),
                scale_rms=scale_rms,
                only_radec=only_radec,
                cameras=fit_cameras,
            )

            # If we already had a solution and this fit failed or the fit RMS is bad,
            # just use the previous fit.
            if guider_fit is not False and (fit_rms_sigma <= 0 or not tmp_guider_fit):
                break

            # Update the fit with the previous one.
            guider_fit = tmp_guider_fit

            # If the fit failed, exit.
            if guider_fit is False:
                break

            # If only 3 cameras remain we exit. We don't want to reject any more.
            if len(fit_cameras) <= 3:
                break

            sc = SigmaClip(fit_rms_sigma)
            rms_clip = sc(guider_fit.fit_rms.loc[fit_cameras, "rms"])
            assert isinstance(rms_clip, numpy.ma.MaskedArray)

            # All the camera fit RMS are within X sigma. Exit.
            if rms_clip.mask.sum() == 0:
                break

            # Find the camera with the largest fit RMS and remove it. Redo the fit.
            cam_max_rms = int(guider_fit.fit_rms.loc[fit_cameras, "rms"].idxmax())

            self.command.warning(
                "Fit RMS found outlier detections. "
                f"Refitting without camera {cam_max_rms}."
            )
            fit_cameras.remove(cam_max_rms)

        return guider_fit

    async def fit_SP(
        self,
        data: list[GuideData],
        wcs_solved: list[str],
        full_offset: list[float],
    ):
        """Calculate the astrometric solution."""

        solved = sorted([d for d in data if d.solved is True], key=lambda x: x.camera)

        field_ra = solved[0].field_ra
        field_dec = solved[0].field_dec
        field_pa = solved[0].field_pa

        guide_cameras = self.command.actor.state.guide_cameras

        self.solve_pointing = SolvePointing(
            raCen=field_ra,
            decCen=field_dec,
            paCen=field_pa,
            offset_ra=full_offset[0],
            offset_dec=full_offset[1],
            offset_pa=full_offset[2],
            db_conn_st=config["guider"]["gaia_connection_string"],
            db_tab_name=config["guider"]["gaia_connection_table"],
        )
        for d in data:
            if d.camera not in guide_cameras:
                continue
            if d.extraction_data.nregions == 0:
                continue
            wcs = None
            if d.camera in wcs_solved:
                wcs = d.wcs

            self.solve_pointing.add_gimg(
                d.path,
                d.extraction_data.regions.copy(),
                wcs,
                d.extraction_data.gain,
            )

        guider_fit = self.solve_pointing.solve()

        for gd in data:
            gfaNum = int(gd.camera.strip("gfa"))
            wcs = self.solve_pointing.fitWCS(gfaNum)
            gd.wcs = wcs

        return guider_fit

    async def correct(
        self,
        data: AstrometricSolution,
        full: bool = False,
        wait_for_correction: bool = True,
        apply_focus: bool = True,
    ):
        """Runs the astrometric fit"""

        if not self.command.actor or self.command.status.is_done:
            raise ChernoError("Cannot run correct without a valid running command.")

        actor_state = self.command.actor.state
        guider_status = actor_state.status

        stopping = guider_status & (GuiderStatus.STOPPING | GuiderStatus.IDLE)

        if stopping:
            data.correction_applied = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.command.info(acquisition_valid=True, did_correct=False)
            return True

        self.command.info("Applying corrections.")

        if self.observatory == "APO":
            await self._correct_apo(data, full=full, apply_focus=apply_focus)
        else:
            await self._correct_lco(
                data,
                full=full,
                wait_for_correction=wait_for_correction,
                apply_focus=apply_focus,
            )

    async def _correct_apo(
        self,
        data: AstrometricSolution,
        full: bool = False,
        apply_focus: bool = True,
    ):
        """Apply corrections at APO."""

        actor_state = self.command.actor.state

        min_isolated = actor_state.guide_loop["rot"]["min_isolated_correction"]
        if abs(data.delta_rot) >= min_isolated:
            self.command.debug("Applying only large rotator correction.")
            coro = apply_axes_correction(
                self.command,
                self.pids,
                delta_rot=data.delta_rot if data.fit_mode == "full" else None,
                full=full,
            )

        else:
            coro = apply_axes_correction(
                self.command,
                self.pids,
                delta_radec=(data.delta_ra, data.delta_dec),
                delta_rot=data.delta_rot if data.fit_mode == "full" else None,
                full=full,
            )

        correct_tasks: list[Coroutine[Any, Any, Any]] = [coro]

        do_focus: bool = False

        # Ignore focus correction when the r2 correlation is bad or when we got
        # an inverted parabola.
        if (
            apply_focus
            and data.focus_r2 > config["guider"]["focus_r2_threshold"]
            and data.focus_coeff[0] > 0
        ):
            do_focus = True
            coro = apply_focus_correction(self.command, self.pids, data.delta_focus)
            correct_tasks.append(coro)
        elif apply_focus:
            self.command.warning("Focus fit poorly constrained. Not correcting focus.")

        self.command.actor.state.set_status(GuiderStatus.CORRECTING, mode="add")
        applied_corrections: Any = await asyncio.gather(*correct_tasks)
        self.command.actor.state.set_status(GuiderStatus.CORRECTING, mode="remove")

        data.correction_applied[:4] = applied_corrections[0]
        if do_focus:
            data.correction_applied[4] = applied_corrections[1] or 0.0
        else:
            data.correction_applied[4] = 0.0

        self.command.info(
            acquisition_valid=True,
            did_correct=any(data.correction_applied),
            correction_applied=data.correction_applied,
        )

    async def _correct_lco(
        self,
        data: AstrometricSolution,
        full: bool = False,
        wait_for_correction: bool = True,
        apply_focus: bool = True,
    ):
        """Apply corrections at LCO"""

        do_focus: bool = False

        enabled_axes = self.command.actor.state.enabled_axes

        # Ignore focus correction when the r2 correlation is bad or when we got
        # an inverted parabola.
        if (
            apply_focus
            and data.focus_r2 > config["guider"]["focus_r2_threshold"]
            and data.focus_coeff[0] > 0
            and "focus" in enabled_axes
        ):
            do_focus = True
        elif apply_focus:
            self.command.warning("Focus fit poorly constrained. Not correcting focus.")

        self.command.actor.state.set_status(GuiderStatus.CORRECTING, mode="add")

        try:
            applied_corrections: Any = await apply_correction_lco(
                self.command,
                self.pids,
                delta_radec=(data.delta_ra, data.delta_dec),
                delta_rot=data.delta_rot if data.fit_mode == "full" else None,
                delta_focus=data.delta_focus if do_focus else None,
                full=full,
                wait_for_correction=wait_for_correction,
            )
        except ChernoError as err:
            self.command.warning(f"Failed applying correction: {err}")
            applied_corrections = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.command.actor.state.set_status(GuiderStatus.CORRECTING, mode="remove")

        data.correction_applied = applied_corrections

        self.command.info(
            acquisition_valid=True,
            did_correct=any(data.correction_applied),
            correction_applied=data.correction_applied,
        )

    async def _astrometry_one(self, ext_data: ExtractionData):
        if config["guider"]["astrometry_net_use_all_regions"]:
            regions = ext_data.regions.copy()
        else:
            regions = ext_data.regions.loc[ext_data.regions.valid == 1].copy()

        if self.astrometry._options.get("dir", None) is None:
            astrometry_dir = pathlib.Path(config["guider"]["astrometry_dir"])
            if astrometry_dir.is_absolute():
                pass
            else:
                astrometry_dir = ext_data.path.parent / astrometry_dir

            astrometry_dir.mkdir(exist_ok=True, parents=True)
        else:
            astrometry_dir = pathlib.Path(self.astrometry._options["dir"])

        outfile_root = astrometry_dir / ext_data.path.stem

        # We use all detections, even invalid ones here.
        xyls_df = regions.loc[:, ["x1", "y1", "flux"]].copy()

        # Rename columns to the names astrometry.net expects.
        xyls_df.rename(columns={"x1": "x", "y1": "y"}, inplace=True)

        gfa_xyls = Table.from_pandas(xyls_df)
        gfa_xyls_path = outfile_root.with_suffix(".xyls")
        gfa_xyls.write(str(gfa_xyls_path), format="fits", overwrite=True)

        wcs_output = outfile_root.with_suffix(".wcs")
        wcs_output.unlink(missing_ok=True)

        camera_id = int(ext_data.camera[-1])
        radec_centre = gfa_to_radec(
            ext_data.observatory,
            1024,
            1024,
            camera_id,
            ext_data.field_ra,
            ext_data.field_dec,
            position_angle=ext_data.field_pa,
        )

        if self.command.actor and self.command.actor.state:
            odds_to_solve = 10**self.command.actor.state.astrometry_net_odds
        else:
            odds_to_solve = None

        proc = await self.astrometry.run_async(
            [gfa_xyls_path],
            stdout=outfile_root.with_suffix(".stdout"),
            stderr=outfile_root.with_suffix(".stderr"),
            ra=radec_centre[0],
            dec=radec_centre[1],
            odds_to_solve=odds_to_solve,
            radius=1,
        )

        guide_data = GuideData(ext_data.camera, ext_data, solve_time=proc.elapsed)

        if wcs_output.exists():
            guide_data.solved = True
            wcs = WCS(open(wcs_output).read())
            guide_data.wcs = wcs
            guide_data.solve_method = "astrometry.net"

        return guide_data

    def output_camera_solution(self, guide_data: GuideData):
        """Calculates and outputs the camera_solution keyword."""

        wcs = guide_data.wcs

        if wcs and hasattr(wcs.wcs, "cd") and guide_data.solved:
            racen, deccen = wcs.pixel_to_world_values([[1024, 1024]])[0]
            guide_data.camera_racen = float(numpy.round(racen, 6))
            guide_data.camera_deccen = float(numpy.round(deccen, 6))

            # TODO: consider parallactic angle here.
            cd: numpy.ndarray = wcs.wcs.cd
            rot_rad = numpy.arctan2(
                numpy.array([cd[0, 1], cd[0, 0]]),
                numpy.array([cd[1, 1], cd[1, 0]]),
            )

            # Rotation is from N to E to the x and y axes of the GFA.
            yrot, xrot = numpy.rad2deg(rot_rad)
            guide_data.xrot = float(numpy.round(xrot % 360.0, 3))
            guide_data.yrot = float(numpy.round(yrot % 360.0, 3))

            # Calculate field rotation.
            cameras = config["cameras"]
            camera_rot = cameras["rotation"][guide_data.camera]
            rotation = numpy.array([xrot - camera_rot - 90, yrot - camera_rot]) % 360
            rotation[rotation > 180.0] -= 360.0
            rotation = numpy.mean(rotation)
            guide_data.rotation = float(numpy.round(rotation, 3))

        self.command.info(
            camera_solution=[
                guide_data.camera,
                guide_data.exposure_no,
                guide_data.solved,
                guide_data.camera_racen,
                guide_data.camera_deccen,
                guide_data.xrot,
                guide_data.yrot,
                guide_data.rotation,
                guide_data.solve_method,
            ]
        )

    async def _gaia_cross_match_one(
        self,
        guide_data: GuideData,
        gaia_phot_g_mean_mag_max: float | None = None,
        gaia_cross_correlation_blur: float | None = None,
        gaia_table_name: str = "catalogdb.gaia_dr2_source_g19",
        gaia_connection_string: str = config["guider"]["gaia_connection_string"],
    ):
        """Solves a field cross-matching to Gaia."""

        cam = guide_data.camera

        regions = guide_data.extraction_data.regions.copy()
        regions = regions.loc[regions.valid == 1]

        xy_regions = regions.loc[:, ["x1", "y1", "flux"]].copy()

        if len(xy_regions) < 4:
            self.command.warning(f"{cam}: too few sources. Cannot cross-match to Gaia.")
            return

        acq_config = config["guider"]

        ra = guide_data.extraction_data.field_ra
        dec = guide_data.extraction_data.field_dec
        pa = guide_data.extraction_data.field_pa

        if self.command.actor is not None:
            offsets = self.command.actor.state.offset
        else:
            offsets = [0.0] * 3

        default_offset = config.get("default_offset", (0.0, 0.0, 0.0))

        offra = default_offset[0] + offsets[0]
        offdec = default_offset[1] + offsets[1]
        offpa = default_offset[2] + offsets[2]

        cam_id = int(guide_data.camera[-1])
        obstime_jd = guide_data.extraction_data.obstime.jd

        ccd_centre = gfa_to_radec(
            self.observatory,
            1024,
            1024,
            cam_id,
            ra,
            dec,
            pa,
            offra,
            offdec,
            offpa,
            obstime_jd,
            icrs=True,
        )

        gaia_search_radius = acq_config["gaia_search_radius"]
        g_mag = gaia_phot_g_mean_mag_max or acq_config["gaia_phot_g_mean_mag_max"]

        fid = guide_data.extraction_data.field_id
        if fid != -999 and (fid, cam_id) in self._gaia_sources:
            gaia_stars = self._gaia_sources[(fid, cam_id)]

        else:
            gaia_stars = pandas.read_sql(
                f"SELECT * FROM {gaia_table_name} "
                "WHERE q3c_radial_query(ra, dec, "
                f"{ccd_centre[0]}, {ccd_centre[1]}, {gaia_search_radius}) AND "
                f"phot_g_mean_mag < {g_mag}",
                gaia_connection_string,
            )
            self._gaia_sources[(fid, cam_id)] = gaia_stars

        gaia_x, gaia_y = radec_to_gfa(
            self.observatory,
            numpy.array(gaia_stars["ra"].values),
            numpy.array(gaia_stars["dec"].values),
            cam_id,
            ra,
            dec,
            pa,
            offra,
            offdec,
            offpa,
            obstime_jd,
        )

        gaia = numpy.vstack([gaia_x, gaia_y, gaia_stars.ra, gaia_stars.dec]).T
        gaia = gaia[
            (gaia[:, 0] >= 0)
            & (gaia[:, 0] < 2048)
            & (gaia[:, 1] >= 0)
            & (gaia[:, 1] < 2048)
        ]

        shift = acq_config["gaia_use_cross_correlation_shift"]
        blur = gaia_cross_correlation_blur or acq_config["gaia_cross_correlation_blur"]
        distance_upper_bound = acq_config["gaia_distance_upper_bound"]
        min_error = acq_config["gaia_cross_correlation_min_error"]

        loop = asyncio.get_running_loop()
        cross_match_func = partial(
            cross_match,
            xy_regions.values[:, :2],
            gaia[:, 0:2],
            gaia[:, 2:],
            2048,
            2048,
            blur=blur,
            upsample_factor=100,
            cross_corrlation_shift=shift,
            distance_upper_bound=distance_upper_bound,
        )
        wcs, error = await loop.run_in_executor(None, cross_match_func)

        if wcs is None:
            # Failed probably because not enough independent measurements
            pass
        elif shift and error < min_error:
            self.command.warning(f"{cam}: cross-matching error {error}. Cannot solve.")
        else:
            guide_data.solved = True
            guide_data.solve_method = "gaia"
            guide_data.wcs = wcs


def get_proc_headers(
    data: AstrometricSolution,
    guider_state: ChernoState,
    guide_data: GuideData,
):
    """Builds the header object for the ``proc-gimg`` files."""

    # Largely modified from update_proc_headers
    # returns a list of headers to be updated in a separate
    # process (guider_state can't be offloaded to ProcessPool)
    # but a list can.
    guide_loop = guider_state.guide_loop

    enabled_axes = guider_state.enabled_axes
    enabled_ra = "ra" in enabled_axes
    enabled_dec = "dec" in enabled_axes
    enabled_rot = "rot" in enabled_axes
    enabled_focus = "focus" in enabled_axes

    cra, cdec, crot, cscl, cfoc = data.correction_applied

    ra_pid_k = guide_loop["ra"]["pid"]["k"]
    ra_pid_td = guide_loop["ra"]["pid"].get("td", 0.0)
    ra_pid_ti = guide_loop["ra"]["pid"].get("ti", 0.0)

    dec_pid_k = guide_loop["dec"]["pid"]["k"]
    dec_pid_td = guide_loop["dec"]["pid"].get("td", 0.0)
    dec_pid_ti = guide_loop["dec"]["pid"].get("ti", 0.0)

    rot_pid_k = guide_loop["rot"]["pid"]["k"]
    rot_pid_td = guide_loop["rot"]["pid"].get("td", 0.0)
    rot_pid_ti = guide_loop["rot"]["pid"].get("ti", 0.0)

    focus_pid_k = guide_loop["focus"]["pid"]["k"]
    focus_pid_td = guide_loop["focus"]["pid"].get("td", 0.0)
    focus_pid_ti = guide_loop["focus"]["pid"].get("ti", 0.0)

    if "scale" in guide_loop:
        scale_pid_k = guide_loop["scale"]["pid"]["k"]
        scale_pid_td = guide_loop["scale"]["pid"].get("td", 0.0)
        scale_pid_ti = guide_loop["scale"]["pid"].get("ti", 0.0)
    else:
        scale_pid_k = 0.0
        scale_pid_td = 0.0
        scale_pid_ti = 0.0

    rms = data.rms
    delta_ra = data.delta_ra
    delta_dec = data.delta_dec
    delta_rot = data.delta_rot
    delta_scale = data.delta_scale
    delta_focus = data.delta_focus

    cra, cdec, crot, cscl, cfoc = data.correction_applied

    # generate header list
    header = []

    header.append(("SOLVED", guide_data.solved, "field was solved"))
    header.append(
        (
            "SOLVMODE",
            guide_data.solve_method,
            "Method used to solve the field",
        )
    )
    header.append(
        (
            "SOLVTIME",
            guide_data.solve_time,
            "Time to solve the field or fail",
        )
    )

    offsets = guider_state.offset
    header.append(("OFFRA", offsets[0], "Relative offset in RA [arcsec]"))
    header.append(("OFFDEC", offsets[1], "Relative offset in Dec [arcsec]"))
    header.append(("OFFPA", offsets[2], "Relative offset in PA [arcsec]"))

    default_offset = config.get("default_offset", (0.0, 0.0, 0.0))
    aoffset = (
        default_offset[0] + offsets[0],
        default_offset[1] + offsets[1],
        default_offset[2] + offsets[2],
    )
    header.append(("AOFFRA", aoffset[0], "Absolute offset in RA [arcsec]"))
    header.append(("AOFFDEC", aoffset[1], "Absolute offset in Dec [arcsec]"))
    header.append(("AOFFPA", aoffset[2], "Absolute offset in PA [arcsec]"))

    header.append(
        (
            "EXTMETH",
            guide_data.e_data.algorithm,
            "Algorithm for star finding",
        )
    )

    header.append(("RAK", ra_pid_k, "PID K term for RA"))
    header.append(("RATD", ra_pid_td, "PID Td term for RA"))
    header.append(("RATI", ra_pid_ti, "PID Ti term for RA"))

    header.append(("DECK", dec_pid_k, "PID K term for Dec"))
    header.append(("DECTD", dec_pid_td, "PID Td term for Dec"))
    header.append(("DECTI", dec_pid_ti, "PID Ti term for Dec"))

    header.append(("ROTK", rot_pid_k, "PID K term for Rot."))
    header.append(("ROTTD", rot_pid_td, "PID Td term for Rot."))
    header.append(("ROTTI", rot_pid_ti, "PID Ti term for Rot."))

    header.append(("SCLK", scale_pid_k, "PID K term for Scale"))
    header.append(("SCLTD", scale_pid_td, "PID Td term for Scale"))
    header.append(("SCLTI", scale_pid_ti, "PID Ti term for Scale"))

    header.append(("FOCUSK", focus_pid_k, "PID K term for Focus"))
    header.append(("FOCUSTD", focus_pid_td, "PID Td term for Focus"))
    header.append(("FOCUSTI", focus_pid_ti, "PID Ti term for Focus"))

    header.append(("FWHM", guide_data.e_data.fwhm_median, "Mesured FWHM [arcsec]"))
    header.append(("FWHMFIT", data.fwhm_fit, "Fitted FWHM [arcsec]"))

    header.append(("RMS", rms, "Guide RMS [arcsec]"))
    header.append(("FITMODE", data.fit_mode, "Fit mode (full or RA/Dec)"))

    header.append(("E_RA", enabled_ra, "RA corrections enabled?"))
    header.append(("E_DEC", enabled_dec, "Dec corrections enabled?"))
    header.append(("E_ROT", enabled_rot, "Rotator corrections enabled?"))
    header.append(("E_FOCUS", enabled_focus, "Focus corrections enabled?"))
    header.append(("E_SCL", False, "Scale corrections enabled?"))

    header.append(("DELTARA", delta_ra, "RA measured delta [arcsec]"))
    header.append(("DELTADEC", delta_dec, "Dec measured delta [arcsec]"))
    header.append(("DELTAROT", delta_rot, "Rotator measured delta [arcsec]"))
    header.append(("DELTASCL", delta_scale, "Scale measured factor"))
    header.append(("DELTAFOC", delta_focus, "Focus delta [microns]"))

    header.append(("CORR_RA", cra, "RA applied correction [arcsec]"))
    header.append(("CORR_DEC", cdec, "Dec applied correction [arcsec]"))
    header.append(("CORR_ROT", crot, "Rotator applied correction [arcsec]"))
    header.append(("CORR_SCL", cscl, "Scale applied correction"))
    header.append(("CORR_FOC", cfoc, "Focus applied correction [microns]"))

    return header


def write_proc_image(
    guide_data: GuideData,
    overwrite: bool = False,
    bossExposing: bool = False,
    bossExpNum: int = -999,
    header_updates: list | None = None,
    gaia_matches: pandas.DataFrame | None = None,
):
    """Writes the processed image to disk."""

    niceness = os.nice(0)
    dnice = 5 - niceness
    if dnice != 0:
        niceness = os.nice(dnice)

    # Mostly coppied from write_proc_image method on guider
    ext_data = guide_data.extraction_data

    proc_hdu = fits.open(str(guide_data.path)).copy()
    header = proc_hdu[1].header

    calibs = config["calib"][guide_data.camera]

    # apply calibs again for writing the processed image data (done in extraction
    # phase too, but data not saved)
    bias_path = calibs["bias"]
    dark_path = calibs["dark"]
    flat_path = calibs["flat"]

    # flat/bias/dark adjust the raw image
    _calib_data = apply_calibs(
        data=proc_hdu[1].data,
        bias_path=bias_path,
        dark_path=dark_path,
        flat_path=flat_path,
        gain=proc_hdu[1].header["GAIN"],
        exptime=proc_hdu[1].header["EXPTIMEN"],
    )
    proc_hdu[1].data = _calib_data
    header["BKG_ADU"] = (
        numpy.median(_calib_data),
        "background (median) counts in calibrated frame",
    )

    header["BIASFILE"] = bias_path
    header["FLATFILE"] = flat_path
    header["DARKFILE"] = dark_path

    rec = Table.from_pandas(guide_data.extraction_data.regions).as_array()
    proc_hdu.append(fits.BinTableHDU(rec, name="CENTROIDS"))

    gfa_coords = calibration.gfaCoords.reset_index()
    gfa_coords_rec = Table.from_pandas(gfa_coords).as_array()
    proc_hdu.append(fits.BinTableHDU(gfa_coords_rec, name="GFACOORD"))

    if guide_data.wcs is not None:
        header.update(guide_data.wcs.to_header())

    proc_path: pathlib.Path

    proc_path = ext_data.path.parent / ("proc-" + ext_data.path.name)

    if header_updates is not None:
        for kw, val, comment in header_updates:
            header[kw] = (val, comment)
    if gaia_matches is not None:
        rec = Table.from_pandas(gaia_matches).as_array()
        proc_hdu.append(fits.BinTableHDU(rec, name="GAIAMATCH"))

    header["SPEXPNG"] = (bossExposing, "BOSS spectrograph exposing")
    header["SPIMGNO"] = (bossExpNum, "BOSS spectrograph current exposure number")

    proc_hdu.writeto(proc_path, overwrite=overwrite, output_verify="silentfix")
    proc_hdu.close()

    # probably not necessary to add this to the guide_data anymore
    guide_data.proc_image = proc_path
