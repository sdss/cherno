#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-07
# @Filename: guider.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib
import re
import time
import warnings
from dataclasses import dataclass, field
from functools import partial

from typing import TYPE_CHECKING, Any, Coroutine

import numpy
import pandas
from astropy.io import fits
from astropy.stats.sigma_clipping import SigmaClip
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from simple_pid.PID import PID

from clu.command import FakeCommand
from coordio import defaults
from coordio.astrometry import AstrometryNet
from coordio.guide import (
    GuiderFit,
    GuiderFitter,
    cross_match,
    gfa_to_radec,
    radec_to_gfa,
)

from cherno import config, log
from cherno.exceptions import ChernoError
from cherno.extraction import Extraction, ExtractionData, PathLike
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

        # To cache Gaia sources.
        self._gaia_sources: dict = {}

        # Clear RMS history.
        if self.command.actor:
            self.command.actor.state.rms_history.clear()

        self._database_lock = asyncio.Lock()

    def set_command(self, command: ChernoCommandType):
        """Sets the command."""

        self.command = command

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

        self.command.info("Extracting sources.")

        ext_data = await asyncio.gather(
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

        guide_data: list[GuideData]

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
        else:
            guide_data = [GuideData(ed.camera, ed) for ed in ext_data]

        # Use Gaia cross-match for the cameras that did not solve with astrometry.net.
        not_solved = [ad for ad in guide_data if ad.solved is False]
        if use_gaia and len(not_solved) > 0:
            self.command.info("Running Gaia cross-match.")
            res = await asyncio.gather(
                *[
                    self._gaia_cross_match_one(
                        ad,
                        gaia_phot_g_mean_mag_max=gaia_phot_g_mean_mag_max,
                        gaia_cross_correlation_blur=gaia_cross_correlation_blur,
                    )
                    for ad in not_solved
                ],
                return_exceptions=True,
            )
            for ii, rr in enumerate(res):
                if isinstance(rr, Exception):
                    cam = not_solved[ii].camera
                    self.command.warning(f"{cam}: Gaia cross-match failed: {str(rr)}")

        # Output all the camera_solution keywords at once.
        for ad in guide_data:
            self.output_camera_solution(ad)

        self.command.debug("Saving proc- file.")
        if write_proc:
            await asyncio.gather(
                *[self.write_proc_image(d, overwrite=overwrite) for d in guide_data]
            )

        ast_solution = await self.fit(
            list(guide_data),
            offset=offset,
            scale_rms=scale_rms,
            only_radec=only_radec,
            auto_radec_min=auto_radec_min,
            fit_all_detections=fit_all_detections,
            fit_focus=fit_focus,
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

        if self.command.actor:
            update_proc_headers(ast_solution, self.command.actor.state)

        return ast_solution

    async def fit(
        self,
        data: list[GuideData],
        offset: list[float] | None = None,
        scale_rms: bool = False,
        fit_focus: bool = True,
        only_radec: bool = False,
        auto_radec_min: int = 2,
        fit_all_detections: bool = True,
    ):
        """Calculate the astrometric solution."""

        ast_solution = AstrometricSolution(False, data)

        solved = sorted([d for d in data if d.solved is True], key=lambda x: x.camera)
        weights = [s.extraction_data.nvalid for s in solved]

        if len(solved) == 0:
            self.command.error(acquisition_valid=False, did_correct=False)
            return ast_solution

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
            return ast_solution

        if offset is None:
            if self.command.actor:
                offset = list(self.command.actor.state.offset)
            else:
                offset = list(config.get("offset", [0.0, 0.0, 0.0]))

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

        default_offset = config.get("default_offset", (0.0, 0.0, 0.0))
        full_offset = numpy.array(offset) + numpy.array(default_offset)

        self.command.debug(default_offset=default_offset)
        if any(offset):
            self.command.warning(offset=offset)
        else:
            self.command.debug(offset=offset)

        if only_radec is True:
            self.command.warning(
                "Only fitting RA/Dec. The rotation and scale offsets "
                "are informational-only and not corrected."
            )
        elif auto_radec_min >= 0 and len(solved) <= auto_radec_min:
            only_radec = True
            self.command.warning(
                f"Only {len(solved)} cameras solved. Only fitting RA/Dec. "
                "The rotation and scale offsets are informational-only "
                "and not corrected."
            )

        fit_cameras = [d.camera_id for d in solved]
        fit_rms_sigma = config["guider"].get("fit_rms_sigma", 3)
        guider_fit = False
        while True:
            tmp_guider_fit = self.fitter.fit(
                field_ra,
                field_dec,
                field_pa,
                offset=full_offset,
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

        plate_scale = defaults.PLATE_SCALE[self.observatory]
        mm_to_arcsec = 1 / plate_scale * 3600

        exp_no = solved[0].exposure_no  # Should be the same for all.

        if guider_fit:
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
            fit_rms_camera = [numpy.round(fit_rms.loc[0].rms * mm_to_arcsec, 4)]
            for cid in range(1, 7):
                if cid in fit_rms.index:
                    this_fit_rms = numpy.round(fit_rms.loc[cid].rms * mm_to_arcsec, 4)
                    fit_rms_camera.append(this_fit_rms)
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
            try:
                fwhm_fit, x_min, a, b, c, r2 = focus_fit(
                    [d.e_data for d in data],
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

        self.command.info(
            astrometry_fit=[
                exp_no,
                len(solved),
                -999.0,
                -999.0,
                numpy.round(fwhm, 2),
                -999,
                numpy.round(camera_rotation, 3),
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
            and fwhm < 2.5
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

        return ast_solution

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

        if wcs and guide_data.solved:
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
                "SELECT * FROM catalogdb.gaia_dr2_source_g19 "
                "WHERE q3c_radial_query(ra, dec, "
                f"{ccd_centre[0]}, {ccd_centre[1]}, {gaia_search_radius}) AND "
                f"phot_g_mean_mag < {g_mag}",
                config["guider"]["gaia_connection_string"],
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

    async def write_proc_image(
        self,
        guide_data: GuideData,
        outpath: PathLike | None = None,
        overwrite: bool = False,
    ):
        """Writes the proc-gimg image."""

        ext_data = guide_data.extraction_data

        proc_hdu = fits.open(str(guide_data.path)).copy()

        rec = Table.from_pandas(guide_data.extraction_data.regions).as_array()
        proc_hdu.append(fits.BinTableHDU(rec, name="CENTROIDS"))

        proc_hdu[1].header["SOLVED"] = guide_data.solved
        proc_hdu[1].header["SOLVMODE"] = (
            guide_data.solve_method,
            "Method used to solve the field",
        )
        proc_hdu[1].header["SOLVTIME"] = (
            guide_data.solve_time,
            "Time to solve the field or fail",
        )

        if self.command.actor is not None:
            offsets = self.command.actor.state.offset
        else:
            offsets = [-999.0] * 3

        proc_hdu[1].header["OFFRA"] = (offsets[0], "Relative offset in RA [arcsec]")
        proc_hdu[1].header["OFFDEC"] = (offsets[1], "Relative offset in Dec [arcsec]")
        proc_hdu[1].header["OFFPA"] = (offsets[2], "Relative offset in PA [arcsec]")

        default_offset = config.get("default_offset", (0.0, 0.0, 0.0))
        aoffset = (
            default_offset[0] + offsets[0],
            default_offset[1] + offsets[1],
            default_offset[2] + offsets[2],
        )
        proc_hdu[1].header["AOFFRA"] = (aoffset[0], "Absolute offset in RA [arcsec]")
        proc_hdu[1].header["AOFFDEC"] = (aoffset[1], "Absolute offset in Dec [arcsec]")
        proc_hdu[1].header["AOFFPA"] = (aoffset[2], "Absolute offset in PA [arcsec]")

        proc_hdu[1].header.update(guide_data.wcs.to_header())

        proc_path: pathlib.Path
        if outpath is not None:
            proc_path = pathlib.Path(outpath)
            if proc_path.is_dir():
                proc_path = proc_path / ("proc-" + ext_data.path.name)
        else:
            proc_path = ext_data.path.parent / ("proc-" + ext_data.path.name)

        loop = asyncio.get_running_loop()
        func = partial(
            proc_hdu.writeto,
            proc_path,
            overwrite=overwrite,
            output_verify="silentfix",
        )
        await loop.run_in_executor(None, func)

        guide_data.proc_image = proc_path


def update_proc_headers(data: AstrometricSolution, guider_state: ChernoState):
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

    # Update headers of proc images with deltas.
    for a_data in data.guide_data:
        if a_data.proc_image is not None:
            hdus = fits.open(str(a_data.proc_image), mode="update")
            header = hdus[1].header

            header["EXTMETH"] = (a_data.e_data.algorithm, "Algorithm for star finding")

            header["RAK"] = (ra_pid_k, "PID K term for RA")
            header["RATD"] = (ra_pid_td, "PID Td term for RA")
            header["RATI"] = (ra_pid_ti, "PID Ti term for RA")

            header["DECK"] = (dec_pid_k, "PID K term for Dec")
            header["DECTD"] = (dec_pid_td, "PID Td term for Dec")
            header["DECTI"] = (dec_pid_ti, "PID Ti term for Dec")

            header["ROTK"] = (rot_pid_k, "PID K term for Rot.")
            header["ROTTD"] = (rot_pid_td, "PID Td term for Rot.")
            header["ROTTI"] = (rot_pid_ti, "PID Ti term for Rot.")

            header["SCLK"] = (scale_pid_k, "PID K term for Scale")
            header["SCLTD"] = (scale_pid_td, "PID Td term for Scale")
            header["SCLTI"] = (scale_pid_ti, "PID Ti term for Scale")

            header["FOCUSK"] = (focus_pid_k, "PID K term for Focus")
            header["FOCUSTD"] = (focus_pid_td, "PID Td term for Focus")
            header["FOCUSTI"] = (focus_pid_ti, "PID Ti term for Focus")

            header["FWHM"] = (a_data.e_data.fwhm_median, "Mesured FWHM [arcsec]")
            header["FWHMFIT"] = (data.fwhm_fit, "Fitted FWHM [arcsec]")

            header["RMS"] = (rms, "Guide RMS [arcsec]")
            header["FITMODE"] = (data.fit_mode, "Fit mode (full or RA/Dec)")

            header["E_RA"] = (enabled_ra, "RA corrections enabled?")
            header["E_DEC"] = (enabled_dec, "Dec corrections enabled?")
            header["E_ROT"] = (enabled_rot, "Rotator corrections enabled?")
            header["E_FOCUS"] = (enabled_focus, "Focus corrections enabled?")
            header["E_SCL"] = (False, "Scale corrections enabled?")

            header["DELTARA"] = (delta_ra, "RA measured delta [arcsec]")
            header["DELTADEC"] = (delta_dec, "Dec measured delta [arcsec]")
            header["DELTAROT"] = (delta_rot, "Rotator measured delta [arcsec]")
            header["DELTASCL"] = (delta_scale, "Scale measured factor")
            header["DELTAFOC"] = (delta_focus, "Focus delta [microns]")

            header["CORR_RA"] = (cra, "RA applied correction [arcsec]")
            header["CORR_DEC"] = (cdec, "Dec applied correction [arcsec]")
            header["CORR_ROT"] = (crot, "Rotator applied correction [arcsec]")
            header["CORR_SCL"] = (cscl, "Scale applied correction")
            header["CORR_FOC"] = (cfoc, "Focus applied correction [microns]")

            hdus.close()
