#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-07
# @Filename: acquisition.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from functools import partial

from typing import TYPE_CHECKING, Any, Coroutine

import numpy
import pandas
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from simple_pid.PID import PID

from clu.command import FakeCommand
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


__all__ = ["Acquisition"]


@dataclass
class AcquisitionData:
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

    @property
    def e_data(self):
        """Shortcut for ``extraction_data``."""

        return self.extraction_data


@dataclass
class AstrometricSolution:
    """Astrometric solution data."""

    valid_solution: bool
    acquisition_data: list[AcquisitionData]
    guider_fit: GuiderFit | None = None
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


class Acquisition:
    """Runs the steps for field acquisition."""

    def __init__(
        self,
        observatory: str,
        extractor: Extraction | None = None,
        astrometry: AstrometryNet | None = None,
        command: ChernoCommandType | None = None,
        astrometry_params: dict = {},
        extraction_params: dict = {},
    ):

        self.extractor = extractor or Extraction(observatory, **extraction_params)

        if astrometry is not None:
            self.astrometry = astrometry
        else:
            pixel_scale = config["pixel_scale"]
            astrometry_net_config = config["acquisition"]["astrometry_net_config"]
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
                cpulimit=config["acquisition"]["cpulimit"],
                **astrometry_params,
            )

        self.observatory = observatory.upper()
        self.fitter = GuiderFitter(self.observatory)
        self.command = command or FakeCommand(log)
        self.pids = AxesPID(command.actor if command is not None else None)

        # To cache Gaia sources.
        self._gaia_sources: dict = {}

        self._database_lock = asyncio.Lock()

    def set_command(self, command: ChernoCommandType):
        """Sets the command."""

        self.command = command

    async def process(
        self,
        command: ChernoCommandType | None,
        images: PathLike | list[PathLike],
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
    ):
        """Performs extraction and astrometry."""

        if command is not None:
            self.set_command(command)

        if not isinstance(images, (list, tuple)):
            images = [images]

        self.command.info("Extracting sources.")

        ext_data = await asyncio.gather(
            *[
                run_in_executor(self.extractor.process, im, executor="process")
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

        acq_data: list[AcquisitionData]

        use_astrometry_net = (
            use_astrometry_net
            if use_astrometry_net is not None
            else config["acquisition"].get("use_astrometry_net", True)
        )

        if use_astrometry_net:
            self.command.info("Running astrometry.net.")
            acq_data = await asyncio.gather(
                *[self._astrometry_one(d) for d in ext_data]
            )
        else:
            acq_data = [AcquisitionData(ed.camera, ed) for ed in ext_data]

        # Use Gaia cross-match for the cameras that did not solve with astrometry.net.
        not_solved = [ad for ad in acq_data if ad.solved is False]
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
        for ad in acq_data:
            self.output_camera_solution(ad)

        self.command.debug("Saving proc- file.")
        if write_proc:
            await asyncio.gather(
                *[self.write_proc_image(d, overwrite=overwrite) for d in acq_data]
            )

        ast_solution = await self.fit(
            list(acq_data),
            offset=offset,
            scale_rms=scale_rms,
            only_radec=only_radec,
            auto_radec_min=auto_radec_min,
            fit_all_detections=fit_all_detections,
            fit_focus=fit_focus,
        )

        if correct and ast_solution.valid_solution is True:
            await self.correct(
                ast_solution,
                full=full_correction,
                wait_for_correction=wait_for_correction,
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
        data: list[AcquisitionData],
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
        fwhm = numpy.average(fwhm)

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

        guider_fit = self.fitter.fit(
            field_ra,
            field_dec,
            field_pa,
            offset=full_offset,
            scale_rms=scale_rms,
            only_radec=only_radec,
        )

        exp_no = solved[0].exposure_no  # Should be the same for all.

        if guider_fit is False:
            rms = delta_ra = delta_dec = delta_rot = delta_scale = -999.0
            ast_solution.guider_fit = None
        else:
            delta_ra = guider_fit.delta_ra
            delta_dec = guider_fit.delta_dec
            delta_rot = guider_fit.delta_rot
            delta_scale = guider_fit.delta_scale

            xrms = guider_fit.xrms
            yrms = guider_fit.yrms
            rms = guider_fit.rms

            ast_solution.guider_fit = guider_fit

            self.command.info(guide_rms=[exp_no, xrms, yrms, rms])

        ast_solution.valid_solution = True

        ast_solution.delta_ra = float(delta_ra)
        ast_solution.delta_dec = float(delta_dec)
        ast_solution.delta_rot = float(delta_rot)
        ast_solution.delta_scale = float(delta_scale)
        ast_solution.rms = float(rms)

        if guider_fit and guider_fit.only_radec:
            ast_solution.fit_mode = "radec"

        if fit_focus:
            try:
                fwhm_fit, x_min, a, b, c, r2 = focus_fit(
                    [d.e_data for d in data],
                    plot=config["acquisition"]["plot_focus"],
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
        for d in ast_solution.acquisition_data:
            if d.extraction_data.fwhm_median > 0 and d.extraction_data.nvalid > 0:
                focus_data += [
                    int(d.camera[-1]),
                    d.extraction_data.fwhm_median,
                    d.extraction_data.focus_offset,
                ]

        self.command.debug(focus_data=focus_data)

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
            and guider_fit
            and not guider_fit.only_radec
        ):
            # If we measured the scale, add it to the actor state. This is later
            # used to compute the average scale over a period. We also add the time
            # because we'll want to reject measurements that are too old.
            self.command.actor.state.scale_history.append((time.time(), delta_scale))

        return ast_solution

    async def correct(
        self,
        data: AstrometricSolution,
        full: bool = False,
        wait_for_correction: bool = True,
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
            await self._correct_apo(data, full=full)
        else:
            await self._correct_lco(
                data,
                full=full,
                wait_for_correction=wait_for_correction,
            )

    async def _correct_apo(self, data: AstrometricSolution, full: bool = False):

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
            data.focus_r2 > config["acquisition"]["focus_r2_threshold"]
            and data.focus_coeff[0] > 0
        ):
            do_focus = True
            coro = apply_focus_correction(self.command, self.pids, data.delta_focus)
            correct_tasks.append(coro)
        else:
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
    ):

        do_focus: bool = False

        enabled_axes = self.command.actor.state.enabled_axes

        # Ignore focus correction when the r2 correlation is bad or when we got
        # an inverted parabola.
        if (
            data.focus_r2 > config["acquisition"]["focus_r2_threshold"]
            and data.focus_coeff[0] > 0
            and "focus" in enabled_axes
        ):
            do_focus = True
        else:
            self.command.warning("Focus fit poorly constrained. Not correcting focus.")

        self.command.actor.state.set_status(GuiderStatus.CORRECTING, mode="add")

        applied_corrections: Any = await apply_correction_lco(
            self.command,
            self.pids,
            delta_radec=(data.delta_ra, data.delta_dec),
            delta_rot=data.delta_rot if data.fit_mode == "full" else None,
            delta_focus=data.delta_focus if do_focus else None,
            full=full,
            wait_for_correction=wait_for_correction,
        )

        self.command.actor.state.set_status(GuiderStatus.CORRECTING, mode="remove")

        data.correction_applied = applied_corrections

        self.command.info(
            acquisition_valid=True,
            did_correct=any(data.correction_applied),
            correction_applied=data.correction_applied,
        )

    async def _astrometry_one(self, ext_data: ExtractionData):

        if config["acquisition"]["astrometry_net_use_all_regions"]:
            regions = ext_data.regions.copy()
        else:
            regions = ext_data.regions.loc[ext_data.regions.valid == 1].copy()

        if self.astrometry._options.get("dir", None) is None:
            astrometry_dir = pathlib.Path(config["acquisition"]["astrometry_dir"])
            if astrometry_dir.is_absolute():
                pass
            else:
                astrometry_dir = ext_data.path.parent / astrometry_dir

            astrometry_dir.mkdir(exist_ok=True, parents=True)
        else:
            astrometry_dir = pathlib.Path(self.astrometry._options["dir"])

        outfile_root = astrometry_dir / ext_data.path.stem

        # We use all detections, even invalid ones here.
        if ext_data.algorithm == "marginal":
            xyls_df = regions.loc[:, ["x", "y", "flux"]].copy()
            xyls_df = xyls_df.rename(columns={"x_fit": "x", "y_fit": "y"})
        elif ext_data.algorithm == "daophot":
            xyls_df = regions.loc[:, ["x_0", "y_0", "flux_0"]].copy()
            xyls_df = xyls_df.rename(columns={"x_0": "x", "y_0": "y", "flux_0": "flux"})
        else:
            xyls_df = regions.loc[:, ["x", "y", "flux"]]

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

        acq_data = AcquisitionData(ext_data.camera, ext_data, solve_time=proc.elapsed)

        if wcs_output.exists():
            acq_data.solved = True
            wcs = WCS(open(wcs_output).read())
            acq_data.wcs = wcs
            acq_data.solve_method = "astrometry.net"

        return acq_data

    def output_camera_solution(self, acq_data: AcquisitionData):
        """Calculates and outputs the camera_solution keyword."""

        wcs = acq_data.wcs

        if wcs and acq_data.solved:

            racen, deccen = wcs.pixel_to_world_values([[1024, 1024]])[0]
            acq_data.camera_racen = float(numpy.round(racen, 6))
            acq_data.camera_deccen = float(numpy.round(deccen, 6))

            # TODO: consider parallactic angle here.
            cd: numpy.ndarray = wcs.wcs.cd
            rot_rad = numpy.arctan2(
                numpy.array([cd[0, 1], cd[0, 0]]),
                numpy.array([cd[1, 1], cd[1, 0]]),
            )

            # Rotation is from N to E to the x and y axes of the GFA.
            yrot, xrot = numpy.rad2deg(rot_rad)
            acq_data.xrot = float(numpy.round(xrot % 360.0, 3))
            acq_data.yrot = float(numpy.round(yrot % 360.0, 3))

            # Calculate field rotation.
            cameras = config["cameras"]
            camera_rot = cameras["rotation"][acq_data.camera]
            rotation = numpy.array([xrot - camera_rot - 90, yrot - camera_rot]) % 360
            rotation[rotation > 180.0] -= 360.0
            rotation = numpy.mean(rotation)
            acq_data.rotation = float(numpy.round(rotation, 3))

        self.command.info(
            camera_solution=[
                acq_data.camera,
                acq_data.exposure_no,
                acq_data.solved,
                acq_data.camera_racen,
                acq_data.camera_deccen,
                acq_data.xrot,
                acq_data.yrot,
                acq_data.rotation,
                acq_data.solve_method,
            ]
        )

    async def _gaia_cross_match_one(
        self,
        acquisition_data: AcquisitionData,
        gaia_phot_g_mean_mag_max: float | None = None,
        gaia_cross_correlation_blur: float | None = None,
    ):
        """Solves a field cross-matching to Gaia."""

        cam = acquisition_data.camera

        regions = acquisition_data.extraction_data.regions.copy()
        if acquisition_data.extraction_data.algorithm == "marginal":
            xyls_df = regions.loc[:, ["x", "y", "flux"]].copy()
            xyls_df = xyls_df.rename(columns={"x_fit": "x", "y_fit": "y"})
        elif acquisition_data.extraction_data.algorithm == "daophot":
            xyls_df = regions.loc[:, ["x_0", "y_0", "flux_0"]].copy()
            xyls_df = xyls_df.rename(columns={"x_0": "x", "y_0": "y", "flux_0": "flux"})
        else:
            xyls_df = regions.loc[:, ["x", "y", "flux"]]

        if len(xyls_df) < 4:
            self.command.warning(f"{cam}: too few sources. Cannot cross-match to Gaia.")
            return

        acq_config = config["acquisition"]

        ra = acquisition_data.extraction_data.field_ra
        dec = acquisition_data.extraction_data.field_dec
        pa = acquisition_data.extraction_data.field_pa

        if self.command.actor is not None:
            offsets = self.command.actor.state.offset
        else:
            offsets = [0.0] * 3

        default_offset = config.get("default_offset", (0.0, 0.0, 0.0))

        offra = default_offset[0] + offsets[0]
        offdec = default_offset[1] + offsets[1]
        offpa = default_offset[2] + offsets[2]

        cam_id = int(acquisition_data.camera[-1])
        obstime_jd = acquisition_data.extraction_data.obstime.jd

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

        fid = acquisition_data.extraction_data.field_id
        if fid != -999 and (fid, cam_id) in self._gaia_sources:
            gaia_stars = self._gaia_sources[(fid, cam_id)]

        else:

            gaia_stars = pandas.read_sql(
                "SELECT * FROM catalogdb.gaia_dr2_source_g19 "
                "WHERE q3c_radial_query(ra, dec, "
                f"{ccd_centre[0]}, {ccd_centre[1]}, {gaia_search_radius}) AND "
                f"phot_g_mean_mag < {g_mag}",
                config["acquisition"]["gaia_connection_string"],
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
            xyls_df.values[:, :2],
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
            acquisition_data.solved = True
            acquisition_data.solve_method = "gaia"
            acquisition_data.wcs = wcs

    async def write_proc_image(
        self,
        acq_data: AcquisitionData,
        outpath: PathLike = None,
        overwrite: bool = False,
    ):
        """Writes the proc-gimg image."""

        ext_data = acq_data.extraction_data

        proc_hdu = fits.open(str(acq_data.path)).copy()

        rec = Table.from_pandas(acq_data.extraction_data.regions).as_array()
        proc_hdu.append(fits.BinTableHDU(rec, name="CENTROIDS"))

        proc_hdu[1].header["SOLVED"] = acq_data.solved
        proc_hdu[1].header["SOLVMODE"] = (
            acq_data.solve_method,
            "Method used to solve the field",
        )
        proc_hdu[1].header["SOLVTIME"] = (
            acq_data.solve_time,
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

        proc_hdu[1].header.update(acq_data.wcs.to_header())

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

        acq_data.proc_image = proc_path


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
    for a_data in data.acquisition_data:
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
