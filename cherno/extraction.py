#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-07
# @Filename: extraction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, field

from typing import Any, TypeVar, cast

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
import sep
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D, Trapezoid1D
from astropy.stats import SigmaClip, gaussian_sigma_to_fwhm
from astropy.table import Table
from astropy.time import Time, TimeDelta
from matplotlib.patches import Ellipse
from photutils.background import MedianBackground, StdBackgroundRMS
from photutils.detection import DAOStarFinder
from photutils.psf import BasicPSFPhotometry, DAOGroup, IntegratedGaussianPRF

from cherno import config


PathLike = TypeVar("PathLike", pathlib.Path, str)
seaborn.set_theme(style="white")


__all__ = ["ExtractionData", "Extraction"]


@dataclass
class ExtractionData:
    """Data from extraction."""

    image: str
    path: pathlib.Path
    camera: str
    exposure_no: int
    mjd: int
    obstime: Time
    observatory: str
    field_id: int
    field_ra: float
    field_dec: float
    field_pa: float
    algorithm: str
    regions: pandas.DataFrame = field(default_factory=pandas.DataFrame)
    nregions: int = 0
    nvalid: int = 0
    fwhm_median: float = -999.0
    focus_offset: float = 0.0


class Extraction:
    """Extract centroids and PSF information from an image."""

    __VALID_STAR_FINDER = ["daophot", "sextractor", "marginal"]

    def __init__(
        self,
        observatory: str,
        star_finder: str | None = None,
        pixel_scale: float | None = None,
        daophot_params={},
        marginal_params={},
        **params,
    ):

        self.observatory = observatory.upper()
        self.pixel_scale = pixel_scale or config["pixel_scale"]

        self.params = deepcopy(config["extraction"])
        self.params.update(params)
        self.params["daophot"].update(daophot_params)
        self.params["marginal"].update(marginal_params)

        self.output_dir = pathlib.Path(self.params["output_dir"])

        self.star_finder = star_finder or self.params["star_finder"]
        if self.star_finder not in self.__VALID_STAR_FINDER:
            raise ValueError(
                f"Invalid star finder. Valid values are {self.__VALID_STAR_FINDER}."
            )

    def process(self, image: PathLike) -> ExtractionData:
        """Process an image."""

        hdu = fits.open(image)
        data = hdu[1].data
        header = hdu[1].header

        camera = header["CAMNAME"][0:-1]  # Remove the n/s at the end of the camera name
        observatory = header["OBSERVAT"]

        obstime = Time(header["DATE-OBS"], format="iso", scale="tai")
        obstime += TimeDelta(header["EXPTIMEN"] / 2.0, format="sec")

        path = pathlib.Path(image)
        path = path.absolute()

        mjd = int(path.parts[-2])

        if match := re.match(r".*gimg\-gfa(\d)[ns]\-(\d+)\.fits", path.parts[-1]):
            cam_no = int(match.group(1))
            exp_no = int(match.group(2))
        else:
            cam_no = 0
            exp_no = 0

        if self.star_finder == "sextractor":
            regions = self._process_sextractor(data, path)[0]
        elif self.star_finder == "daophot":
            regions = self._process_daophot(data, path)
        elif self.star_finder == "marginal":
            regions = self._process_marginal(data, path)
        else:
            regions = pandas.DataFrame()

        regions["mjd"] = mjd
        regions["exposure"] = exp_no
        regions["camera"] = cam_no
        regions["fwhm_valid"] = 1

        if len(regions) > 0 and self.params["rejection_method"] is not None:
            self.reject(regions)

        valid = regions.loc[regions.fwhm_valid == 1]

        if len(valid) > 0:
            perc_50 = numpy.percentile(valid.fwhm, 50)
            fwhm_median = valid.loc[valid.fwhm < perc_50].fwhm.median()
            fwhm_median_round = float(numpy.round(fwhm_median, 3))

            # Prevent NaNs here since this is output to the headers.
            if numpy.isnan(fwhm_median_round):
                fwhm_median_round = -999.0
        else:
            fwhm_median_round = -999.0

        extraction_data = ExtractionData(
            str(image),
            path,
            camera,
            exp_no,
            int(obstime.mjd),
            obstime,
            observatory,
            field_id=header.get("FIELDID", -999),
            field_ra=header["RAFIELD"],
            field_dec=header["DECFIELD"],
            field_pa=header["FIELDPA"],
            algorithm=self.star_finder,
            regions=regions,
            nregions=len(regions),
            nvalid=sum(regions.valid == 1),
            fwhm_median=fwhm_median_round,
            focus_offset=config["cameras"]["focus_offset"][camera],
        )

        output_file = self._get_output_path(path).with_suffix(".csv")
        output_file.unlink(missing_ok=True)

        regions.to_csv(str(output_file))

        return extraction_data

    def multiprocess(
        self,
        images: list[PathLike],
        n_cpus: int | None = None,
    ) -> list[ExtractionData]:

        n_cpus = n_cpus or multiprocessing.cpu_count()

        with multiprocessing.Pool(n_cpus) as pool:
            results = pool.map(self.process, images)

        return results

    def _get_output_path(self, path: pathlib.Path):
        """Returns the root path for output files."""

        path_no_suffix = path.with_suffix("")

        dirname = path_no_suffix.parent
        basename = path_no_suffix.parts[-1]

        if self.output_dir.is_absolute():
            output_dir = self.output_dir
        else:
            output_dir = dirname / self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / basename

    def _process_sextractor(
        self,
        data: numpy.ndarray,
        path: pathlib.Path,
        plot: bool | None = None,
    ):
        """Process image data using SExtractor."""

        if plot is None:
            plot = self.params.get("plot", False)

        output_root = self._get_output_path(path)

        back = sep.Background(data.astype("i4"))

        if plot:
            with plt.ioff():
                fig, ax = plt.subplots()
                ax.imshow(back, origin="lower")
                ax.set_title("Background: " + path.parts[-1])
                ax.grid(False)
                fig.savefig(str(output_root) + "-background.pdf")

        regions = sep.extract(
            data - back.back(),
            self.params["background_sigma"],
            err=back.globalrms,
        )

        regions = pandas.DataFrame(regions)
        regions.index.set_names("regions_id", inplace=True)
        regions.loc[:, "valid"] = 0

        min_npix = self.params["min_npix"]
        if len(regions) > 0:
            regions.loc[regions.npix > min_npix, "valid"] = 1

            # Keep only detectons with flux and sort them by highest flux first.
            regions.dropna(subset=["flux"], inplace=True)
            regions.sort_values("flux", inplace=True, ascending=False)

        if self.params["max_stars"]:
            regions = regions.loc[regions.valid == 1, :]
            regions = regions.head(self.params["max_stars"])

        regions = calculate_fwhm_from_ellipse(regions, self.pixel_scale)

        if plot:
            data_back = data - back.back()
            self.plot_regions(
                regions,
                data_back,
                path=path,
                vmin=data_back.mean() - back.globalrms,
                vmax=data_back.mean() + back.globalrms,
                factor=5.0,
                title=path.parts[-1] + " (SExtractor)",
            )

        return regions, back

    def _process_daophot(
        self,
        data: numpy.ndarray,
        path: pathlib.Path,
        plot: bool | None = None,
    ) -> pandas.DataFrame:
        """Process using iteratively subtracted PSF photometry."""

        if plot is None:
            plot = self.params.get("plot", False)

        psf_fwhm = self.params["daophot"]["initial_psf_fwhm"]
        psf_sigma = psf_fwhm / gaussian_sigma_to_fwhm

        background_sigma = self.params["background_sigma"]
        max_stars = self.params["daophot"].get("max_stars", None)

        data = data.astype("float32")

        bkgrms = StdBackgroundRMS()
        std = bkgrms(data)

        if self.params["daophot"].get("use_sep_finder", False):
            sep_regions = self._process_sextractor(data, path, plot=False)[0]
            initial_guesses = Table(
                names=["x_0", "y_0"],
                data=[sep_regions.x.iloc[0:max_stars], sep_regions.y.iloc[0:max_stars]],
            )
            finder = None
        else:
            initial_guesses = None
            finder = DAOStarFinder(
                threshold=background_sigma * std,
                fwhm=psf_fwhm,
                brightest=max_stars,
            )

        daogroup = DAOGroup(3.0 * psf_fwhm)

        mmm_bkg = MedianBackground()

        psf_model = IntegratedGaussianPRF(sigma=psf_sigma)
        psf_model.sigma.fixed = self.params["daophot"].get("fixed_sigma", False)

        fitter = LevMarLSQFitter()

        photometry = BasicPSFPhotometry(
            finder=finder,
            group_maker=daogroup,
            bkg_estimator=mmm_bkg,
            psf_model=psf_model,
            fitter=fitter,
            fitshape=(15, 15),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regions = photometry(image=data, init_guesses=initial_guesses).to_pandas()

        regions["fwhm"] = regions.sigma_fit * gaussian_sigma_to_fwhm * self.pixel_scale
        regions["valid"] = 1

        if plot:
            self.plot_regions(
                regions,
                data,
                path=path,
                vmin=data.mean() - std,
                vmax=data.mean() + std,
                factor=5.0,
                xcen_col="x_0",
                ycen_col="y_0",
                a_col="sigma_fit",
                b_col="sigma_fit",
                theta_col=None,
                title=path.parts[-1] + " (DAOphot)",
            )

        return regions

    def _process_marginal(
        self,
        data: numpy.ndarray,
        path: pathlib.Path,
        plot: bool | None = None,
    ):
        """Extracts regions using the marginal distribution.

        Determines the initial centroids using the SExtractor routine.
        Then a box around each detection is selected and background-subtracted.
        The marginal distributions of the detection (the collapsed sum on each axis)
        are fitted using both a 1D Gaussian and 1D trapezoid, and the FWHM for each
        are calculated. Then the best match for each detection is selected based on
        the option that minimises the residuals.

        """

        marginal_params = self.params["marginal"]

        regions, back = self._process_sextractor(data, path, plot=False)
        regions = regions.loc[regions.valid == 1]
        regions = regions.rename(columns={"fwhm": "fwhm_sextractor"})

        regions["valid"] = 1 * (regions.npix > self.params["min_npix"])

        # Parameters of the Gaussian fit.
        regions["gaussian_fit"] = 1
        regions["x_gaussian"] = numpy.nan
        regions["y_gaussian"] = numpy.nan
        regions["fwhm_gaussian"] = numpy.nan
        regions["residual_gaussian"] = numpy.nan

        # Parameters of the trapezoid fit.
        regions["trapezoid_fit"] = 1
        regions["x_trapezoid"] = numpy.nan
        regions["y_trapezoid"] = numpy.nan
        regions["fwhm_trapezoid"] = numpy.nan
        regions["residual_trapezoid"] = numpy.nan

        # Parameters of the best fit. model_fit=g if the best fit was the Gaussian,
        # t if the trapezoid, and empty if invalid.
        regions["model_fit"] = ""
        regions["x_fit"] = numpy.nan
        regions["y_fit"] = numpy.nan
        regions["fwhm"] = numpy.nan
        regions["residual_fit"] = numpy.nan

        fitter = LevMarLSQFitter()
        fit_box = marginal_params["fit_box"]

        p_scale = self.pixel_scale
        data_back = data - back.back()

        for index, row in regions.iterrows():

            index = cast(Any, index)

            # Ignore detections that we have already marked as invalid in SExtractor.
            if row.valid == 0:
                continue

            cen = numpy.round([row.y, row.x]).astype(int)

            gauss_residual = []
            gauss_centroids = []
            gauss_fwhm = []
            gauss_valid = True

            trap_residual = []
            trap_centroids = []
            trap_fwhm = []
            trap_valid = True

            data_region = data_back[
                cen[0] - fit_box[0] // 2 : cen[0] + fit_box[0] // 2 + 1,
                cen[1] - fit_box[1] // 2 : cen[1] + fit_box[1] // 2 + 1,
            ].astype("f4")

            # Deal with regions near the edges
            if (
                data_region.shape[0] != data_region.shape[1]
                or data_region.shape[0] < fit_box[0]
                or data_region.shape[1] < fit_box[1]
            ):
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                for axis in [0, 1]:
                    mid = fit_box[axis] // 2
                    x_mesh = numpy.arange(fit_box[axis])

                    data_marginal = data_region.sum(axis)
                    data_marginal /= data_marginal.max()

                    gauss_init = Gaussian1D(
                        amplitude=data_marginal.max(),
                        stddev=row.fwhm_sextractor / p_scale / gaussian_sigma_to_fwhm,
                        mean=mid,
                    )
                    g = fitter(gauss_init, x_mesh, data_marginal)
                    if gauss_valid is True:
                        # Reject if the fitting routine doesn't say this is a good fit.
                        gauss_valid = fitter.fit_info["ierr"] <= 4
                    if numpy.abs(g.mean - mid) > 10:
                        gauss_valid = False
                    gauss_centroids.append(cen[axis] + (g.mean - mid) + 0.5)
                    gauss_fwhm.append(g.stddev * gaussian_sigma_to_fwhm * p_scale)
                    gauss_residual.append(numpy.sum((data_marginal - g(x_mesh)) ** 2))

                    trap_init = Trapezoid1D(
                        amplitude=data_marginal.max(),
                        x_0=mid,
                        width=row.fwhm_sextractor / p_scale / gaussian_sigma_to_fwhm,
                        slope=0.1,
                    )
                    t = fitter(trap_init, x_mesh, data_marginal)
                    if trap_valid is True:
                        trap_valid = fitter.fit_info["ierr"] <= 4
                    if numpy.abs(t.x_0 - mid):
                        trap_valid = False
                    trap_centroids.append(cen[axis] + (t.x_0 - mid) + 0.5)
                    trap_fwhm.append((t.amplitude / t.slope + t.width) * p_scale)
                    trap_residual.append(numpy.sum((data_marginal - t(x_mesh)) ** 2))

            #         with plt.ioff():
            #             fig, ax = plt.subplots()
            #             ax.plot(data_marginal, c="k", ls="dotted")
            #             ax.plot(g(x_mesh), "b-")
            #             ax.plot(t(x_mesh), "g-")
            #             ax.axhline(y=0.0, ls="--", c="r")
            #             fig.savefig(
            #                 path.parent
            #                 / "extraction"
            #                 / (path.stem + f"-{index}-{axis}.pdf")
            #             )

            # plt.close("all")

            regions.loc[index, "gaussian_fit"] = int(gauss_valid)
            regions.loc[index, "x_gaussian"] = gauss_centroids[::-1][0]
            regions.loc[index, "y_gaussian"] = gauss_centroids[::-1][1]
            regions.loc[index, "fwhm_gaussian"] = float(numpy.mean(gauss_fwhm))
            gauss_residual_mean = (numpy.array(gauss_residual) ** 2).sum() ** 0.5
            regions.loc[index, "residual_gaussian"] = gauss_residual_mean

            regions.loc[index, "trapezoid_fit"] = int(trap_valid)
            regions.loc[index, "x_trapezoid"] = trap_centroids[::-1][0]
            regions.loc[index, "y_trapezoid"] = trap_centroids[::-1][1]
            regions.loc[index, "fwhm_trapezoid"] = float(numpy.mean(trap_fwhm))
            trap_residual_mean = (numpy.array(trap_residual) ** 2).sum() ** 0.5
            regions.loc[index, "residual_trapezoid"] = trap_residual_mean

            if not gauss_valid and not trap_valid:
                continue
            elif gauss_valid and gauss_residual_mean <= trap_residual_mean:
                regions.loc[index, "x_fit"] = gauss_centroids[::-1][0]
                regions.loc[index, "y_fit"] = gauss_centroids[::-1][1]
                regions.loc[index, "fwhm"] = float(numpy.mean(gauss_fwhm))
                regions.loc[index, "residual_fit"] = gauss_residual_mean
                regions.loc[index, "model_fit"] = "g"
            else:
                regions.loc[index, "x_fit"] = trap_centroids[::-1][0]
                regions.loc[index, "y_fit"] = trap_centroids[::-1][1]
                regions.loc[index, "fwhm"] = float(numpy.mean(trap_fwhm))
                regions.loc[index, "residual_fit"] = trap_residual_mean
                regions.loc[index, "model_fit"] = "t"

        regions.loc[regions.model_fit == "", "valid"] = 0

        if plot:
            self.plot_regions(
                regions.loc[regions.model_fit != ""],
                data,
                path=path,
                vmin=data.mean() - 5 * back.globalrms,
                vmax=data.mean() + 5 * back.globalrms,
                factor=1.0,
                xcen_col="x_fit",
                ycen_col="y_fit",
                a_col="fwhm",
                b_col="fwhm",
                theta_col=None,
                title=path.parts[-1] + " (marginal)",
            )

        return regions

    def reject(self, regions: pandas.DataFrame):
        """Rejects invalid FHWM measurements."""

        method = self.params.get("rejection_method", "sigclip")

        if self.star_finder in ["sextractor", "marginal"]:

            # Filter out bad FWHM values.
            ecc = numpy.sqrt(regions.a**2 - regions.b**2) / regions.a

            filter = (
                ((regions.a * self.pixel_scale) < 5)
                & ((regions.a * self.pixel_scale) > 0.4)
                & (regions.cpeak < 60000)
                & (ecc < 0.7)
            )

            regions.loc[~filter, "fwhm_valid"] = 0

        if method == "sigclip":
            fwhm = numpy.ma.array(
                regions.fwhm.values,
                mask=((regions.valid == 0) | (regions.fwhm_valid == 0)),
            )
            sigma = self.params.get("reject_sigma", 3.0)
            sigma_clip = SigmaClip(sigma, cenfunc="median")
            masked: Any = sigma_clip(fwhm, masked=True)
            regions.loc[masked.mask, "fwhm_valid"] = 0

        elif method == "nreject":
            nreject = self.params.get("nreject", 3)
            if isinstance(nreject, (int, float)):
                nreject = [int(nreject), int(nreject)]
            nreject = numpy.array(nreject, dtype=int)

            regions_sorted = regions.sort_values("fwhm", ascending=True)

            while True:
                # Endure we get at least some valid targets.
                valid = regions_sorted.iloc[nreject[0] : -nreject[1]]
                if len(valid) > 0:
                    break
                else:
                    nreject = nreject // 2

            regions.loc[~regions.index.isin(valid.index), "fwhm_valid"] = 0

        else:
            raise ValueError(f"Invalid rejection method {method!r}.")

        return regions

    def plot_regions(
        self,
        regions: pandas.DataFrame,
        data: numpy.ndarray,
        path: pathlib.Path,
        only_valid: bool = True,
        xcen_col: str = "x",
        ycen_col: str = "y",
        a_col: str = "a",
        b_col: str = "b",
        theta_col: str | None = "theta",
        factor: float = 1.0,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
    ):
        """Plot regions."""

        with plt.ioff():
            fig, ax = plt.subplots()

        ax.set_title(title or path.parts[-1])
        ax.grid(False)

        # Image
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

        if only_valid:
            regions = regions.loc[regions.valid == 1]

        # Centroids
        ax.scatter(
            regions[xcen_col],
            regions[ycen_col],
            marker="x",  # type: ignore
            s=6,
            c="b",
            linewidths=0.5,
        )

        # Ellipses colour-coded by FWHM
        for _, region in regions.iterrows():
            ell = Ellipse(
                (region[xcen_col], region[ycen_col]),
                region[b_col] * 2 * factor,
                region[a_col] * 2 * factor,
                region[theta_col] if theta_col else 0.0,
                facecolor="None",
                edgecolor="r",
                linewidth=0.5,
            )
            ell.set_clip_box(ax.bbox)
            ax.add_patch(ell)

        ax.set_xlim(0, data.shape[1] - 1)
        ax.set_ylim(0, data.shape[0] - 1)

        output_root = self._get_output_path(path)
        fig.savefig(str(output_root) + "-centroids.pdf")


def calculate_fwhm_from_ellipse(
    regions: pandas.DataFrame,
    pixel_scale: float,
    a_col: str = "a",
    b_col: str = "b",
):
    """Calcualtes the FWHM from a list of SExtractor detections.

    Parameters
    ----------
    regions
        The pandas data frame with the list of regions. Usually an output from
        ``sep``. Must include columns ``valid``, ``a``, and ``b``.
    pixel_scale
        The pixel scale in arcsec per pixel.
    a_col
        Column with the semi-major axis measurement.
    b_col
        Column with the semi-minor axis measurement.

    Returns
    -------
    dataframe
        The input ``regions`` data frame with an additional column ``fwhm``
        calculated from the ellipse parameters

    """

    fwhm_pixel = 2 * (numpy.log(2) * (regions[a_col] ** 2 + regions[b_col] ** 2)) ** 0.5
    fwhm = pixel_scale * fwhm_pixel

    regions = regions.copy()
    regions["fwhm"] = fwhm

    return regions
