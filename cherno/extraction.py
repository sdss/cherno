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

from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
import sep
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
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
    obstime: Time
    observatory: str
    field_ra: float
    field_dec: float
    field_pa: float
    regions: pandas.DataFrame = field(default_factory=pandas.DataFrame)
    nregions: int = 0
    nvalid: int = 0
    fwhm_median: float = -999.0


class Extraction:
    """Extract centroids and PSF information from an image."""

    __VALID_STAR_FINDER = ["dao", "daophot", "sep", "sextractor"]

    def __init__(
        self,
        star_finder: str | None = None,
        pixel_scale: float | None = None,
        daophot_params={},
        **params,
    ):

        self.pixel_scale = pixel_scale or config["cameras"]["pixel_scale"]

        self.params = deepcopy(config["extraction"])
        self.params.update(params)
        self.params["daophot"].update(daophot_params)

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

        if self.star_finder in ["sep", "sextractor"]:
            regions = self._process_sextractor(data, path)
        elif self.star_finder in ["daophot", "dao"]:
            regions = self._process_daophot(data, path)
        else:
            regions = pandas.DataFrame()

        regions["mjd"] = mjd
        regions["exposure"] = exp_no
        regions["camera"] = cam_no

        if len(regions) > 0 and self.params["rejection_method"] is not None:
            self.reject(regions)

        extraction_data = ExtractionData(
            str(image),
            path,
            camera,
            exp_no,
            obstime,
            observatory,
            field_ra=header["RAFIELD"],
            field_dec=header["DECFIELD"],
            field_pa=header["FIELDPA"],
            regions=regions,
            nregions=len(regions),
            nvalid=sum(regions.valid == 1),
            fwhm_median=numpy.round(regions.loc[regions.valid == 1].fwhm.median(), 3),
        )

        output_file = self._get_output_path(path).with_suffix(".hdf")
        output_file.unlink(missing_ok=True)

        regions.to_hdf(str(output_file), "data")

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
            output_dir = dirname / self.output_dir
        else:
            output_dir = self.output_dir

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
            with plt.ioff():  # type: ignore
                fig, ax = plt.subplots()
                ax.imshow(back, origin="lower")
                ax.set_title("Background: " + path.parts[-1])
                ax.set_gid(False)
                fig.savefig(str(output_root) + "-background.pdf")

        regions = sep.extract(
            data - back.back(),
            self.params["background_sigma"],
            err=back.globalrms,
        )

        regions = pandas.DataFrame(regions)
        regions.loc[:, "valid"] = 0

        min_npix = self.params["min_npix"]
        if len(regions) > 0:
            regions.loc[regions.npix > min_npix, "valid"] = 1

            # Keep only detectons with flux and sort them by highest flux first.
            regions.dropna(subset=["flux"], inplace=True)
            regions.sort_values("flux", inplace=True, ascending=False)

        regions = calculate_fwhm_from_ellipse(regions)
        regions.fwhm *= self.pixel_scale

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

        return regions

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
            sep_regions = self._process_sextractor(data, path, plot=False)
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

        daogroup = DAOGroup(2.0 * psf_fwhm)

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
            fitshape=(11, 11),
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

    def reject(self, regions: pandas.DataFrame):
        """Rejects invalid FHWM measurements."""

        fwhm = regions.fwhm

        method = self.params.get("rejection_method", "sigclip")

        if method == "sigclip":
            sigma = self.params.get("reject_sigma", 3.0)
            sigma_clip = SigmaClip(sigma, cenfunc="median")
            masked: Any = sigma_clip(fwhm, masked=True)
            regions.loc[masked.mask, "valid"] = 0

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

            regions.loc[~regions.index.isin(valid.index), "valid"] = 0

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

        with plt.ioff():  # type: ignore
            fig, ax = plt.subplots()

        ax.set_title(title or path.parts[-1])
        ax.set_gid(False)

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
    a_col: str = "a",
    b_col: str = "b",
):
    """Calcualtes the FWHM from a list of SExtractor detections.

    Parameters
    ----------
    regions
        The pandas data frame with the list of regions. Usually an output from
        ``sep``. Must include columns ``valid``, ``a``, and ``b``.
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

    fwhm = numpy.max(numpy.array([regions[a_col] * 2, regions[b_col] * 2]), axis=0)

    # From Dylan
    # fwhm = 2 * numpy.sqrt(numpy.log(2) * (regions[a_col] ** 2 + regions[b_col] ** 2))

    regions = regions.copy()
    regions["fwhm"] = fwhm

    return regions
