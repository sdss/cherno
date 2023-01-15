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
from astropy.stats import SigmaClip, gaussian_sigma_to_fwhm
from astropy.time import Time, TimeDelta
from matplotlib.patches import Ellipse

from coordio.extraction import extract_marginal

from cherno import config


__all__ = ["ExtractionData", "Extraction"]

seaborn.set_theme(style="white")


PathLike = TypeVar("PathLike", pathlib.Path, str)


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

    __VALID_METHODS = ["sextractor", "marginal"]

    def __init__(
        self,
        observatory: str,
        method: str | None = None,
        pixel_scale: float | None = None,
        marginal_params={},
        **params,
    ):
        self.observatory = observatory.upper()
        self.pixel_scale = pixel_scale or config["pixel_scale"]

        self.params = deepcopy(config["extraction"])
        self.params.update(params)
        self.params["marginal"].update(marginal_params)

        self.output_dir = pathlib.Path(self.params["output_dir"])

        self.method = method or self.params["method"]
        if self.method not in self.__VALID_METHODS:
            raise ValueError(
                f"Invalid star finder. Valid values are {self.__VALID_METHODS}."
            )

    def process(self, image: PathLike, plot: bool | None = None) -> ExtractionData:
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

        if plot is None:
            plot = config["extraction"]["plot"]

        if self.method == "sextractor":
            regions = self._process_sextractor(data, path, plot=plot)[0]
        elif self.method == "marginal":
            regions = self._process_marginal(data, path, plot=plot)
        else:
            raise ValueError(f"Invalid extraction method {self.method}")

        regions["mjd"] = mjd
        regions["exposure"] = exp_no
        regions["camera"] = cam_no

        regions["fwhm_valid"] = 1
        regions.loc[regions.valid == 0, "fwhm_valid"] = 0

        if len(regions) > 0 and self.params["rejection_method"] is not None:
            self.reject(regions)

        fwhm_valid = regions.loc[regions.fwhm_valid == 1]

        if len(fwhm_valid) > 0:
            perc_25 = numpy.percentile(fwhm_valid.fwhm, 25)
            fwhm_median_round = float(numpy.round(perc_25, 3))

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
            algorithm=self.method,
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
            self.params["sextractor"]["background_sigma"],
            err=back.globalrms,
        )

        regions = pandas.DataFrame(regions)
        regions.index.set_names("regions_id", inplace=True)
        regions.loc[:, "valid"] = 0

        min_npix = self.params["sextractor"]["min_npix"]
        if len(regions) > 0:
            regions.loc[regions.npix > min_npix, "valid"] = 1

            # Keep only detectons with flux and sort them by highest flux first.
            regions.dropna(subset=["flux"], inplace=True)
            regions.sort_values("flux", inplace=True, ascending=False)

        regions = calculate_fwhm_from_ellipse(regions, self.pixel_scale)

        # Copy x,y columns as x1,y1. Guider expects those columns regardless of the
        # extraction method.
        regions[["x1", "y1"]] = regions.loc[:, ["x", "y"]]

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
        are fitted using both a 1D Gaussian. See ``coordio.extraction.extract_marginal``
        for details.

        """

        if plot is None:
            plot = self.params.get("plot", False)

        marginal_params = self.params["marginal"]

        output_root = self._get_output_path(path)
        plot_path = str(output_root) + "-marginal.pdf" if plot else None

        # Build an empty DF with the expected  columns to return in case
        # extract_marginal fails.
        default_columns = ["x1", "y1", "flux", "fwhm", "valid"]
        mock_regions = pandas.DataFrame([], columns=default_columns)

        try:
            regions = extract_marginal(
                data,
                marginal_params["background_sigma"],
                sextractor_quick_options={"minarea": marginal_params["minarea"]},
                max_detections=marginal_params.get("max_detections", None),
                plot=plot_path,
            )
        except Exception as err:
            warnings.warn(f"extract_marginal failed with error: {err}", UserWarning)
            return mock_regions

        if len(regions) == 0:
            return mock_regions

        # Reject detections in which any of the marginal fits failed.
        valid = regions.loc[:, ["xfitvalid", "yfitvalid"]].all(1).astype(int)
        regions["valid"] = valid

        # Calculate FWHM
        regions["fwhm"] = regions.loc[:, ["xstd", "ystd"]].mean(axis=1)
        regions["fwhm"] *= gaussian_sigma_to_fwhm * self.pixel_scale

        return regions

    def reject(self, regions: pandas.DataFrame):
        """Rejects invalid FHWM measurements."""

        method = self.params.get("rejection_method", "sigclip")

        if self.method in ["sextractor", "marginal"]:
            # Filter out bad FWHM values.
            ecc = numpy.sqrt(regions.a**2 - regions.b**2) / regions.a
            filter = (regions.cpeak < 60000) & (ecc < 0.8) & (ecc > 0.05)
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
                # Ensure we get at least some valid targets.
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
        vmin: float,
        vmax: float,
        only_valid: bool = True,
        xcen_col: str = "x",
        ycen_col: str = "y",
        a_col: str = "a",
        b_col: str = "b",
        theta_col: str | None = "theta",
        factor: float = 1.0,
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
            ell.set_clip_box(ax.bbox)  # type: ignore
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
