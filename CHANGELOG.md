# Changelog

## Next version

### 🔧 Fixed

* Fix a case in which the exposure time could increase in an uncontrolled way.


## 0.6.9 - April 18, 2024

### ✨ Improved

* [#18](https://github.com/sdss/cherno/pull/18) Two efficiency improvements:
  * The `wait_time` is only used when a correction has actually applied. Otherwise the next iteration start immediately.
  * Allow to specify a `--dynamic-exposure-time` that combined with `--max-exposure-time` can be used to set an initial short exposure time. If the field fails to solve the exposure time is automatically increased for the following iteration up to `--max-exposure-time`.

### 🏷️ Changed

* Set integral term for RA and Dec at APO to 0.01.


## 0.6.8 - February 27, 2024

### 🔧 Fixed

* Upgraded SQLAlchemy to `^2.0.0`. It seems that Pandas now requires `>=2.0` to run `read_sql` using a connection URI.
* Increased scale factor tolerance and `max_delta_scale_ppm` for LCO to account for the new IMB position and plate scale.
* Fixed docs building.


## 0.6.7 - February 7, 2024

### ✨ Improved

* [#17](https://github.com/sdss/cherno/pull/17) Allow to select what cameras to use for guiding or focusing by invoking `cherno set cameras [guide|focus] gfa2 gfa3 ...` (same as `cherno set cameras` but only affects guiding or focusing).


## 0.6.6 - January 15, 2024

### ✨ Improved

* Added a warning when a non-zero offset is used.
* Updated Pandas to 2.x.
* Bump `coordio` to 1.9.2 to use new `sdss-sep` package.


## 0.6.5 - December 10, 2023

### 🔧 Fixed

* Bump `sdsstools` to 1.5.5 to prevent an issue when pickling the `Guider` object in multiprocessing.

### ⚙️ Engineering

* Upgrade `astropy` to 6.0.0.


## 0.6.4 - December 7, 2023

### ✨ Improved

* Add `GFACOORD` extension to `proc-` frames with the `gfaCoords` table used during guiding.

### ⚙️ Engineering

* Stop supporting Python 3.9. Support for 3.12.
* Lint with `ruff`.
* Updated workflows for testing and releasing.


## 0.6.3 - July 28, 2023

### ✨ Improved

* Add a couple checks to reject fits where either all the cameras have bad fit RMS (controlled by the `guider.max_fit_rms` parameters) or the delta scale value is too large (`guider.max_delta_scale_ppm`).

### 🏷️ Changed

* Changed default rotation offset to -420 arcsec at APO and -365 arcsec at LCO.
* Changed default RA/Dec offsets to zero for APO.


## 0.6.2 - April 28, 2023

### 🔧 Fixed

* Use `sdss-coordio>=1.7.3` to exclude rejected cameras also for (normal) global RMS calculation.


## 0.6.1 - April 27, 2023

### 🔧 Fixed

* Use `sdss-coordio>=1.7.2` to exclude rejected cameras from global fit RMS calculation.
* Addressed a corner case in which if the guider fit was unsuccessful that could lead to an error.


## 0.6.0 - April 25, 2023

### ✨ Improved

* [#14](https://github.com/sdss/cherno/pull/14) Outputs fit RMS and rejects cameras with outlier fit RMS.

### 🏷️ Changed

* Apply RA/Dec corrections at LCO at the same time as rotator corrections.
* Changed rotation tolerance at LCO to 15 arcsec.
* Set the focus offset of all LCO cameras to zero. This prevents bad FWHM measurements when GFA1 (which had nominal focus offset zero) had bad FWHM, for example because of a galaxy, while the other cameras measured consistent FWHMs.

### 🔧 Fixed

* Only use measurements with RMS < 1 arcsec for the RMS and scale history.
* Use `get_sjd()` in `Exposer()` when determining the next sequence number for GFA exposures. This may be behind the `"Guider failed: The keyword filename_bundle was not output."`.
* Ignore corrections when delta is -999.0.


## 0.5.2 - January 15, 2023

### 🔥 Removed

* Removed the DAOPhot method and `photutils` dependency.

### ✨ Improved

* Upgraded the version of `coordio.extraction.extract_marginal()` to `coordio` 1.6.1.
* Limited the maximum number of detections to 50, sorted by flux, to speed up extraction.
* A few changes to source rejection.
* The FWHM returned with `astrometry_fit` now weights the offsets of the cameras so that cameras with large focus offsets count less towards the final value.

### 🏷️ Changed

* Use `--wait 15` by default at LCO when guiding/acquiring.

### 🔧 Fixed

* Fixed a couple references to `_acquisition_obj` that should be `_guider_obj` in `cherno set`.


## 0.5.1 - January 10, 2023

### 🏷️ Changed

* By default, apply one last correction after RMS reached.

### 🔧 Fixed

* Add back sigma-clipping to the median scale calculation.


## 0.5.0 - January 2, 2023

### 🚀 New

* [#10](https://github.com/sdss/cherno/issues/10) (COS-84) Added a `guide` command that replaces the current `acquire`. `acquire` is still there and does basically the same (they share the same code) but by default applies full corrections and is not continuous. `acquire` also accepts a `--target-rms` flag that will stop the acquisition process if that RMS is reached. In the future we may introduce more differences, with `acquire` aiming to quick acquisition, so some features may be disabled.
* [#11](https://github.com/sdss/cherno/issues/11) The marginal distribution extraction now uses the code from `coordio.extraction.extract_marginal()`.
* While the guide/acquire  loop is running, the guider status does not change to `IDLE`. If there is a delay between the loop iterations, the guider status will change to `WAITING`.


## 0.4.0 - December 20, 2022

### ✨ Improved

* Try to reconnect to the database if necessary when cross-matching with Gaia.
* Add `--fit-all-detections` flag for `cherno acquire` (default to `acquisition.fit_all_detections`) to adjust whether all detections or only the centre of each camera are used.
* `cherno get-scale` now calculates the weighted average, giving higher weigth to recent scale measurements.

### 🏷️ Changed

* Use astrometry.net 5200 indices for LCO.
* Use Gaia limit of `G<19` for APO

### 🔧 Fixed

* The cached Gaia query was not being used due to an issue retrieving the field ID.
* Fix typo in `--fit-all-detections`.


## 0.4.0b2 - October 20, 2022

### 🚀 New

* Added a `cherno config` command group that allows to change configuration parameters during runtime.

### ✨ Improved

* Improved how observatory configuration is managed. There is now a `set_observatory()` function that can be imported directory from `cherno` that allows to set the current observatory. On initial import, that function is called with the value of the `$OBSERVATORY` environment variable.


## 0.4.0b1 - October 15, 2022

### 🚀 New

* Added a new acquisition mode that uses cross-match with Gaia sources to generate a WCS solution for a field (see `coordio.guide.cross_match()`). This method first cross-correlates the detected regions with Gaia sources below a certain magnitude range (`acquisition.gaia_phot_g_mean_mag_max`) to determine the initial translation shift. It then uses a KD-tree nearest neighbout to determine the best matches and creates a WCS using Gaia astrometry. There are three different acquisition modes: `hybrid` (the default) that uses astrometry.net first and Gaia for the cameras that failed to solve; `astrometrynet` that uses only astrometry.net (the past behaviour); and `gaia` that uses Gaia cross-matching for all cameras.

### 🔧 Fixed

* The TCC module at APO was not applying negative rotator corrections!


## 0.3.2 - September 15, 2022

### 🚀 New

* `cherno acquire` now accepts a flag `--only-radec`. When passed, the RA and Dec offsets in the guide loop will calculated as a simple average translation between the GFA positions and the astrometric solutions. Rotation and scale are still measured and output as usual but they are not corrected. This approach is also used then the number of cameras solving is equal or below `config.acquisition.auto_radec_min` (set to 2 by default); this can be changed by setting the `--auto-radec-min` flag.

### ✨ Improved

* Allow to use all extracted regions (`config.acquisition.astrometry_net_use_all_regions`).

### 🔧 Fixed

* Calculate FWHM from all cameras with `fwhm_median > 0` instead of using cameras solved.


## 0.3.1 - September 11, 2022

### 🚀 New

* `version` command report the versions of astrometry.net, coordio, and fps_calibrations.
* `status` now reports the index paths for astrometry.net.

### ✨ Improved

* Split astrometry.net configuration for APO and LCO and use index_5200 for APO.


## 0.3.0 - September 11, 2022

### 🚀 New

* [#5](https://github.com/sdss/cherno/issues/5) Merge LCO changes.
* Split configuration between APO and LCO.
* Various changes to account for differences at LCO.
* Add `lcotcc.py` to apply corrections at LCO.
* Move most of the astrometry.net and fitting code to `coordio`.
* Implement full PID loop with `K`, `Ti` and `Td` terms.
* Separate RA/Dec corrections and PID coefficients.
* Added `converge` command.

### ✨ Improved

* Added an option to tweak the odds for astrometry.net to find a solution.
* Added option `--no-block` to `cherno acquire`.


## 0.2.0 - August 31, 2022

### 🚀 New

* [#2](https://github.com/sdss/cherno/issues/2) Output scale median.
* [#3](https://github.com/sdss/cherno/issues/3) Refactor extraction classes, use the DAOphot algorithm for extraction, and guide in focus.
* Allow to disable specific axes.
* Added PID and offset keywords.
* Added absolute offset parameter.
* Report `focus_data` for Boson plotting.

### ✨ Improved

* Allow to define what cameras to use.
* Output `PROCESSING` and `CORRECTING` guider states.
* Allow to reprocess raw images.
* Updated `focus_offset` values for APO.

* 🔧 Fixed

* Divide by `cos(dec)` when applying offset in RA.
* Use correct values for APO and LCO for focus sensitivity.


## 0.1.0 - January 7, 2022

### 🚀 New

* *Initial version.
* *Basic exposure and acquisition functionality using `astrometry.net`.
