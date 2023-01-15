# Changelog

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
