# Changelog

## Next version

### 🚀 New

* `cherno acquire` now accepts a flag `--only-radec`. When passed, the RA and Dec offsets in the guide loop will calculated as a simple average translation between the GFA positions and the astrometric solutions. Rotation and scale are still measured and output as usual but they are not corrected. This approach is also used then the number of cameras solving is equal or below `config.acquisition.auto_radec_min` (set to 2 by default); this can be changed by setting the `--auto-radec-min` flag.

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
