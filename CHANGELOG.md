# Changelog

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
