#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-09-13
# @Filename: astrometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import subprocess
import time


class AstrometryNet:
    """A wrapper for the astrometry.net ``solve-field`` command.

    Parameters
    ----------
    configure_params : dict
        Parameters to be passed to `.configure`.

    """

    def __init__(self, **configure_params):

        solve_field_cmd = subprocess.run('which solve-field',
                                         shell=True,
                                         capture_output=True)
        solve_field_cmd.check_returncode()

        self.solve_field_cmd = solve_field_cmd.stdout.decode().strip()

        self._options = {}
        self.configure(**configure_params)

    def configure(self, backend_config=None, width=None, height=None,
                  sort_column=None, sort_ascending=None, no_plots=None,
                  ra=None, dec=None, radius=None, scale_low=None,
                  scale_high=None, scale_units=None, dir=None):
        """Configures how to run of ``solve-field```.

        The parameters this method accepts are identical to those of
        ``solve-field`` and are passed unchanged.

        Parameters
        ----------
        backend_config : str
            Use this config file for the ``astrometry-engine`` program.
        width : int
            Specify the field width, in pixels.
        height : int
            Specify the field height, in pixels.
        sort_column : str
            The FITS column that should be used to sort the sources.
        sort_ascending : bool
            Sort in ascending order (smallest first);
            default is descending order.
        no_plot : bool
            Do not produce plots.
        ra : float
            RA of field center for search, in degrees.
        dec : float
            Dec of field center for search, in degrees.
        radius : float
            Only search in indexes within ``radius`` degrees of the field
            center given by ``ra`` and ``dec``.
        scale_low : float
            Lower bound of image scale estimate.
        scale_high : float
            Upper bound of image scale estimate.
        scale_units : str
            In what units are the lower and upper bounds? Choices:
            ``'degwidth'``, ``'arcminwidth'``, ``'arcsecperpix'``,
            ``'focalmm'``.
        dir : str
            Path to the directory where all output files will be saved.

        """

        self._options = {
            'backend-config': backend_config,
            'width': width,
            'height': height,
            'sort-column': sort_column,
            'sort-ascending': sort_ascending,
            'no-plots': no_plots,
            'ra': ra,
            'dec': dec,
            'radius': radius,
            'scale-low': scale_low,
            'scale-high': scale_high,
            'scale-units': scale_units,
            'dir': dir
        }

        return

    def _build_command(self, files, options=None):
        """Builds the ``solve-field`` command to run."""

        if options is None:
            options = self._options

        flags = ['no-plots', 'sort-ascending']

        cmd = [self.solve_field_cmd]

        for option in options:
            if options[option] is None:
                continue
            if option in flags:
                if options[option] is True:
                    cmd.append('--' + option)
            else:
                cmd.append('--' + option)
                cmd.append(str(options[option]))

        cmd += list(files)

        return cmd

    def run(self, files, shell=True, stdout=None, stderr=None, **kwargs):
        """ Runs astromerty.net.

        Parameters
        ----------
        files : list
            List of files to be processed.
        shell : bool
            Whether to call `subprocess.run` with ``shell=True``.
        stdout : str
            Path where to save the stdout output.
        stderr : str
            Path where to save the stderr output.
        kwargs : dict
            Configuration parameters (see `.configure`) to override. The
            configuration applies only to this run of ``solve-field`` and it
            is not saved.

        Returns
        -------
        `subprocess.CompletedProcess`
            The completed process.

        """

        options = self._options.copy()
        options.update(kwargs)

        if not isinstance(files, (tuple, list)):
            files = [files]

        cmd = ' '.join(self._build_command(files, options=options))

        t0 = time.time()

        solve_field = subprocess.run(cmd,
                                     capture_output=True,
                                     shell=shell)

        solve_field.time = time.time() - t0

        if stdout:
            with open(stdout, 'wb') as out:
                out.write(cmd.encode() + b'\n')
                out.write(solve_field.stdout)

        if stderr:
            with open(stderr, 'wb') as err:
                err.write(solve_field.stderr)

        return solve_field
