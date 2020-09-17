#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-09-13
# @Filename: astrometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import subprocess


class AstrometryNet:
    """A wrapper for astrometry.net.

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

    def _build_command(self, files):

        flags = ['no-plots', 'sort-ascending']

        cmd = [self.solve_field_cmd]

        for option in self._options:
            if self._options[option] is None:
                continue
            if option in flags:
                if self._options[option] is True:
                    cmd.append('--' + option)
            else:
                cmd.append('--' + option)
                cmd.append(str(self._options[option]))

        cmd += list(files)

        return cmd

    def run(self, files, shell=True, stdout=None, stderr=None, **kwargs):

        if not isinstance(files, (tuple, list)):
            files = [files]

        cmd = ' '.join(self._build_command(files))
        solve_field = subprocess.run(cmd,
                                     capture_output=True,
                                     shell=shell)

        if stdout:
            with open(stdout, 'wb') as out:
                out.write(cmd.encode() + b'\n')
                out.write(solve_field.stdout)

        if stderr:
            with open(stderr, 'wb') as err:
                err.write(solve_field.stderr)

        return solve_field
