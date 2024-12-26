#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# ==========================================================================
# FRyDoM - frydom-ce.org
#
# Copyright (c) Ecole Centrale de Nantes (LHEEA lab.) and D-ICE Engineering.
# All rights reserved.
#
# Use of this source code is governed by a GPLv3 license that can be found
# in the LICENSE file of FRyDoM.
#
# ==========================================================================

import h5py
import numpy as np
from scipy import interpolate
from datetime import datetime

from hdb5_io.HDB5tool.wave_dispersion_relation import solve_wave_dispersion_relation

inf = float('inf') # Definition of infinity for depth.

class pyHDB(object):
    """
        Class for storing the hydrodynamique database.
    """

    def __init__(self):

        # Environmental data.
        self.rho_water = 0
        self.grav = 0
        self.depth = -1
        self.x_wave_measure = 0. # Only for Nemoh ?
        self.y_wave_measure = 0. # Only for Nemoh ?

        # Wave frequencies.
        self.nb_wave_freq = 0
        self.min_wave_freq = 0.
        self.max_wave_freq = 0.
        self.wave_freq = np.array([])
        self._iwcut = None

        # Wave directions.
        self.nb_wave_dir = 0
        self.min_wave_dir = 0. # deg.
        self.max_wave_dir = 0. # deg.
        self.wave_dir = np.array([]) # rad.

        # Symmetries.
        self.bottom_sym = None
        self.xoz_sym = None
        self.yoz_sym = None

        # Kochin parameters.
        self.has_kochin = False
        self.nb_angle_kochin = 0
        self.min_angle_kochin = 0. # deg.
        self.max_angle_kochin = 0. # deg.
        self.angle_kochin = np.array([]) # rad.
        self.kochin_diffraction = None # Diffraction Kochin functions.
        self.kochin_diffraction_derivative = None # Derivative of the diffraction Kochin functions.
        self.kochin_radiation = None # Radiation Kochin functions.
        self.kochin_radiation_derivative = None # Derivative of the radiation Kochin functions.
        self.nb_dir_kochin = 0 # Different from self.nb_wave_dir if the symmetry was used.
        self.min_dir_kochin = 0. # Different from self.min_wave_dir if the symmetry was used (deg).
        self.max_dir_kochin = 0. # Different from self.max_wave_dir if the symmetry was used (deg).
        self.wave_dir_kochin = np.array([]) # Different from self.wave_dir if the symmetry was used (rad).

        # The diffraction Kochin functions are not symetrised, only the mean wave drift loads are. That is why there are
        # two wave direction vectors : self.wave_dir_kochin and self.wave_dir.

        # Bodies.
        self.nb_bodies = 0
        self.bodies = []

        # Version.
        self.version = 4.0
        self.version_max = 4.0

        # Infinite added mass.
        self._has_infinite_added_mass = False

        # Froude-Krylov loads.
        self._has_froude_krylov = False

        # x-derivatives.
        self._has_x_derivatives = False

        # Impulse response functions.
        self.dt = None
        self.time = None
        self.nb_time_samples = None

        # Normalization length.
        self.normalization_length = 1.

        # RAO.
        self.has_RAO = False

        # Eigenfrequencies.
        self.has_Eigenfrequencies = False

        # Wave field.
        self.has_wave_field = False

        # Energy spectral moments.
        self.has_EnergySpectralMoments = False

        # Vector fitting.
        self.has_VF = False
        self.max_order = None
        self.relaxed = None
        self.tolerance = None

        # Solver.
        self.solver = None

        # Commit hash.
        self.commit_hash = None

        # Expert numerical parameters.
        self.has_expert_parameters = False
        self.surface_integration_order = None
        self.green_function = None
        self.crmax = None

    def set_wave_frequencies(self):
        """Frequency array of BEM computations in rad/s.

        Returns
        -------
        np.ndarray
        """

        self.wave_freq = np.linspace(self.min_wave_freq, self.max_wave_freq, self.nb_wave_freq, dtype=np.float64)

    @property
    def omega(self):
        """Frequency array of BEM computations in rad/s

        Returns
        -------
        np.ndarray
        """
        if self._iwcut is None:
            return self.wave_freq
        else:
            return self.wave_freq[:self._iwcut]

    def get_full_omega(self):
        return self.wave_freq

    @property
    def wcut(self):

        """This function gives the cutting wave frequency.

        Returns
        -------
        float
            Cutting wave frequency.
        """

        if self._iwcut is None:
            return None
        else:
            w = self.get_full_omega()
            return w[self._iwcut]

    @wcut.setter
    def wcut(self, wcut):  # TODO: finir l'implementation

        """This function sets the cutting wave frequency.

        Parameter
        ----------
        wcut : float
            Cutting wave frequency.
        """

        if wcut is None:
            self._iwcut = None
        else:
            assert self._min_frequency < wcut <= self._max_frequency
            w = self.get_full_omega()
            self._iwcut = np.where(w >= wcut)[0][0]  # TODO: a verifier

    @property
    def has_Drift(self):
        for body in self.bodies:
            if body.has_Drift:
                return True
        return False

    def set_wave_directions(self):
        """Frequency array of BEM computations in rad/s

        Returns
        -------
        np.ndarray
        """

        self.wave_dir = np.radians(np.linspace(self.min_wave_dir, self.max_wave_dir, self.nb_wave_dir, dtype=np.float64))

    def set_wave_directions_Kochin(self):
        """This function initializes the incident wave directions in the diffraction kochin problems.

        Returns
        -------
        np.ndarray
        """

        self.nb_dir_kochin = self.nb_wave_dir
        self.min_dir_kochin = self.min_wave_dir
        self.max_dir_kochin = self.max_wave_dir
        self.wave_dir_kochin = np.radians(np.linspace(self.min_dir_kochin, self.max_dir_kochin, self.nb_dir_kochin, dtype=np.float64))

    def append(self, body):

        """This function adds a body.

        Parameter
        ----------
        BodyDB : body
            Hydrodynamic body.
        """

        self.bodies.append(body)

    @property
    def nb_forces(self):

        """ This function computes the sum of the force modes for all bodies.
        Returns
        -------
        int
            Number of force modes for all bodies.
        """

        n = 0
        for body in self.bodies:
            n = n + int(body.Force_mask.sum())

        return n

    @property
    def nb_motions(self):

        """ This function computes the sum of the motion modes for all bodies.
        Returns
        -------
        int
            Number of motion modes for all bodies.
        """

        n = 0
        for body in self.bodies:
            n = n + int(body.Motion_mask.sum())

        return n

    def Eval_Froude_Krylov_loads(self):

        """ This functions computes the Froude-Krylov loads."""

        if not self._has_froude_krylov:

            """Computes the Froude-Krylov complex coefficients from indident wave field."""

            # Wave numbers.
            k_wave = solve_wave_dispersion_relation(self.wave_freq, self.depth, self.grav)

            # Computation of the Froude-Krylov loads for each body.
            for body in self.bodies:

                # Center of every face.
                centers = body.mesh.faces_centers
                x = centers[:, 0]
                y = centers[:, 1]
                z = centers[:, 2]

                # COMPUTING FIELDS.
                ctheta = np.cos(self.wave_dir) # cos(beta).
                stheta = np.sin(self.wave_dir) # sin(beta).

                kctheta = np.einsum('i, j -> ij', k_wave, ctheta) # k*cos(beta).
                kstheta = np.einsum('i, j -> ij', k_wave, stheta) # k*sin(beta).

                kw_bar = np.einsum('i, jk -> ijk', x - self.x_wave_measure, kctheta) # kw = k * (x - xref)*cos(beta).
                kw_bar += np.einsum('i, jk -> ijk', y - self.y_wave_measure, kstheta) # kw = k * ((x - xref)*cos(beta) + (y - yref)*sin(beta)).
                exp_jkw_bar = np.exp(1j * kw_bar) # e^(j * k * ((x - xref)*cos(beta) + (y - yref)*sin(beta))).

                if np.isinf(self.depth): # Infinite depth.

                    kxzph = np.einsum('i, j -> ij', z, k_wave) # k*z.
                    cih = np.exp(kxzph) # e^(kz).

                else: # Finite depth.

                    kxzph = np.einsum('i, j -> ij', z + self.depth, k_wave) # k*(z+h).
                    chkh_1 = 1. / np.cosh(k_wave * self.depth) # 1/ch(k*h).

                    cih = np.einsum('ij, j -> ij', np.cosh(kxzph), chkh_1) # ch(k(z+h)) / ch(k*h).

                cih_exp_jkw_bar = np.einsum('ij, ijk -> ijk', cih, exp_jkw_bar) # ch(k(z+h)) / ch(k*h) *  e^(j * k * ((x - xref)*cos(beta) + (y - yref)*sin(beta))).

                # Pressure.
                pressure = self.rho_water * self.grav * cih_exp_jkw_bar # rho * g * ch(k(z+h)) / ch(k*h) *  e^(j * k * ((x - xref)*cos(beta) + (y - yref)*sin(beta))).

                # Integration of the pressure of the wetted surface.
                for i_force in range(0,6):
                    if(body.Force_mask[i_force]):
                        nds = body.get_nds(i_force) # n*ds.
                        body.Froude_Krylov[i_force, :, :] = np.einsum('ijk, i -> jk', pressure, -nds) # Il s'agit de la normale entrante.

    def eval_impulse_response_function(self, full=True):
        """Computes the impulse response functions.

        It uses the Ogilvie formulas based on radiation damping integration (Inverse Fourier Transform).

        Parameters
        ----------
        full : bool, optional
            If True (default), it will use the full wave frequency range for computations.
        """

        # Wave frequency range.
        if full:
            w = self.get_full_omega()
        else:
            w = self.omega

        # Computation.
        wt = np.einsum('i, j -> ij', w, self.time) # w*t.
        cwt = np.cos(wt) # cos(w*t).

        for body in self.bodies:

            irf_data = np.empty(0, dtype=np.float64)

            if full:
                ca = np.einsum('ijk, ij -> ijk', body.Damping, body._flags) # B(w).
            else:
                ca = body.radiation_damping(self._iwcut) # B(w).

            kernel = np.einsum('ijk, kl -> ijkl', ca, cwt) # B(w)*cos(wt).

            irf_data = (2 / np.pi) * np.trapz(kernel, x=w, axis=2) # (2/pi) * Int(B(w)*cos(wt), dw).

            body.irf = irf_data

    def eval_infinite_added_mass(self, full=True):
        """Evaluates the infinite added mass matrix coefficients using Ogilvie formula.

        Parameter
        ---------
        full : bool, optional
            If True (default), it will use the full frequency range for computations.

         It uses the Ogilvie formula to get the coefficients from the impulse response functions.
        """

        #TODO: The value at w = 0 and so the interval [0, w_min] are not taken into account.

        if not self._has_infinite_added_mass:

            # Wave frequency range.
            if full:
                w = self.get_full_omega()
            else:
                w = self.omega

            # Computation.
            wt = np.einsum('i, j -> ij', w, self.time)  # w*t.
            sin_wt = np.sin(wt) # sin(w*t).

            for body in self.bodies:

                # Initialization.
                body.Inf_Added_mass = np.zeros((6, 6*self.nb_bodies), dtype = np.float64)

                # IRF.
                irf = body.irf

                # Added mass.
                if full:
                    cm = body.Added_mass
                else:
                    cm = body.radiation_added_mass(self._iwcut)

                kernel = np.einsum('ijk, lk -> ijlk', irf, sin_wt)  # irf*sin(w*t).
                integral = np.einsum('ijk, k -> ijk', np.trapz(kernel, x=self.time, axis=3), 1. / w)  # 1/w * int(irf*sin(w*t), dt).

                body.Inf_Added_mass = (cm + integral).mean(axis=2)  # mean( A(w) + 1/w * int(irf*sin(w*t), dt) ) wrt w.

    def eval_impulse_response_function_Ku(self, full=True):
        """Computes the impulse response functions proportional to the forward speed without x-derivatives.

        Parameter
        ---------
        full : bool, optional
            If True (default), it will use the full frequency range for computations.
        """

        # TODO: The value at w = 0 and so the interval [0, w_min] are not taken into account.

        # Wave frequency range.
        if full:
            w = self.get_full_omega()
        else:
            w = self.omega

        # Computation.
        wt = np.einsum('i, j ->ij', w, self.time) # w*t.
        cwt = np.cos(wt) # cos(w*t).

        for body in self.bodies:

            # Initialization.
            irf_data = np.empty(0, dtype=np.float64)

            # Added mass.
            if full:
                cm = np.einsum('ijk, ij -> ijk', body.Added_mass, body._flags) # A(w).
            else:
                cm = self.radiation_added_mass(self._iwcut) # A(w).

            # Infinite-frequency added mass.
            cm_inf = body.Inf_Added_mass # A(inf).

            cm_diff = np.zeros(cm.shape)
            for j in range(w.size):
                cm_diff[:, :, j] = cm_inf[:, :] - cm[:, :, j] # A(inf) - A(w).

            # [A(inf) - A(w)]*L.
            cm_diff[:, 4, :] = -cm_diff[:, 2, :]
            cm_diff[:, 5, :] = cm_diff[:, 1, :]
            cm_diff[:, 0, :] = 0.
            cm_diff[:, 1, :] = 0.
            cm_diff[:, 2, :] = 0.
            cm_diff[:, 3, :] = 0.

            kernel = np.einsum('ijk, kl -> ijkl', cm_diff, cwt) # int([A(inf) - A(w)]*L*cos(wt), dw).

            irf_data = (2. / np.pi) * np.trapz(kernel, x=w, axis=2) # (2/pi) * int([A(inf) - A(w)]*L*cos(wt), dw).

            body.irf_ku = irf_data

    def eval_impulse_response_function_Ku_x_derivative(self, full=True):
        """Computes the impulse response functions proportional to the forward speed with x-derivatives.

        Parameter
        ---------
        full : bool, optional
            If True (default), it will use the full frequency range for computations.
        """

        # TODO: The value at w = 0 and so the interval [0, w_min] are not taken into account.

        # Wave frequency range.
        if full:
            w = self.get_full_omega()
        else:
            w = self.omega

        # Computation.
        wt = np.einsum('i, j ->ij', w, self.time)  # w*t.
        cwt = np.cos(wt)  # cos(w*t).

        for body in self.bodies:

            # Initialization.
            irf_data = np.empty(0, dtype=np.float64)

            # x-derivative of the added mass.
            if full:
                cm = np.einsum('ijk, ij -> ijk', body.Added_mass_x_derivative, body._flags)
            else:
                cm = self.radiation_added_mass_x_derivative(self._iwcut)

            cm_inf = body.Inf_Added_mass_x_derivative

            cm_diff = np.zeros(cm.shape)
            for j in range(w.size):
                cm_diff[:, :, j] = cm_inf[:, :] - cm[:, :, j] # dAdx(inf) - dAdx(w).

            kernel = np.einsum('ijk, kl -> ijkl', cm_diff, cwt) # int((dAdx(inf) - dAdx(w))*cos(wt),dw).

            irf_data = (2. / np.pi) * np.trapz(kernel, x=w, axis=2) # (2/pi) * int((dAdx(inf) - dAdx(w))*cos(wt),dw).

            body.irf_ku_x_derivative = irf_data

    def eval_impulse_response_function_Ku2(self, full=True):
        """Computes the impulse response functions proportional to the square of the forward speed.

        Parameter
        ---------
        full : bool, optional
            If True (default), it will use the full frequency range for computations.
        """

        # TODO: The value at w = 0 and so the interval [0, w_min] are not taken into account.

        # Wave frequency range.
        if full:
            w = self.get_full_omega()
        else:
            w = self.omega

        # Computation.
        wt = np.einsum('i, j ->ij', w, self.time) # w*t.
        cwt = np.cos(wt) # cos(w*t).

        for body in self.bodies:

            # Initialization.
            irf_data = np.empty(0, dtype=np.float64)

            # x-derivative of the damping.
            ca = np.einsum('ijk, ij -> ijk', body.Damping_x_derivative / (w * w), body._flags) # dBdx(w) / w^2.

            # [dBdx(w) / w^2]*L.
            ca[:, 4, :] = -ca[:, 2, :]
            ca[:, 5, :] = ca[:, 1, :]
            ca[:, 0, :] = 0.
            ca[:, 1, :] = 0.
            ca[:, 2, :] = 0.
            ca[:, 3, :] = 0.

            kernel = np.einsum('ijk, kl -> ijkl', ca, cwt) # int([dBdx(w) / w^2]*L*cos(wt),dw).

            irf_data = -(2. / np.pi) * np.trapz(kernel, x=w, axis=2) # (2/pi) * int([dBdx(w) / w^2]*L*cos(wt),dw).

            body.irf_ku2 = irf_data

    def interpolation(self, discretization):
        """this function interpolates with respect to the wave directions and the wave frequencies."""

        for body in self.bodies:

            # Diffraction loads - Wave frequencies.
            f_interp_diffraction_freq = interpolate.interp1d(self.wave_freq, body.Diffraction, axis=1) # axis = 1 -> wave frequencies.
            body.Diffraction = f_interp_diffraction_freq(discretization._wave_frequencies)

            # x-derivative of the diffraction loads - Wave frequencies.
            if(self._has_x_derivatives):
                f_interp_diffraction_x_derivative_freq = interpolate.interp1d(self.wave_freq, body.Diffraction_x_derivative, axis=1) # axis = 1 -> wave frequencies.
                body.Diffraction_x_derivative = f_interp_diffraction_x_derivative_freq(discretization._wave_frequencies)

            # Froude-Krylov loads - Wave frequencies.
            f_interp_fk_freq = interpolate.interp1d(self.wave_freq, body.Froude_Krylov, axis=1) # axis = 1 -> wave frequencies.
            body.Froude_Krylov = f_interp_fk_freq(discretization._wave_frequencies)

            # x-derivative of the Froude-Krylov loads - Wave frequencies.
            if (self._has_x_derivatives):
                f_interp_fk_x_derivative_freq = interpolate.interp1d(self.wave_freq, body.Froude_Krylov_x_derivative, axis=1) # axis = 1 -> wave frequencies.
                body.Froude_Krylov_x_derivative = f_interp_fk_x_derivative_freq(discretization._wave_frequencies)

            # Added mass - Wave frequencies.
            f_interp_Added_mass_freq = interpolate.interp1d(self.wave_freq, body.Added_mass, axis=2) # axis = 2 -> wave frequencies.
            body.Added_mass = f_interp_Added_mass_freq(discretization._wave_frequencies)

            # x-derivative of the added mass - Wave frequencies.
            if (self._has_x_derivatives):
                f_interp_Added_mass_x_derivative_freq = interpolate.interp1d(self.wave_freq, body.Added_mass_x_derivative, axis=2) # axis = 2 -> wave frequencies.
                body.Added_mass_x_derivative = f_interp_Added_mass_x_derivative_freq(discretization._wave_frequencies)

            # Damping - Wave frequencies.
            f_interp_Damping_freq = interpolate.interp1d(self.wave_freq, body.Damping, axis=2) # axis = 2 -> wave frequencies.
            body.Damping = f_interp_Damping_freq(discretization._wave_frequencies)

            # x-derivative of the damping - Wave frequencies.
            if (self._has_x_derivatives):
                f_interp_Damping_x_derivative_freq = interpolate.interp1d(self.wave_freq, body.Damping_x_derivative, axis=2) # axis = 2 -> wave frequencies.
                body.Damping_x_derivative = f_interp_Damping_x_derivative_freq(discretization._wave_frequencies)

            # Wave directions.
            if(self.nb_wave_dir > 1): # Several wave directions, so the interpolations are possible.
                if (discretization._nb_wave_directions is not None):
                    if (discretization._nb_wave_directions > 1):

                        # Diffraction loads - Wave directions.
                        f_interp_diffraction_dir = interpolate.interp1d(self.wave_dir, body.Diffraction, axis=2) # axis = 2 -> wave directions.
                        body.Diffraction = f_interp_diffraction_dir(discretization._wave_dirs) # Application of the interpolation.

                        # x-derivative of the diffraction loads - Wave directions.
                        if (self._has_x_derivatives):
                            f_interp_diffraction_x_derivative_dir = interpolate.interp1d(self.wave_dir, body.Diffraction_x_derivative, axis=2)  # axis = 2 -> wave directions.
                            body.Diffraction_x_derivative = f_interp_diffraction_x_derivative_dir(discretization._wave_dirs)  # Application of the interpolation.

                        # Froude-Krylov loads - Wave directions.
                        f_interp_fk_dir = interpolate.interp1d(self.wave_dir, body.Froude_Krylov, axis=2) # axis = 2 -> wave directions.
                        body.Froude_Krylov = f_interp_fk_dir(discretization._wave_dirs) # Application of the interpolation.

                        # x-derivative of the Froude-Krylov loads - Wave directions.
                        if (self._has_x_derivatives):
                            f_interp_fk_x_derivative_dir = interpolate.interp1d(self.wave_dir, body.Froude_Krylov_x_derivative, axis=2)  # axis = 2 -> wave directions.
                            body.Froude_Krylov_x_derivative = f_interp_fk_x_derivative_dir(discretization._wave_dirs)  # Application of the interpolation.

            else: # Only one wave direction so the data are copied along a second direction.

                # Diffraction loads - Wave directions.
                body.Diffraction = np.repeat(body.Diffraction, 2, axis=2) # axis = 2 -> wave directions.

                # x-derivative of the diffraction loads - Wave directions.
                if (self._has_x_derivatives):
                    body.Diffraction_x_derivative = np.repeat(body.Diffraction_x_derivative, 2, axis=2)  # axis = 2 -> wave directions.

                # Froude-Krylov loads - Wave directions.
                body.Froude_Krylov = np.repeat(body.Froude_Krylov, 2, axis=2) # axis = 2 -> wave directions.

                # x-derivative of the Froude-Krylov loads - Wave directions.
                if (self._has_x_derivatives):
                    body.Froude_Krylov_x_derivative = np.repeat(body.Froude_Krylov_x_derivative, 2, axis=2)  # axis = 2 -> wave directions.

            # Mean wave drift loads.
            if body.has_Drift:

                # Wave frequencies.
                f_interp_Drift_freq = interpolate.interp1d(self.wave_freq, body.Wave_drift_force, axis=1) # axis = 1 -> wave frequencies.
                body.Wave_drift_force = f_interp_Drift_freq(discretization._wave_frequencies)

                # Wave directions.
                if (self.nb_wave_dir > 1): # Several wave directions, so the interpolation is possible.
                    if (discretization._nb_wave_directions is not None):
                        if (discretization._nb_wave_directions > 1):

                            f_interp_Drift_dir = interpolate.interp1d(self.wave_dir, body.Wave_drift_force, axis=2) # axis = 2 -> wave directions.
                            body.Wave_drift_force = f_interp_Drift_dir(discretization._wave_dirs) # Application of the interpolation.

                else:  # Only one wave direction so the data are copied along a second direction.

                    body.Wave_drift_force = np.repeat(body.Wave_drift_force, 2, axis=2) # axis = 2 -> wave directions.


        # Kochin functions.
        if(self.has_kochin):

            # Diffraction Kochin functions - Wave frequencies.
            f_interp_kochin_diffraction_freq = interpolate.interp1d(self.wave_freq, self.kochin_diffraction, axis=1) # axis = 1 -> wave frequencies.
            self.kochin_diffraction = f_interp_kochin_diffraction_freq(discretization._wave_frequencies) # Application of the interpolation.

            # Wave directions.
            if (self.nb_wave_dir > 1):  # Several wave directions, so the interpolations are possible.

                # Diffraction Kochin functions - Wave directions.
                f_interp_kochin_diffraction_dir = interpolate.interp1d(self.wave_dir, self.kochin_diffraction, axis=0) # axis = 0 -> wave directions.
                self.kochin_diffraction = f_interp_kochin_diffraction_dir(discretization._wave_dirs) # Application of the interpolation.

            else: # Only one wave direction so the data are copied along a second direction.

                # Diffraction Kochin functions - Wave directions.
                self.kochin_diffraction = np.repeat(self.kochin_diffraction, 2, axis=0) # axis = 0 -> wave directions.

            # Derivatives of the diffraction Kochin functions.
            if(self.kochin_diffraction_derivative is not None):

                # Diffraction Kochin function derivatives - Wave frequencies.
                f_interp_kochin_derivative_diffraction_freq = interpolate.interp1d(self.wave_freq, self.kochin_diffraction_derivative, axis=1)  # axis = 1 -> wave frequencies.
                self.kochin_diffraction_derivative = f_interp_kochin_derivative_diffraction_freq(discretization._wave_frequencies)  # Application of the interpolation.

                # Wave directions.
                if (self.nb_wave_dir > 1):  # Several wave directions, so the interpolations are possible.

                    # Diffraction Kochin function derivatives - Wave directions.
                    f_interp_kochin_derivative_diffraction_dir = interpolate.interp1d(self.wave_dir, self.kochin_diffraction_derivative, axis=0)  # axis = 0 -> wave directions.
                    self.kochin_diffraction_derivative = f_interp_kochin_derivative_diffraction_dir(discretization._wave_dirs)  # Application of the interpolation.

                else:  # Only one wave direction so the data are copied along a second direction.

                    # Diffraction Kochin function derivatives - Wave directions.
                    self.kochin_diffraction_derivative = np.repeat(self.kochin_diffraction_derivative, 2, axis=0)  # axis = 0 -> wave directions.

            # Radiation Kochin functions - Wave frequencies.
            f_interp_kochin_radiation_freq = interpolate.interp1d(self.wave_freq, self.kochin_radiation, axis=1) # axis = 1 -> wave frequencies.
            self.kochin_radiation = f_interp_kochin_radiation_freq(discretization._wave_frequencies) # Application of the interpolation.

            # Derivatives of the radiation Kochin functions.
            if (self.kochin_radiation_derivative is not None):
                f_interp_kochin_derivative_radiation_freq = interpolate.interp1d(self.wave_freq, self.kochin_radiation_derivative, axis=1)  # axis = 1 -> wave frequencies.
                self.kochin_radiation_derivative = f_interp_kochin_derivative_radiation_freq(discretization._wave_frequencies)  # Application of the interpolation.

        # Update wave directions and frequencies vectors.
        self.min_wave_freq = discretization._min_frequency
        self.max_wave_freq = discretization._max_frequency
        self.nb_wave_freq = discretization.nb_frequencies
        self.wave_freq = discretization._wave_frequencies

        self.min_wave_dir = discretization._min_angle
        self.max_wave_dir = discretization._max_angle
        self.nb_wave_dir = discretization._nb_wave_directions
        self.wave_dir = discretization._wave_dirs

        if (self.has_kochin):
            self.min_dir_kochin = discretization._min_angle
            self.max_dir_kochin = discretization._max_angle
            self.nb_dir_kochin = discretization._nb_wave_directions
            self.wave_dir_kochin = discretization._wave_dirs

        print("")
        print("-- Interpolations --")
        print(" Min frequency: %3.2f" % self.min_wave_freq)
        print(" Max frequency: %3.2f" % self.max_wave_freq)
        print(" Nb Wave frequencies: %i" % self.nb_wave_freq)
        print(" Angle min: %3.2f" % self.min_wave_dir)
        print(" Angle max: %3.2f" % self.max_wave_dir)
        print(" Nb Wave directions: %i" % self.nb_wave_dir)
        print("")

    def symetrize(self):

        """This function updates the hdb due to a modification of the wave direction convention."""

        # The diffraction Kochin functions are not symetrised, only the mean wave drift loads are. That is why there is
        # two wave direction vectors : self.wave_dir_kochin and self.wave_dir.

        ndir = self.nb_wave_dir
        nw = self.nb_wave_freq

        for i in range(ndir):

            if(np.degrees(self.wave_dir[i]) > np.float32(0.)):

                # New wave direction.
                new_dir = -np.degrees(self.wave_dir[i]) % 360
                if new_dir < 0:
                    new_dir += 360.

                # Add corresponding data.
                self.wave_dir = np.append(self.wave_dir, np.radians(new_dir))

                # Loop over the bodies.
                for body in self.bodies:

                    # Froude-Krylov loads.
                    fk_db_temp = np.copy(body.Froude_Krylov[:, :, i])
                    fk_db_temp[(1, 3, 5), :] = -fk_db_temp[(1, 3, 5), :] # 1 = sway, 3 = roll and 5 = yaw.
                    body.Froude_Krylov = np.concatenate((body.Froude_Krylov, fk_db_temp.reshape(6, nw, 1)), axis=2) # Axis of the wave directions.

                    # x-derivative of the Froude-Krylov loads.
                    if (self._has_x_derivatives):
                        fk_x_derivative_db_temp = np.copy(body.Froude_Krylov_x_derivative[:, :, i])
                        fk_x_derivative_db_temp[(1, 3, 5), :] = -fk_x_derivative_db_temp[(1, 3, 5), :]  # 1 = sway, 3 = roll and 5 = yaw.
                        body.Froude_Krylov_x_derivative = np.concatenate((body.Froude_Krylov_x_derivative, fk_x_derivative_db_temp.reshape(6, nw, 1)), axis=2)  # Axis of the wave directions.

                    # Diffraction loads.
                    diff_db_temp = np.copy(body.Diffraction[:, :, i])
                    diff_db_temp[(1, 3, 5), :] = -diff_db_temp[(1, 3, 5), :] # 1 = sway, 3 = roll and 5 = yaw.
                    body.Diffraction = np.concatenate((body.Diffraction, diff_db_temp.reshape(6, nw, 1)), axis=2) # Axis of the wave directions.

                    # x-derivative of the diffraction loads.
                    if (self._has_x_derivatives):
                        diff_x_derivative_db_temp = np.copy(body.Diffraction_x_derivative[:, :, i])
                        diff_x_derivative_db_temp[(1, 3, 5), :] = -diff_x_derivative_db_temp[(1, 3, 5), :]  # 1 = sway, 3 = roll and 5 = yaw.
                        body.Diffraction_x_derivative = np.concatenate((body.Diffraction_x_derivative, diff_x_derivative_db_temp.reshape(6, nw, 1)), axis=2)  # Axis of the wave directions.

                    # RAO.
                    if(self.has_RAO):
                        RAO_db_temp = np.copy(body.RAO[:, :, i])
                        RAO_db_temp[(1, 3, 5), :] = -RAO_db_temp[(1, 3, 5), :]  # 1 = sway, 3 = roll and 5 = yaw.
                        body.RAO = np.concatenate((body.RAO, RAO_db_temp.reshape(6, nw, 1)), axis=2)  # Axis of the wave directions.

                    # Wave drift loads.
                    if(body.has_Drift):
                        Drift_db_temp = np.copy(body.Wave_drift_force[:, :, i])
                        Drift_db_temp[(1, 3, 5), :] = -Drift_db_temp[(1, 3, 5), :] # 1 = sway and 2 = yaw.
                        body.Wave_drift_force = np.concatenate((body.Wave_drift_force, Drift_db_temp.reshape(6, nw, 1)), axis=2)
        return

    def _initialize_wave_dir(self):

        """This function updates the wave directions by adjusting the convention with the one used in FRyDoM, the FK and diffraction loads are updated accordingly."""

        # Symmetrization.
        if self.min_wave_dir >= -np.float32() and self.max_wave_dir <= 180. + np.float32():
            self.symetrize()

        # Updating the loads accordingly.
        n180 = 0
        i360 = -9
        for idir in range(self.wave_dir.size):
            wave_dir = self.wave_dir[idir]

            if abs(np.degrees(wave_dir)) < 0.01:
                i360 = idir
            elif abs(np.degrees(wave_dir) - 180) < 0.01:
                n180 += 1
                if n180 == 2:
                    self.wave_dir[idir] = np.radians(360.)

                    # Loop over the bodies.
                    for body in self.bodies:

                        # Froude-Krylov loads.
                        body.Froude_Krylov[:, :, idir] = body.Froude_Krylov[:, :, i360]

                        # x-derivative of the Froude-Krylov loads.
                        if (self._has_x_derivatives):
                            body.Froude_Krylov_x_derivative[:, :, idir] = body.Froude_Krylov_x_derivative[:, :, i360]

                        # Diffraction loads.
                        body.Diffraction[:, :, idir] = body.Diffraction[:, :, i360]

                        # x-derivative of the diffraction loads.
                        if (self._has_x_derivatives):
                            body.Diffraction_x_derivative[:, :, idir] = body.Diffraction_x_derivative[:, :, i360]

                        # RAO.
                        if (self.has_RAO):
                            body.RAO[:, :, idir] = body.RAO[:, :, i360]

                        # Wave drift loads.
                        if (body.has_Drift):
                            body.Wave_drift_force[:, :, idir] = body.Wave_drift_force[:, :, i360]

        # Sorting wave directions and creates the final FK and diffraction loads data.
        sort_dirs = np.argsort(self.wave_dir)
        self.wave_dir = self.wave_dir[sort_dirs]

        # Loop over the bodies.
        for body in self.bodies:

            # Froude-Krylov loads.
            body.Froude_Krylov = body.Froude_Krylov[:, :, sort_dirs]

            # x-derivative of the Froude-Krylov loads.
            if (self._has_x_derivatives):
                body.Froude_Krylov_x_derivative = body.Froude_Krylov_x_derivative[:, :, sort_dirs]

            # Diffraction loads.
            body.Diffraction = body.Diffraction[:, :, sort_dirs]

            # x-derivative of the diffraction loads.
            if (self._has_x_derivatives):
                body.Diffraction_x_derivative = body.Diffraction_x_derivative[:, :, sort_dirs]

            # RAO.
            if (self.has_RAO):
                body.RAO = body.RAO[:, :, sort_dirs]

            # Wave drift loads.
            if (body.has_Drift):
                body.Wave_drift_force = body.Wave_drift_force[:, :, sort_dirs]

        # Update parameters.
        self.min_wave_dir = np.degrees(np.min(self.wave_dir)) # deg.
        self.max_wave_dir = np.degrees(np.max(self.wave_dir)) # deg.
        self.nb_wave_dir = self.wave_dir.shape[0]

        print("")
        print("-- Symmetrization --")
        print(" Angle min: %3.2f" % self.min_wave_dir)
        print(" Angle max: %3.2f" % self.max_wave_dir)
        print(" Nb Wave directions: %i" % self.nb_wave_dir)
        print("")

    def write_hdb5(self, hdb5_file):
        """This function writes the hydrodynamic database into a *.hdb5 file.

        Parameter
        ---------
        output_file : string, optional.
            Name of the hdf5 output file.
        """

        with h5py.File(hdb5_file, 'w') as writer:

            # Environment.
            self.write_environment(writer)

            # Discretization.
            self.write_discretization(writer)

            # Symmetries.
            self.write_symmetries(writer)

            # Bodies.
            for body in self.bodies:
                self.write_body(writer, body)

            # Wave field.
            if(self.has_wave_field):
                self.write_wave_field(writer, "/WaveField")

            # Vector fitting.
            if(self.has_VF):
                self.write_VF(writer, "/VectorFitting")

            # Version.
            self.write_version(writer)

            # Commit hash.
            if(self.commit_hash is not None):
                self.write_commit_hash(writer)

            # Expert numerical parameters.
            if(self.has_expert_parameters):
                self.write_numerical_parameters(writer, "/ExpertParameters")

    def write_environment(self, writer):
        """This function writes the environmental data into the *.hdb5 file.

        Parameter
        ---------
        writer : string.
            *.hdb5 file.
        """

        # Date.
        dset = writer.create_dataset('CreationDate', data=str(datetime.now()))
        dset.attrs['Description'] = "Date of the creation of this database."

        # Gravity acceleration.
        dset = writer.create_dataset('GravityAcc', data=self.grav)
        dset.attrs['Unit'] = 'm/s**2'
        dset.attrs['Description'] = "Gravity acceleration."

        # Water density.
        dset = writer.create_dataset('WaterDensity', data=self.rho_water)
        dset.attrs['Unit'] = 'kg/m**3'
        dset.attrs['Description'] = 'Water Density.'

        # Normalisation length.
        dset = writer.create_dataset('NormalizationLength', data=self.normalization_length)
        dset.attrs['Unit'] = 'm'
        dset.attrs['Description'] = 'Normalization length.'

        # Water depth.
        dset = writer.create_dataset('WaterDepth', data=self.depth, dtype=float)
        dset.attrs['Unit'] = 'm'
        dset.attrs['Description'] = 'Water depth: 0 for infinite depth and positive for finite depth.'

        # Number of bodies.
        dset = writer.create_dataset('NbBody', data=self.nb_bodies)
        dset.attrs['Description'] = 'Number of hydrodynamic bodies.'

        # Solver.
        dset = writer.create_dataset('Solver', data=self.solver)
        dset.attrs['Description'] = 'Hydrodynamic solver used for computing the hydrodynamic database.'

    def write_discretization(self,writer):
        """This function writes the discretization parameters into the *.hdb5 file.

        Parameter
        ---------
        Writer : string.
            *.hdb5 file.
        """

        discretization_path = "/Discretizations"
        writer.create_group(discretization_path)

        # Frequency discretization.

        frequential_path = discretization_path + "/Frequency"

        dset = writer.create_dataset(frequential_path, data=self.wave_freq)
        dset.attrs['Unit'] = "rad/s"
        dset.attrs['Description'] = "Wave frequencies."

        # Wave direction discretization.

        wave_direction_path = discretization_path + "/WaveDirection"

        dset = writer.create_dataset(wave_direction_path, data=self.wave_dir * 180 / np.pi)
        dset.attrs['Unit'] = "deg"
        dset.attrs['Description'] = "Wave directions."

        # Time sample.

        if self.time is not None:
            time_path = discretization_path + "/Time"

            dset = writer.create_dataset(time_path, data=self.time)
            dset.attrs['Description'] = "Time samples."
            dset.attrs['Unit'] = "s"
            dset.attrs['Description'] = "Final time for the evaluation of the impulse response functions."

    def write_symmetries(self,writer):
        """This function writes the symmetry parameters into the *.hdb5 file.

        Parameter
        ---------
        Writer : string.
            *.hdb5 file.
        """

        symmetry_path = "/Symmetries"
        writer.create_group(symmetry_path)

        dset = writer.create_dataset(symmetry_path + "/Bottom", data=self.bottom_sym, dtype=int)
        dset.attrs['Description'] = "Bottom symmetry."

        dset = writer.create_dataset(symmetry_path + "/xOz", data=self.xoz_sym, dtype=int)
        dset.attrs['Description'] = "(xOz) symmetry."

        dset = writer.create_dataset(symmetry_path + "/yOz", data=self.yoz_sym, dtype=int)
        dset.attrs['Description'] = "(yOz)) symmetry."

    def write_mask(self, writer, body, mask_path="/Mask"):
        """This function writes the Force and Motion masks into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        mask_path : string, optional
            Path to the masks.
        """

        writer.create_group(mask_path)
        writer.create_dataset(mask_path + "/MotionMask", data=body.Motion_mask.astype(int))
        writer.create_dataset(mask_path + "/ForceMask", data=body.Force_mask.astype(int))

    def write_mesh(self, writer, body, mesh_path="/Mesh"):

        """This function writes the mesh quantities into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        mesh_path : string
            Path to the mesh folder.
        """

        writer.create_dataset(mesh_path + "/NbVertices", data=body.mesh.nb_vertices)
        writer.create_dataset(mesh_path + "/Vertices", data=body.mesh.vertices)
        writer.create_dataset(mesh_path + "/NbFaces", data=body.mesh.nb_faces)
        writer.create_dataset(mesh_path + "/Faces", data=body.mesh.faces)

    def write_excitation(self, writer, body, excitation_path="/Excitation"):

        """This function writes the diffraction and Froude-Krylov loads into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        excitation_path : string, optional
            Path to excitation loads.
        """

        # Froude-Krylov loads.

        fk_path = excitation_path + "/FroudeKrylov"
        writer.create_group(fk_path)

        for idir in range(0,self.nb_wave_dir):

            wave_dir_path = fk_path + "/Angle_%u" % idir
            writer.create_group(wave_dir_path)

            dset = writer.create_dataset(wave_dir_path + "/Angle", data=np.degrees(self.wave_dir[idir]))
            dset.attrs['Unit'] = 'deg'
            dset.attrs['Description'] = "Wave direction."

            # Real parts.
            dset = writer.create_dataset(wave_dir_path + "/RealCoeffs", data=body.Froude_Krylov[:, :, idir].real)
            dset.attrs['Unit'] = ''
            dset.attrs['Description'] = "Real part of the Froude-Krylov loads " \
                                        "on body %u for a wave direction of %.1f deg." % \
                                        (body.i_body, np.degrees(self.wave_dir[idir]))

            # Imaginary parts.
            dset = writer.create_dataset(wave_dir_path + "/ImagCoeffs", data=body.Froude_Krylov[:, :, idir].imag)
            dset.attrs['Unit'] = ''
            dset.attrs['Description'] = "Imaginary part of the Froude-Krylov loads " \
                                        "on body %u for a wave direction of %.1f deg." % \
                                        (body.i_body, np.degrees(self.wave_dir[idir]))

        # x-derivative of the Froude-Krylov loads.
        if (self._has_x_derivatives):

            fk_x_derivative_path = excitation_path + "/FroudeKrylovXDerivative"
            writer.create_group(fk_x_derivative_path)

            for idir in range(0, self.nb_wave_dir):

                wave_dir_path = fk_x_derivative_path + "/Angle_%u" % idir
                writer.create_group(wave_dir_path)

                dset = writer.create_dataset(wave_dir_path + "/Angle", data=np.degrees(self.wave_dir[idir]))
                dset.attrs['Unit'] = 'deg'
                dset.attrs['Description'] = "Wave direction."

                # Real parts.
                dset = writer.create_dataset(wave_dir_path + "/RealCoeffs", data=body.Froude_Krylov_x_derivative[:, :, idir].real)
                dset.attrs['Unit'] = ''
                dset.attrs['Description'] = "Real part of the x-derivative of the Froude-Krylov loads " \
                                            "on body %u for a wave direction of %.1f deg." % \
                                            (body.i_body, np.degrees(self.wave_dir[idir]))

                # Imaginary parts.
                dset = writer.create_dataset(wave_dir_path + "/ImagCoeffs", data=body.Froude_Krylov_x_derivative[:, :, idir].imag)
                dset.attrs['Unit'] = ''
                dset.attrs['Description'] = "Imaginary part of the x-derivative of the Froude-Krylov loads " \
                                            "on body %u for a wave direction of %.1f deg." % \
                                            (body.i_body, np.degrees(self.wave_dir[idir]))

        # Diffraction loads.

        diffraction_path = excitation_path + "/Diffraction"
        writer.create_group(diffraction_path)

        for idir in range(0,self.nb_wave_dir):

            wave_dir_path = diffraction_path + "/Angle_%u" % idir
            writer.create_group(wave_dir_path)

            dset = writer.create_dataset(wave_dir_path + "/Angle", data=np.degrees(self.wave_dir[idir]))
            dset.attrs['Unit'] = 'deg'
            dset.attrs['Description'] = "Wave direction."

            # Real parts.
            writer.create_dataset(wave_dir_path + "/RealCoeffs", data=body.Diffraction[:, :, idir].real)
            dset.attrs['Unit'] = ''
            dset.attrs['Description'] = "Real part of the diffraction loads " \
                                        "on body %u for a wave direction of %.1f deg." % \
                                        (body.i_body, np.degrees(self.wave_dir[idir]))

            # Imaginary parts.
            writer.create_dataset(wave_dir_path + "/ImagCoeffs", data=body.Diffraction[:, :, idir].imag)
            dset.attrs['Unit'] = ''
            dset.attrs['Description'] = "Imaginary part of the diffraction loads " \
                                        "on body %u for a wave direction of %.1f deg." % \
                                        (body.i_body, np.degrees(self.wave_dir[idir]))

        # x-derivative of the diffraction loads.
        if (self._has_x_derivatives):

            diffraction_x_derivative_path = excitation_path + "/DiffractionXDerivative"
            writer.create_group(diffraction_x_derivative_path)

            for idir in range(0, self.nb_wave_dir):
                wave_dir_path = diffraction_x_derivative_path + "/Angle_%u" % idir
                writer.create_group(wave_dir_path)

                dset = writer.create_dataset(wave_dir_path + "/Angle", data=np.degrees(self.wave_dir[idir]))
                dset.attrs['Unit'] = 'deg'
                dset.attrs['Description'] = "Wave direction."

                # Real parts.
                writer.create_dataset(wave_dir_path + "/RealCoeffs", data=body.Diffraction_x_derivative[:, :, idir].real)
                dset.attrs['Unit'] = ''
                dset.attrs['Description'] = "Real part of the x-derivative of the diffraction loads " \
                                            "on body %u for a wave direction of %.1f deg." % \
                                            (body.i_body, np.degrees(self.wave_dir[idir]))

                # Imaginary parts.
                writer.create_dataset(wave_dir_path + "/ImagCoeffs", data=body.Diffraction_x_derivative[:, :, idir].imag)
                dset.attrs['Unit'] = ''
                dset.attrs['Description'] = "Imaginary part of the x-derivative of the diffraction loads " \
                                            "on body %u for a wave direction of %.1f deg." % \
                                            (body.i_body, np.degrees(self.wave_dir[idir]))

    def write_radiation(self, writer, body, radiation_path="/Radiation"):

        """This function writes the added mass and damping coefficients and the impulse response functions with and without forward speed into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        radiation_path : string, optional
            Path to radiation loads.
        """

        writer.create_group(radiation_path)

        for j in range(self.nb_bodies):

            # Paths.
            radiation_body_motion_path = radiation_path + "/BodyMotion_%u" % j

            dg = writer.create_group(radiation_body_motion_path)
            dg.attrs['Description'] = "Hydrodynamic coefficients for motion of body %u that radiates waves and " \
                                      " generate force on body %u." % (j, body.i_body)

            added_mass_path = radiation_body_motion_path + "/AddedMass"
            dg = writer.create_group(added_mass_path)
            dg.attrs['Description'] = "Added mass coefficients for acceleration of body %u that radiates waves " \
                                      "and generates forces on body %u." % (j, body.i_body)

            if (self._has_x_derivatives):
                added_mass_x_derivative_path = radiation_body_motion_path + "/AddedMassXDerivative"
                dg = writer.create_group(added_mass_x_derivative_path)
                dg.attrs['Description'] = "x-derivative of the added mass coefficients for acceleration of body %u that radiates waves " \
                                          "and generates forces on body %u." % (j, body.i_body)

            radiation_damping_path = radiation_body_motion_path + "/RadiationDamping"
            dg = writer.create_group(radiation_damping_path)
            dg.attrs['Description'] = "Damping coefficients for velocity of body %u that radiates waves " \
                                      "and generates forces on body %u." % (j, body.i_body)

            if (self._has_x_derivatives):
                radiation_damping_x_derivative_path = radiation_body_motion_path + "/RadiationDampingXDerivative"
                dg = writer.create_group(radiation_damping_x_derivative_path)
                dg.attrs['Description'] = "x-derivative of the damping coefficients for velocity of body %u that radiates waves " \
                                          "and generates forces on body %u." % (j, body.i_body)

            if (body.irf is not None):
                irf_path = radiation_body_motion_path + "/ImpulseResponseFunctionK"
                dg = writer.create_group(irf_path)
                dg.attrs['Description'] = "Impulse response functions K due to the velocity of body %u that radiates waves " \
                                          "and generates forces on body %u." % (j, body.i_body)

            if (body.irf_ku is not None):
                irf_ku_path = radiation_body_motion_path + "/ImpulseResponseFunctionKU"
                dg = writer.create_group(irf_ku_path)
                dg.attrs['Description'] = "Impulse response functions Kub proportional to the forward speed without x-derivatives due to the velocity of body %u that radiates waves " \
                                          "and generates forces on body %u." % (j, body.i_body)

            if (body.irf_ku_x_derivative is not None):
                irf_ku_x_derivative_path = radiation_body_motion_path + "/ImpulseResponseFunctionKUXDerivative"
                dg = writer.create_group(irf_ku_x_derivative_path)
                dg.attrs['Description'] = "Impulse response functions Kua proportional to the forward speed with x-derivatives due to the velocity of body %u that radiates waves " \
                                          "and generates forces on body %u." % (j, body.i_body)

            if (body.irf_ku2 is not None):
                irf_ku2_path = radiation_body_motion_path + "/ImpulseResponseFunctionKU2"
                dg = writer.create_group(irf_ku2_path)
                dg.attrs['Description'] = "Impulse response functions Kub proportional to the square of the forward speed due to the velocity of body %u that radiates waves " \
                                          "and generates forces on body %u." % (j, body.i_body)

            if(self.has_VF):
                modal_path = radiation_body_motion_path + "/Modal"
                dg = writer.create_group(modal_path)
                dg.attrs['Description'] = "Poles and residues corresponding to the radiation coefficients due to the motion of body %u " \
                                          "and generating the loads on body %u." % (j, body.i_body)

            # Zero-frequency added mass.
            if (body.Zero_Added_mass is not None):
                dset = writer.create_dataset(radiation_body_motion_path + "/ZeroFreqAddedMass",
                                             data=body.Zero_Added_mass[:, 6 * j:6 * (j + 1)])
                dset.attrs['Description'] = "Zero-frequency added mass matrix that modifies the apparent mass of body %u from " \
                                            "acceleration of body %u." % (body.i_body, j)

            # x-derivative of the zero-frequency added mass.
            if (self._has_x_derivatives):
                if (body.Zero_Added_mass_x_derivative is not None):
                    dset = writer.create_dataset(radiation_body_motion_path + "/ZeroFreqAddedMassXDerivative",
                                                 data=body.Zero_Added_mass_x_derivative[:, 6 * j:6 * (j + 1)])
                    dset.attrs['Description'] = "x-derivative of the zero-frequency added mass matrix that modifies the apparent mass of body %u from " \
                                                "acceleration of body %u." % (body.i_body, j)

            # Infinite-frequency added mass.
            if(body.Inf_Added_mass is not None):
                dset = writer.create_dataset(radiation_body_motion_path + "/InfiniteAddedMass",
                                             data=body.Inf_Added_mass[:, 6 * j:6 * (j + 1)])
                dset.attrs['Description'] = "Infinite-frequency added mass matrix that modifies the apparent mass of body %u from " \
                                            "acceleration of body %u." % (body.i_body, j)

            # x-derivative of the infinite-frequency added mass.
            if (self._has_x_derivatives):
                if (body.Inf_Added_mass_x_derivative is not None):
                    dset = writer.create_dataset(radiation_body_motion_path + "/InfiniteAddedMassXDerivative",
                                                 data=body.Inf_Added_mass_x_derivative[:, 6 * j:6 * (j + 1)])
                    dset.attrs['Description'] = "x-derivative of the infinite-frequency added mass matrix that modifies the apparent mass of body %u from " \
                                                "acceleration of body %u." % (body.i_body, j)

            # Radiation mask.
            dset = writer.create_dataset(radiation_body_motion_path + "/RadiationMask",
                                         data=body.Radiation_mask[:, 6 * j:6 * (j + 1)])
            dset.attrs['Description'] = "Radiation mask of body %u from " \
                                        "acceleration of body %u." % (body.i_body, j)

            for idof in range(0,6):

                # Added mass.
                dset = writer.create_dataset(added_mass_path + "/DOF_%u" % idof, data=body.Added_mass[:, 6*j+idof, :])
                dset.attrs['Unit'] = ""
                dset.attrs['Description'] = "Added mass coefficients for an acceleration of body %u and force on " \
                                            "body %u." % (j, body.i_body)

                # x-derivative of the added mass.
                if (self._has_x_derivatives):
                    dset = writer.create_dataset(added_mass_x_derivative_path + "/DOF_%u" % idof,
                                                 data=body.Added_mass_x_derivative[:, 6 * j + idof, :])
                    dset.attrs['Unit'] = ""
                    dset.attrs['Description'] = "x-derivative of the added mass coefficients for an acceleration of body %u and force on " \
                                                "body %u." % (j, body.i_body)

                # Damping.
                dset = writer.create_dataset(radiation_damping_path + "/DOF_%u" % idof,
                                        data=body.Damping[:, 6*j+idof, :])
                dset.attrs['Unit'] = ""
                dset.attrs['Description'] = "Wave damping coefficients for an acceleration of body %u and force " \
                                            "on body %u." % (j, body.i_body)

                # x-derivative of the damping.
                if (self._has_x_derivatives):
                    dset = writer.create_dataset(radiation_damping_x_derivative_path + "/DOF_%u" % idof,
                                                 data=body.Damping_x_derivative[:, 6 * j + idof, :])
                    dset.attrs['Unit'] = ""
                    dset.attrs['Description'] = "x-derivative of the wave damping coefficients for an acceleration of body %u and force " \
                                                "on body %u." % (j, body.i_body)

                # Impulse response functions without forward speed.
                if(body.irf is not None):
                    dset = writer.create_dataset(irf_path + "/DOF_%u" % idof, data=body.irf[:, 6*j+idof, :])
                    dset.attrs['Description'] = "Impulse response functions K."

                # Impulse response function proportional to the forward speed without x-derivatives.
                if(body.irf_ku is not None):
                    dset = writer.create_dataset(irf_ku_path + "/DOF_%u" % idof, data=body.irf_ku[:, 6*j+idof, :])
                    dset.attrs['Description'] = "Impulse response functions Kub."

                # Impulse response function propotional to the forward speed with x-derivative.
                if (body.irf_ku_x_derivative is not None):
                    dset = writer.create_dataset(irf_ku_x_derivative_path + "/DOF_%u" % idof, data=body.irf_ku_x_derivative[:, 6 * j + idof, :])
                    dset.attrs['Description'] = "Impulse response functions Kua."

                # Impulse response function proportional to the square of the forward speed.
                if (body.irf_ku2 is not None):
                    dset = writer.create_dataset(irf_ku2_path + "/DOF_%u" % idof,
                                                 data=body.irf_ku2[:, 6 * j + idof, :])
                    dset.attrs['Description'] = "Impulse response functions Ku2."

                # Poles and residues.
                if(self.has_VF):
                    dg = writer.create_group(modal_path + "/DOF_%u" % idof)
                    for iforce in range(0, 6):
                        modal_coef_path = modal_path + "/DOF_%u/FORCE_%u" % (idof, iforce)
                        dg = writer.create_group(modal_coef_path)
                        PR = body.poles_residues[36 * j + 6 * idof + iforce]

                        # Real poles and residues.
                        if(PR.nb_real_poles() > 0):
                            dset = writer.create_dataset(modal_coef_path + "/RealPoles", data=PR.real_poles())
                            dset = writer.create_dataset(modal_coef_path + "/RealResidues", data=PR.real_residues())

                        # Complex poles and residues.
                        if(PR.nb_cc_poles() > 0):

                            # Poles.
                            cc_poles_path = modal_coef_path + "/ComplexPoles"
                            dg = writer.create_group(cc_poles_path)
                            dset = writer.create_dataset(cc_poles_path + "/RealCoeff", data=PR.cc_poles().real)
                            dset = writer.create_dataset(cc_poles_path + "/ImagCoeff", data=PR.cc_poles().imag)

                            # Residues.
                            cc_residues_path = modal_coef_path + "/ComplexResidues"
                            dg = writer.create_group(cc_residues_path)
                            dset = writer.create_dataset(cc_residues_path + "/RealCoeff", data=PR.cc_residues().real)
                            dset = writer.create_dataset(cc_residues_path + "/ImagCoeff", data=PR.cc_residues().imag)

    def write_hydrostatic(self, writer, body, hydrostatic_path="/Hydrostatic"):

        """This function writes the hydrostatic stiffness matrix into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        hydrostatic_path : string, optional
            Path to hydrostatic stiffness matrix.
        """

        dg = writer.create_group(hydrostatic_path)

        dset = dg.create_dataset(hydrostatic_path + "/StiffnessMatrix", data=body.hydrostatic.matrix)
        dset.attrs['Description'] = "Hydrostatic stiffness matrix."

    def write_mass_matrix(self, writer, body, inertia_path="/Inertia"):

        """This function writes the mass matrix into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        inertia_path : string, optional
            Path to inertia matrix.
        """

        dg = writer.create_group(inertia_path)

        dset = dg.create_dataset(inertia_path + "/InertiaMatrix", data=body.inertia.matrix)
        dset.attrs['Description'] = "Mass matrix."

    def write_mooring_matrix(self, writer, body, mooring_path="/Mooring"):

        """This function writes the mooring matrix into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        mooring_path : string
            Path to mooring matrix.
        """

        dg = writer.create_group(mooring_path)

        dset = dg.create_dataset(mooring_path + "/MooringMatrix", data=body.mooring)
        dset.attrs['Description'] = "Mooring matrix."

    def write_extra_linear_damping_matrix(self, writer, body, extra_linear_damping_path ="/LinearDampign"):

        """This function writes the extra linear damping matrix matrix into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        extra_linear_damping_path : string
            Path to extra linear damping matrix.
        """

        dg = writer.create_group(extra_linear_damping_path)

        dset = dg.create_dataset(extra_linear_damping_path + "/DampingMatrix", data=body.extra_damping)
        dset.attrs['Description'] = "Extra linear damping matrix."

    def write_RAO(self, writer, body, RAO_path="/RAO"):

        """This function writes the RAO into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        body : BodyDB.
            Body.
        excitation_path : string, optional
            Path to excitation loads.
        """

        writer.create_group(RAO_path)

        for idir in range(0,self.nb_wave_dir):

            wave_dir_path = RAO_path + "/Angle_%u" % idir
            writer.create_group(wave_dir_path)

            dset = writer.create_dataset(wave_dir_path + "/Angle", data=np.degrees(self.wave_dir[idir]))
            dset.attrs['Unit'] = 'deg'
            dset.attrs['Description'] = "Wave direction."

            # Amplitude.
            dset = writer.create_dataset(wave_dir_path + "/Amplitude", data=np.absolute(body.RAO[:, :, idir]))
            dset.attrs['Unit'] = ''
            dset.attrs['Description'] = "Amplitude of the RAO of" \
                                        " body %u for a wave direction of %.1f deg." % \
                                        (body.i_body, np.degrees(self.wave_dir[idir]))

            # Phase.
            dset = writer.create_dataset(wave_dir_path + "/Phase", data=np.angle(body.RAO[:, :, idir], deg=True)) # Degree.
            dset.attrs['Unit'] = 'deg'
            dset.attrs['Description'] = "Phase in deg of the RAO of" \
                                        " body %u for a wave direction of %.1f deg." % \
                                        (body.i_body, np.degrees(self.wave_dir[idir]))

    def write_body(self, writer, body):
        """This function writes the body data into the *.hdb5 file.

        Parameters
        ----------
        writer : string.
            *.hdb5 file.
        body : BodyDB.
            Body.
        """

        body_path = '/Bodies/Body_%u' % body.i_body
        dset = writer.create_group(body_path)

        # Body name.
        if(body.name is not None):
            dset = writer.create_dataset(body_path + "/BodyName", data=body.name)
            dset.attrs['Description'] = "Body name"

        # Index of the body.
        dset = writer.create_dataset(body_path + "/ID", data=body.i_body)
        dset.attrs['Description'] = "Body index"

        # Horizontal position in world.
        if(body.horizontal_position is not None):
            dset_horizontal_position = writer.create_group(body_path + "/HorizontalPosition")
            dset_horizontal_position = writer.create_dataset(body_path + "/HorizontalPosition/x", data=body.horizontal_position[0]) # m.
            dset_horizontal_position = writer.create_dataset(body_path + "/HorizontalPosition/y", data=body.horizontal_position[1]) # m.
            dset_horizontal_position = writer.create_dataset(body_path + "/HorizontalPosition/psi", data=body.horizontal_position[2]) # deg.

        # Computation point in body frame.
        if (body.computation_point is not None):
            dset = writer.create_dataset(body_path + "/ComputationPoint", data=body.computation_point)

        # Wave reference point in body frame.
        if (body.wave_reference_point_in_body_frame is not None):
            dset = writer.create_dataset(body_path + "/WaveReferencePoint", data=body.wave_reference_point_in_body_frame)

        # Masks.
        self.write_mask(writer, body, body_path + "/Mask")

        # Mesh file.
        if(body.mesh is not None):
            self.write_mesh(writer, body, body_path + "/Mesh")

        # Diffraction and Froude-Krylov loads.
        self.write_excitation(writer, body, body_path + "/Excitation")

        # Added mass and damping coefficients, radiation masks and impulse response functions.
        self.write_radiation(writer, body, body_path + "/Radiation")

        # Hydrostatics.
        if body._hydrostatic:
            self.write_hydrostatic(writer, body, body_path + "/Hydrostatic")

        # Mass matrix.
        if body._inertia:
            self.write_mass_matrix(writer, body, body_path + "/Inertia")

        # Mooring matrix.
        if (body._mooring is not None):
            self.write_mooring_matrix(writer, body, body_path + "/Mooring")

        # Extra linear damping matrix.
        if (body._extra_damping is not None):
            self.write_extra_linear_damping_matrix(writer, body, body_path + "/LinearDamping")

        # RAO.
        if(self.has_RAO):
            self.write_RAO(writer, body, body_path + "/RAO")

        # Wave drift coefficients.
        if (body.has_Drift):
            self.write_wave_drift(writer, body, body_path + "/WaveDrift")

    def write_wave_field(self, writer, wave_field_path="/WaveField"):

        """This function writes the wave field data into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        wave_field_path : string, optional
            Path to wave field loads.
        """

        dg = writer.create_group(wave_field_path)

    def write_kochin(self, writer, dg):

        """This method writes the Kochin functions and their angular derivatives into the *.hdb5 file."""

        nbeta = self.nb_dir_kochin # different from self.nb_wave_dir if symmetry of the hdb was done.
        nbodies = self.nb_bodies

        # Diffraction Kochin functions and their derivatives.
        grp_diffraction = dg.require_group("Diffraction")
        for iwave in range(nbeta):

            grp_angle = grp_diffraction.require_group("Angle_"+str(iwave))

            # Wave direction.
            dset = grp_angle.create_dataset("Angle", data=self.wave_dir[iwave] * 180 / np.pi)
            dset.attrs['Unit'] = 'deg'
            dset.attrs['Description'] = "Wave direction"

            # Function.
            grp_diffraction_function = grp_angle.require_group("Function")
            dset = grp_diffraction_function.create_dataset("RealPart",
                                                           data=self.kochin_diffraction[iwave, :, :].transpose().real)
            dset = grp_diffraction_function.create_dataset("ImagPart",
                                                           data=self.kochin_diffraction[iwave, :, :].transpose().imag)

            # Derivative.
            # Nemoh computes the angular derivatives of the Total Kochin functions and not
            # of the elementary ones so they are useless and not exported.
            if(self.solver == "Helios"):
                grp_diffraction_derivative = grp_angle.require_group("Derivative")
                dset = grp_diffraction_derivative.create_dataset("RealPart",
                                                               data=self.kochin_diffraction_derivative[iwave, :, :].transpose().real)
                dset = grp_diffraction_derivative.create_dataset("ImagPart",
                                                               data=self.kochin_diffraction_derivative[iwave, :, :].transpose().imag)

        # Radiation Kochin functions and their derivatives.
        grp_radiation = dg.require_group("Radiation")
        for body in self.bodies:
            for imotion in range(0, 6):

                grp_dof = grp_radiation.require_group("Body_" + str(body.i_body) + "/DOF_" + str(imotion))

                # Function.
                grp_radiation_function = grp_dof.require_group("Function")
                dset = grp_radiation_function.create_dataset("RealPart",
                                                               data=self.kochin_radiation[6 * body.i_body + imotion, :, :].transpose().real)
                dset = grp_radiation_function.create_dataset("ImagPart",
                                                               data=self.kochin_radiation[6 * body.i_body + imotion, :, :].transpose().imag)

                # Derivative.
                # Nemoh computes the angular derivatives of the Total Kochin functions and not
                # of the elementary ones so they are useless and not exported.
                if (self.solver == "Helios"):
                    grp_radiation_derivative = grp_dof.require_group("Derivative")
                    dset = grp_radiation_derivative.create_dataset("RealPart",
                                                                   data=self.kochin_radiation_derivative[6 * body.i_body + imotion, :, :].transpose().real)
                    dset = grp_radiation_derivative.create_dataset("ImagPart",
                                                                   data=self.kochin_radiation_derivative[6 * body.i_body + imotion, :, :].transpose().imag)

    def write_wave_drift(self, writer, body, wave_drift_path="/WaveDrift"):

        """This function writes the wave drift loads into the *.hdb5 file.

        Parameters
        ----------
        Writer : string
            *.hdb5 file.
        wave_drift_path : string, optional
            Path to wave drift loads.
        """

        dg = writer.create_group(wave_drift_path)

        # Loop over the degrees of freedom.
        idof = 0
        for mode in ["surge", "sway", "heave", "roll", "pitch", "yaw"]:
            grp_modes = dg.require_group(mode)

            # Loop over the wave directions.
            for ibeta in range(0, self.nb_wave_dir):
                grp_dir = grp_modes.require_group("angle_%i" % ibeta)

                # Set heading angle.
                dset = grp_dir.create_dataset("angle", data=self.wave_dir[ibeta] * 180 / np.pi)
                dset.attrs['Unit'] = 'deg'
                dset.attrs['Description'] = "Heading angle"

                # Set data.
                dset = grp_dir.create_dataset("data", data=body.Wave_drift_force[idof, :, ibeta])
                dset.attrs['Description'] = "Mean wave drift load coefficients"

            idof = idof + 1

        # Set sym.
        dset = dg.create_dataset("sym_x", data=body.sym_x, dtype=int)
        dset.attrs['Description'] = "Symmetry along x"
        dset = dg.create_dataset('sym_y', data=body.sym_y, dtype=int)
        dset.attrs['Description'] = "Symmetry along y"

        # Kochin function angular step.
        if((self.solver == "Helios" or self.has_kochin) and body.kochin_step is not None):
            if(("/Kochin" in dg) is False):
                dg_kochin = writer.create_group("/Kochin")
            dset = dg_kochin.create_dataset("KochinStep", data=body.kochin_step)
            dset.attrs['Description'] = "Kochin function angular step"

        # Kochin functions.
        if(self.has_kochin):
            if(("/Kochin" in dg) is False):
                dg_kochin = writer.create_group("/Kochin")
            self.write_kochin(writer, dg_kochin)

    def write_VF(self, writer, VF_path = "/VectorFitting"):
        """This function writes the vector fitting parameters into the *.hdb5 file.

        Parameter
        ---------
        Writer : string.
            *.hdb5 file.
        """
        writer.create_group(VF_path)

        dset = writer.create_dataset(VF_path + "/MaxOrder", data=self.max_order)
        dset.attrs['Description'] = "Maximum vector fitting order."

        dset = writer.create_dataset(VF_path + "/Relaxed", data=self.relaxed)
        dset.attrs['Description'] = "Relaxed vector-fitting (true) or original algorithm (false)."

        dset = writer.create_dataset(VF_path + "/Tolerance", data=self.tolerance)
        dset.attrs['Description'] = "Least-square tolerance of the vector fitting algorithm."

    def write_version(self, writer):
            """This function writes the version of the *.hdb5 file.

            Parameter
            ---------
            Writer : string
                *.hdb5 file.
            """

            # Version.
            dset = writer.create_dataset('Version', data= self.version_max)
            dset.attrs['Description'] = "Version of the hdb5 output file."

    def write_commit_hash(self, writer):
            """This function writes the commit hash of the *.hdb5 file.

            Parameter
            ---------
            Writer : string
                *.hdb5 file.
            """

            # Version.
            dset = writer.create_dataset('NormalizedCommitHash', data= self.commit_hash)
            dset.attrs['Description'] = "Tag - Commit hash - Branch - Date."

    def write_numerical_parameters(self, writer, num_param_path = "/ExpertParameters"):
        """This method writes the expert numerical parameters into the *.hdb5 file.

        Parameter
        ---------
        Writer : string.
            *.hdb5 file.
        """
        writer.create_group(num_param_path)

        dset = writer.create_dataset(num_param_path + "/SurfaceIntegrationOrder", data=self.surface_integration_order)
        dset.attrs['Description'] = "Surface integration order."

        dset = writer.create_dataset(num_param_path + "/GreenFunction", data=self.green_function)
        dset.attrs['Description'] = "Green's function."

        dset = writer.create_dataset(num_param_path + "/Crmax", data=self.crmax)

    def write_info(self, input_file):
        """This function writes the hydrodynamic database into a *.hdb5 file."""

        print("\t-------------------------------------------------------")
        print("\t      Information about the hydrodynamic database")
        print("\t-------------------------------------------------------")

        # *.hdb5 file name.
        if (input_file != None):
            print("\nfile name: " + input_file)

        # Environement.
        print("\nGravity acceleration (m/s2): " + str(self.grav))
        print("Water density (kg/m3): " + str(self.rho_water))
        if self.depth == np.inf or self.depth == 0:
            print("Water depth (m): Infinite")
        else:
            print("Water depth (m): " + str(self.depth))

        # Discretization.
        print("\nNumber of wave frequencies: " + str(self.nb_wave_freq))
        print("Wave frequency minimum (rad/s): " + str(self.min_wave_freq))
        print("Wave frequency maximum (rad/s): " + str(self.max_wave_freq))
        print("\nNumber of wave directions: " + str(self.nb_wave_dir))
        print("Wave frequency direction (deg): " + str(self.min_wave_dir))
        print("Wave frequency direction (deg): " + str(self.max_wave_dir))

        # Symmetries.
        if self.bottom_sym is not None:
            if self.bottom_sym == 0:
                print("\nBottom symmetry: No")
            else:
                print("\nBottom symmetry: Yes")
        if self.bottom_sym is not None:
            if self.xoz_sym == 0:
                print("(xOz) symmetry: No")
            else:
                print("(xOz) symmetry: Yes")
        if self.yoz_sym is not None:
            if self.yoz_sym == 0:
                print("(yOz) symmetry: No")
            else:
                print("(yOz) symmetry: Yes")

        # Bodies.
        print("\nNumber of bodies: " + str(self.nb_bodies))
        for body in self.bodies:
            if(body.i_body > 0):
                print("")
            print("    Body " + str(body.i_body + 1))
            print("    Name: " + str(body.name))
            if(body.horizontal_position is not None):
                print("    Horizontal position in world frame (m, m, deg): " + str(body.horizontal_position))
            if(body.computation_point is not None):
                print("    Computation point in body frame (m, m, m): " + str(body.computation_point))
            if body.horizontal_position is not None and body.computation_point is not None:
                # Rotation of angle psi.
                cpsi = np.cos(np.radians(body.horizontal_position[2]))
                spsi = np.sin(np.radians(body.horizontal_position[2]))
                Rot_psi = np.zeros((3, 3))
                Rot_psi[0, 0] = cpsi
                Rot_psi[0, 1] = -spsi
                Rot_psi[1, 0] = spsi
                Rot_psi[1, 1] = cpsi
                Rot_psi[2, 2] = 1.
                world_frame = np.matmul(Rot_psi, body.computation_point)

                # Translation along x and y.
                world_frame[0] += body.horizontal_position[0]
                world_frame[1] += body.horizontal_position[1]

                print("    Computation point in world frame (m, m, m): " + str(world_frame))
            if (body.wave_reference_point_in_body_frame is not None):
                print("    Wave reference point in body frame (m, m): " + str(body.wave_reference_point_in_body_frame))
            if(body.mesh is not None):
                print("    Number of faces: " + str(body.mesh.nb_faces))
                print("    Number of vertices: " + str(body.mesh.nb_vertices))
            if body._inertia:
                print("    Mass matrix: Yes")
                print("        Mass (kg): " + str(body._inertia.mass))
                print("        Ixx (kg.m2) : " + str(body._inertia.I44))
                print("        Iyy (kg.m2) : " + str(body._inertia.I55))
                print("        Izz (kg.m2) : " + str(body._inertia.I66))
                print("        Ixy (kg.m2) : " + str(body._inertia.I45))
                print("        Ixz (kg.m2) : " + str(body._inertia.I46))
                print("        Iyz (kg.m2) : " + str(body._inertia.I56))
            else:
                print("    Mass matrix: No")
            if body._extra_damping is not None:
                print("    Extra damping matrix: Yes")
                if ((body._extra_damping == np.zeros((6, 6))).all() is False):  # Plot
                    for i in range(0, 6):
                        print("        ", end = '')
                        for j in range(0, 6):
                            print(str(body._extra_damping[i, j]) + " ", end='')
                        print("")
            else:
                print("    Extra damping matrix: No")
            if body._hydrostatic:
                print("    Hydrostatic matrix: Yes")
                print("        K33 (N/m) : " + str(body._hydrostatic.k33))
                print("        K44 (N.m) : " + str(body._hydrostatic.k44))
                print("        K55 (N.m) : " + str(body._hydrostatic.k55))
                print("        K34 (N) : " + str(body._hydrostatic.k34))
                print("        K35 (N) : " + str(body._hydrostatic.k35))
                print("        K45 (N.m) : " + str(body._hydrostatic.k45))
            else:
                print("    Hydrostatic matrix: No")
            if body._mooring is not None:
                print("    Mooring matrix: Yes")
                if((body._mooring == np.zeros((6, 6))).all() is False): # Plot
                    for i in range(0, 6):
                        print("        ", end = '')
                        for j in range(0, 6):
                            print(str(body._mooring[i, j]) + " ", end='')
                        print("")
            else:
                print("    Mooring matrix: No")

        # Generalities.
        print("")
        if (self.has_kochin):
            print("Kochin functions: Yes")
            print("    Angular step (deg): " + str(self.angle_kochin[1] * 180 / np.pi))
        else:
            print("Kochin functions: No")
        if self.has_RAO:
            print("RAO: Yes")
        else:
            print("RAO: No")
        for body in self.bodies:
            if (body.has_Drift):
                print("Drift coefficients (%s): Yes" % body.name)
            else:
                print("Drift coefficients (%s): No" % body.name)
        if (self._has_x_derivatives):
            print("x-derivatives: Yes")
        else:
            print("x-derivatives: No")
        if (self.bodies[0].irf is not None):
            print("Impulse response functions: Yes")
            print("    Final time: " + str(self.time[-1]))
            print("    Time step: " + str(self.dt))
        else:
            print("Impulse response functions: No")
        if (self.bodies[0].irf_ku is not None):
            print("Forward-speed impulse response functions: Yes")
        else:
            print("Forward-speed impulse response functions: No")
        if (self.has_VF):
            print("Pole and residues: Yes")
            print("    Max order: " + str(self.max_order))
            print("    Tolerance: " + str(self.tolerance))
            if (self.relaxed):
                print("    Relaxed VF: Yes")
            else:
                print("    Relaxed VF: No")
        else:
            print("Pole and residues: No")

        # Expert parameters.
        print("\nVersion of the hdb5 file: " + str(self.version))
        if (self.commit_hash is not None):
            print("Commit hash: " + str(self.commit_hash))
        if (self.has_expert_parameters):
            print("Surface integration order: " + str(self.surface_integration_order))
            print("Green's function: " + str(self.green_function))
            print("Crmax = " + str(self.crmax))
        print("Solver: " + self.solver)
        print("")