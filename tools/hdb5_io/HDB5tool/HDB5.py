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
"""
    Module to create a hydrodynamic database for FRyDoM.
"""

import os, sys
import numpy as np
from scipy import integrate

from meshmagick.mmio import write_OBJ

from hdb5_io.HDB5tool.discretization_db import DiscretizationDB
from hdb5_io.HDB5tool.DiodoreHDB_writer import DiodoreHDBWriter
import hdb5_io.HDB5tool.plot_db as plot_db
import hdb5_io.HDB5tool.pyHDB as pyHDB
import hdb5_io.HDB5tool.bem_reader as bem_reader
import hdb5_io.HDB5tool.bem_writer as bem_writer
import hdb5_io.HDB5tool.HDB5_reader as HDB5_reader
import hdb5_io.HDB5tool.CSV_writer as CSV_writer
from hdb5_io.HDB5tool.WaveSpectrum import JonswapWaveSpectrum
from hdb5_io.HDB5tool.wave_dispersion_relation import solve_wave_dispersion_relation


class HDB5(object):

    """
        Class HDB5 for dealing with *.h5 files.
    """

    def __init__(self):

        """
            Constructor of the class HDB5.
        """

        # HDB.
        self._pyHDB = pyHDB.pyHDB()

        # Discretization parameters.
        self._discretization = DiscretizationDB()

        # Initialization parameter.
        self._is_initialized = False

        # Writer.
        self.writer = None

        return

    @property
    def pyHDB(self):

        """This function returns the hdb.

        Returns
        -------
        BodyDB
        """

        return self._pyHDB

    @property
    def body(self):

        """This function returns all the bodies.

        Returns
        -------
        BodyDB
        """

        return self._pyHDB.bodies

    @property
    def discretization(self):

        """This function returns the parameters of the discretization.

        Returns
        -------
        DiscretizationDB
        """

        return self._discretization

    def nemoh_reader(self, input_directory='.', nb_faces_by_wavelength=None):

        """This function reads the *.cal file and stores the data.

        Parameters
        ----------
        input_directory : string, optional
            Path to directory of *.cal file.
        nb_faces_by_wavelength : float, optional
            Number of panels per wave length.
        """

        if not os.path.isabs(input_directory):
            input_directory = os.path.abspath(input_directory)

        # Verifying there is the Nemoh.cal file inside input_directory
        nemoh_cal_file = os.path.join(input_directory, 'Nemoh.cal')
        if not os.path.isfile(nemoh_cal_file):
            raise AssertionError('Folder %s seems not to be a Nemoh calculation folder as '
                                 'we did not find Nemoh.cal' % input_directory)

        print("")
        print('========================')
        print('Reading Nemoh results...')
        print('========================')

        if nb_faces_by_wavelength is None:
            nb_faces_by_wavelength = 10

        # Reading *.cal.
        bem_reader.NemohReader(self._pyHDB,cal_file=nemoh_cal_file, test=True, nb_face_by_wave_length=nb_faces_by_wavelength)

        print('-------> Nemoh data successfully loaded from "%s"' % input_directory)
        print("")

    def _initialize(self):

        """This function updates the hydrodynamic database (computation of FK and diffraction loads, impulse response functions, interpolations, etc.)."""

        # Computing Froude-Krylov loads.
        self._pyHDB.Eval_Froude_Krylov_loads()

        # Initialization of the discretization object.
        self._discretization.initialize(self._pyHDB)

        # Time.
        self._pyHDB.time = self.discretization.time
        self._pyHDB.dt = self.discretization.delta_time
        self._pyHDB.nb_time_samples = self.discretization.nb_time_sample

        # Impule response functions for radiation damping.
        self._pyHDB.eval_impulse_response_function()

        # Infinite masses.
        self._pyHDB.eval_infinite_added_mass()

        # Impule response functions proportional to the forward speed without x-derivatives.
        self._pyHDB.eval_impulse_response_function_Ku()

        if(self._pyHDB._has_x_derivatives):

            # Impule response functions proportional to the forward speed with x-derivatives.
            self._pyHDB.eval_impulse_response_function_Ku_x_derivative()

            # Impule response functions proportional to the square of the forward speed.
            self._pyHDB.eval_impulse_response_function_Ku2()

        # Interpolations with respect to the wave directions and the wave frequencies.
        # self._pyHDB.interpolation(self.discretization)

        # Initialization done.
        self._is_initialized = True

    @property
    def omega(self):
        """Frequency array of BEM computations in rad/s

        Returns
        -------
        np.ndarray
        """
        return self._pyHDB.omega

    @property
    def wave_dir(self):
        """Wave direction angles array of BEM computations in radians

        Returns
        -------
        np.ndarray
            angles array in radians.
        """
        return self._pyHDB.wave_dir

    def symmetry_HDB(self):

        """This function symmetrizes the HDB."""

        # Updating the wave directions.
        self._pyHDB._initialize_wave_dir()

    def initRAO(self):

        """This function initializes the RAO of each body."""

        print("")
        print('========================')
        print('Initialize RAO...')
        print('========================')

        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_wave_dir
        for body in self._pyHDB.bodies:

            # RAO.
            body.RAO = np.zeros((6, nw, nbeta), dtype=np.complex64)

            # Hydrostatics.
            if (body.hydrostatic is None):
                #body.activate_hydrostatic()
                print("error : no hydrostatic for body {}. Cannot compute RAO.\n".format(body.name))
                exit(1)
            elif np.any(body.hydrostatic.matrix) == 0.0:
                print("warning : hydrostatic is null for body : {}. RAO computation may be wrong.\n".format(body.name))


            # Inertia.
            if (body.inertia is None):
                #body.activate_inertia()
                print("error : no mass for body {}. Canot compute RAO.\n".format(body.name))
                exit(1)
            elif (body.inertia.mass <= 0.):
                print("warning : mass for body {} is null or negative. RAO computation may be wrong.\n".format(body.name))

    def computeRAO(self):
        """This function computes the RAO for each body."""

        # Initialization of the RAO structure.
        self.initRAO()

        print("")
        print('========================')
        print('Compute RAO...')
        print('========================')

        # Initialization.
        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_wave_dir
        nBodies = self._pyHDB.nb_bodies
        RAO = np.zeros((6 * nBodies, nw, nbeta), dtype = np.complex64)
        Up = np.zeros((6 * nBodies, nw, nbeta), dtype = np.complex64)
        Down = np.zeros((6 * nBodies, 6 * nBodies, nw), dtype = np.complex64)

        ##########################################
        # Down = -w2*(A+M) - jw*B + C and Up = Fe
        ##########################################

        # Loop over the bodies.
        for body in self._pyHDB.bodies:

            # Excitation loads.
            Up[6 * body.i_body:6 * (body.i_body + 1), :, :] = body.Froude_Krylov[:, :, :] + body.Diffraction[:, :, :]

            iw = 0
            for w in self._pyHDB.omega:

                # Mass, hydrostatic and mooring matrices.
                if body.mooring is not None:
                    Down[6 * body.i_body:6 * (body.i_body + 1), 6 * body.i_body:6 * (body.i_body + 1), iw] += \
                        - w * w * body.inertia.matrix + body.hydrostatic.matrix + body.mooring
                else:
                    Down[6 * body.i_body:6 * (body.i_body + 1), 6 * body.i_body:6 * (body.i_body + 1), iw] += \
                        -w * w * body.inertia.matrix + body.hydrostatic.matrix

                # Radiation (added mass and damping) and linear extra damping.
                Down[6 * body.i_body:6 * (body.i_body + 1), :, iw] += -w * w * body.Added_mass[:, :, iw] \
                                                                      - 1j * w * body.Damping[:, :, iw]

                if body.extra_damping is not None:
                    Down[6 * body.i_body:6 * (body.i_body + 1), 6 * body.i_body:6 * (body.i_body + 1), iw] += - 1j * w * body.extra_damping

                iw = iw + 1

        ################
        # RAO = Up/Down
        ################

        # Loop over the wave directions.
        for ibeta in range(0, nbeta):
            iw = 0
            for w in self._pyHDB.omega:
                RAO[:, iw, ibeta] = np.dot(np.linalg.inv(Down[:, :, iw]), Up[:, iw, ibeta])
                iw = iw + 1

        ##########
        # Storage
        ##########

        # Loop over the bodies.
        for body in self._pyHDB.bodies:

            # RAO.
            body.RAO[:, :, :] = RAO[6 * body.i_body:6 * (body.i_body + 1), :, :]

        self._pyHDB.has_RAO = True

        ##################
        # Eigenfrequencies
        ##################

        if(self._pyHDB.has_Eigenfrequencies is False):
            self.compute_eigenfrequencies()

    def compute_eigenfrequencies(self):
        """This function computes the eigenfrequencies."""

        # Only use the diagonal coefficients of the matrices. The coupling between dof are neglected.

        for body in self._pyHDB.bodies:

            # Initialization.
            body.Eigenfrequencies = np.zeros((6), dtype=np.float64)

            for idof in range(0, 6):

                Stiffness = body.hydrostatic.matrix[idof, idof]
                if body.mooring is not None:
                    Stiffness += body.mooring[idof, idof]

                Mass = body.inertia.matrix[idof, idof]
                Mass += body.Inf_Added_mass[idof, idof]

                if np.abs(Stiffness) > 1e-5:
                    body.Eigenfrequencies[idof] = np.sqrt(Stiffness / Mass)
                else:
                    body.Eigenfrequencies[idof] = -1

        self._pyHDB.has_Eigenfrequencies = True

    def initDrift(self):
        """This function checks if everything is ready for the computation of the mean wave drift loads."""

        # Check the Kochin functions were computed.
        if(self._pyHDB.has_kochin is False):
            print("Wave drift loads cannot be computed because Kochin functions were not evaluated.")
            exit()

        # Check the RAO were computed.
        if (self._pyHDB.has_RAO is False):
            nw = self._pyHDB.nb_wave_freq
            nbeta = self._pyHDB.nb_dir_kochin
            for body in self._pyHDB.bodies:
                body.RAO = np.zeros((6, nw, nbeta), dtype = np.complex64)

    def computeTotalKochinFunctions(self, kochin_diffraction, kochin_radiation):
        """This function computes the total Kochin functions."""

        # Parameters.
        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_dir_kochin
        nBodies = self._pyHDB.nb_bodies
        ntheta = self._pyHDB.nb_angle_kochin # Angular discretization of the Kochin functions.

        kochin_total = np.zeros((nbeta, nw, ntheta), dtype=np.complex64)

        # Loop over the wave directions.
        for ibeta in range(0, nbeta):

            # loop over the wave diffractions.
            iw = 0
            for w in self._pyHDB.omega:

                # Diffraction.
                kochin_total[ibeta, iw, :] = kochin_diffraction[ibeta, iw, :]

                # Loop over the bodies.
                for body in self._pyHDB.bodies:

                    # Loop over the dof of the body.
                    for imotion in range(0,6):

                        # Radiation.
                        kochin_total[ibeta, iw, :] += -1j * w * body.RAO[imotion, iw, ibeta] * kochin_radiation[body.i_body*6 + imotion, iw, :]

                iw = iw + 1


        return kochin_total

    def computeDerivativeKochinFunctions(self):
        """This function computes the angular differentiation of the total Kochin functions."""

        from scipy.interpolate import UnivariateSpline

        # Parameters.
        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_dir_kochin
        nBodies = self._pyHDB.nb_bodies
        ntheta = self._pyHDB.nb_angle_kochin  # Angular discretization of the Kochin functions.

        self._pyHDB.kochin_total_derivative = np.zeros((nbeta, nw, ntheta), dtype=np.complex64)

        # Loop over the wave directions.
        for ibeta in range(0, nbeta):

            # loop over the wave diffractions.
            iw = 0
            for w in self._pyHDB.omega:

                # The interpolation is achieved separately over the real and imaginary parts of the complex total Kochin functions, not over the complex values directly (not possible).

                # Real part.
                Real_part_Kochin_tmp = np.zeros((ntheta), dtype=np.float64)

                # spl_real = UnivariateSpline(self._pyHDB.angle_kochin, self._pyHDB.kochin_total[ibeta, iw, :].real, k=4, s = 1) # Construction of the spline for the real part.
                # diff_spl_real = spl_real.derivative() # Construction of the spline for the real part.
                # Real_part_Kochin_tmp = diff_spl_real(self._pyHDB.angle_kochin)
                Real_part_Kochin_tmp = np.gradient(self._pyHDB.kochin_total[ibeta, iw, :].real, self._pyHDB.angle_kochin, edge_order=2)

                # Imaginary part.
                Imag_part_Kochin_tmp = np.zeros((ntheta), dtype=np.float64)

                # spl_imag = UnivariateSpline(self._pyHDB.angle_kochin, self._pyHDB.kochin_total[ibeta, iw, :].imag, k=4, s = 1) # Construction of the spline for the imaginary part.
                # diff_spl_imag = spl_imag.derivative()  # Construction of the spline for the imaginary part.
                # Imag_part_Kochin_tmp = diff_spl_imag(self._pyHDB.angle_kochin)
                Imag_part_Kochin_tmp = np.gradient(self._pyHDB.kochin_total[ibeta, iw, :].imag, self._pyHDB.angle_kochin, edge_order=2)

                # Complex values.
                self._pyHDB.kochin_total_derivative[ibeta, iw, :] = Real_part_Kochin_tmp + 1j * Imag_part_Kochin_tmp

                iw = iw + 1

    def computeDriftForce(self):
        """This function computes the mean wave drift loads from the Kochin functions."""

        # Parameters.
        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_dir_kochin
        nBodies = self._pyHDB.nb_bodies
        ntheta = self._pyHDB.nb_angle_kochin  # Angular discretization of the Kochin functions.

        # Computation of the total Kochin functions.
        self._pyHDB.kochin_total = self.computeTotalKochinFunctions(self._pyHDB.kochin_diffraction, self._pyHDB.kochin_radiation)

        # Computation of the angular differentiation of the total Kochin functions.
        if(self._pyHDB.kochin_diffraction_derivative is None or self._pyHDB.kochin_radiation_derivative is None):
            self.computeDerivativeKochinFunctions()
        else:
            self._pyHDB.kochin_total_derivative = self.computeTotalKochinFunctions(self._pyHDB.kochin_diffraction_derivative, self._pyHDB.kochin_radiation_derivative)

        # Function for seaching the closest value in an array.
        def find_nearest(array, value):
            "Find the closest value in an array."
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        # Searching the good value of theta which matches beta.
        ind_ibeta = np.zeros(nbeta, dtype = np.int64)
        for ibeta in range(0, nbeta):
            # Index of the wave direction (beta) in the Kochin wave direction discretization.
            ind_ibeta[ibeta] = find_nearest(self._pyHDB.angle_kochin, self._pyHDB.wave_dir_kochin[ibeta])

        # Computation of the wave drift force.
        self._pyHDB.bodies[0].Wave_drift_force = np.zeros((6, nw, nbeta), dtype = np.float64)

        # Water depth.
        depth = self._pyHDB.depth
        grav = self._pyHDB.grav
        rho = self._pyHDB.rho_water

        iw = 0
        # Loop over the wave frequencies.
        for w in self._pyHDB.omega:

            k0 = w * w / grav
            k = solve_wave_dispersion_relation(w, depth, grav)

            Coef_1 = -2 * np.pi * rho * w
            if np.isinf(depth):
                Coef_2 = -2 * np.pi * rho * k0 * k0
            else:
                Coef_2 = -2 * np.pi * rho * (k * k0 * k0 / (k * k * depth - k0 * k0 * depth + k0))

            # Loop over the wave directions.
            for ibeta in range(0, nbeta):

                # Drift force along the x-axis.
                Integrand_1 = np.zeros(ntheta)
                for j in range(0, ntheta):
                    Integrand_1[j] = pow(abs(self._pyHDB.kochin_total[ibeta, iw, j]), 2) * np.cos(self._pyHDB.angle_kochin[j])

                if(self._pyHDB.solver == "Helios"): # Different convention between Helios and Nemoh.
                    self._pyHDB.bodies[0].Wave_drift_force[0, iw, ibeta] = Coef_1 * self._pyHDB.kochin_total[ibeta, iw, ind_ibeta[ibeta]].real * np.cos(self._pyHDB.wave_dir_kochin[ibeta]) \
                                                                 + Coef_2 * integrate.trapz(Integrand_1, self._pyHDB.angle_kochin)
                else: # Nemoh.
                    self._pyHDB.bodies[0].Wave_drift_force[0, iw, ibeta] = Coef_1 * self._pyHDB.kochin_total[ibeta, iw, ind_ibeta[ibeta]].imag * np.cos(self._pyHDB.wave_dir_kochin[ibeta]) \
                                                                 + Coef_2 * integrate.trapz(Integrand_1, self._pyHDB.angle_kochin)

                # Drift force along the y-axis.
                Integrand_2 = np.zeros(ntheta)
                for j in range(0, ntheta):
                    Integrand_2[j] = pow(abs(self._pyHDB.kochin_total[ibeta, iw, j]), 2) * np.sin(self._pyHDB.angle_kochin[j])

                if (self._pyHDB.solver == "Helios"):  # Different convention between Helios and Nemoh.
                    self._pyHDB.bodies[0].Wave_drift_force[1, iw, ibeta] = Coef_1 * self._pyHDB.kochin_total[ibeta, iw, ind_ibeta[ibeta]].real * np.sin(self._pyHDB.wave_dir_kochin[ibeta]) \
                                                                 + Coef_2 * integrate.trapz(Integrand_2, self._pyHDB.angle_kochin)
                else: # Nemoh.
                    self._pyHDB.bodies[0].Wave_drift_force[1, iw, ibeta] = Coef_1 * self._pyHDB.kochin_total[ibeta, iw, ind_ibeta[ibeta]].imag * np.sin(self._pyHDB.wave_dir_kochin[ibeta]) \
                                                                 + Coef_2 * integrate.trapz(Integrand_2, self._pyHDB.angle_kochin)

                # Drift moment along the z-axis.
                Integrand_3 = np.zeros(ntheta)
                for j in range(0, ntheta):
                    Integrand_3[j] = (np.conj(self._pyHDB.kochin_total[ibeta, iw, j]) * self._pyHDB.kochin_total_derivative[ibeta, iw, j]).imag

                if (self._pyHDB.solver == "Helios"):  # Different convention between Helios and Nemoh.
                    self._pyHDB.bodies[0].Wave_drift_force[5, iw, ibeta] = Coef_1 / k * self._pyHDB.kochin_total_derivative[ibeta, iw, ind_ibeta[ibeta]].imag \
                                                                 + Coef_2 / k * integrate.trapz(Integrand_3, self._pyHDB.angle_kochin)
                else: # Nemoh.
                    self._pyHDB.bodies[0].Wave_drift_force[5, iw, ibeta] = -Coef_1/k * self._pyHDB.kochin_total_derivative[ibeta, iw, ind_ibeta[ibeta]].real \
                                                                 + Coef_2 / k * integrate.trapz(Integrand_3, self._pyHDB.angle_kochin)

            iw = iw + 1
        self._pyHDB.bodies[0].has_Drift = True

        return

    def computeEnergySpectrumMomentsMonodirectional(self, Hs, Tp, gamma):

        """This method computes the energy spectrum moments."""

        # Parameters.
        omega = self._pyHDB.wave_freq
        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_wave_dir
        nHs = Hs.shape[0]
        order = [0, 1, 2, 4]

        # Initialization.
        for body in self._pyHDB.bodies:
            body.EnergySpectralMoments = np.zeros((6, nbeta, nHs, len(order)))

        # Motion energy spectrum.
        for iHs in range(0, nHs):

            # Wave spectrum.
            wave_spectrum = JonswapWaveSpectrum(Hs[iHs], Tp[iHs], gamma[iHs])
            wave_spectrum_vect = wave_spectrum.eval(omega)

            # Loop over the bodies.
            for body in self._pyHDB.bodies:

                # Loop over the wave directions.
                for ibeta in range(self._pyHDB.nb_wave_dir):

                    # Loop over the dof.
                    for idof in range(0, 6):

                        # Loop order the moment order.
                        for iorder in range(0, len(order)):

                            moment = np.zeros(nw)
                            for iw in range(0, nw):

                                # Convertion of the RAO in roll, pitch and yaw in degrees.
                                RAO = body.RAO[idof, iw, ibeta]
                                if idof >= 3:
                                    RAO *= 180 / np.pi

                                # Computation of the spectral moments.
                                moment[iw] = pow(omega[iw], order[iorder]) * wave_spectrum_vect[iw] * pow(np.abs(RAO), 2)

                            # Integration.
                            body.EnergySpectralMoments[idof, ibeta, iHs, iorder] = np.trapz(moment, x=omega)

        self._pyHDB.has_EnergySpectralMoments = True

    def computeEnergySpectrumMomentsMultiDirectional(self, Hs, Tp, gamma, spreading_factor):

        """This method computes the energy spectrum moments."""

        # Parameters.
        omega = self._pyHDB.wave_freq
        nw = self._pyHDB.nb_wave_freq
        # beta = self._pyHDB.wave_dir
        nbeta = self._pyHDB.nb_wave_dir
        nHs = Hs.shape[0]
        delta_beta = (self._pyHDB.max_wave_dir - self._pyHDB.min_wave_dir) / (nbeta - 1) # In degrees.
        order = [0, 1, 2, 4]

        # Initialization.
        for body in self._pyHDB.bodies:
            body.EnergySpectralMoments = np.zeros((6, nbeta, nHs, len(order)))

        # Motion energy spectrum.
        for iHs in range(0, nHs):

            # Wave spectrum.
            wave_spectrum = JonswapWaveSpectrum(Hs[iHs], Tp[iHs], gamma[iHs])

            # Loop over the bodies.
            for body in self._pyHDB.bodies:

                # Loop over the mean wave directions.
                for i_wave_dir_mean in range(self._pyHDB.nb_wave_dir):

                    mean_wave_dir_rad = self._pyHDB.wave_dir[i_wave_dir_mean] # In radians.

                    # Directional wave spectrum.
                    mean_wave_dir_deg = np.degrees(mean_wave_dir_rad) # In degrees.
                    directional_wave_spectrum = DirectionalWaveSpectrum(wave_spectrum, mean_wave_dir_deg, spreading_factor)

                    # Definition of theta.
                    theta_min = mean_wave_dir_rad - np.pi
                    theta_max = mean_wave_dir_rad + np.pi
                    theta_rad = np.linspace(theta_min, theta_max, nbeta + 1) # +1 because wave_dir = 2 pi was not taken into account.

                    # Computation of the wave spectrum.
                    wave_spectrum_vect = directional_wave_spectrum.eval(omega, theta_rad)

                    # Loop over the dof.
                    for idof in range(0, 6):

                        # Loop order the moment order.
                        for iorder in range(0, len(order)):

                            moment = np.zeros((nbeta + 1, nw)) # +1 because wave_dir = 2 pi was not taken into account.
                            for itheta in range(0, nbeta + 1): # +1 because wave_dir = 2 pi was not taken into account.

                                # Index of beta in wave_dir.
                                if theta_rad[itheta] >= 0:
                                    if mean_wave_dir_rad == 2 * np.pi:
                                        tmp_list = np.abs(self._pyHDB.wave_dir).tolist() # 2pi rad -> 0 rad.
                                    else:
                                        tmp_list = np.abs(self._pyHDB.wave_dir - theta_rad[itheta]).tolist()
                                else:
                                    tmp_list = np.abs(self._pyHDB.wave_dir - (theta_rad[itheta] + 2 * np.pi)).tolist()
                                index = min(tmp_list)
                                ibeta = tmp_list.index(index)

                                for iw in range(0, nw):

                                    # RAO.
                                    RAO = body.RAO[idof, iw, ibeta]

                                    # Convertion of the RAO in roll, pitch and yaw in degrees.
                                    if idof >= 3:
                                        RAO *= 180 / np.pi

                                    moment[itheta, iw] = pow(omega[iw], order[iorder]) * wave_spectrum_vect[itheta, iw] * pow(np.abs(RAO), 2)

                            # Integrations.
                            body.EnergySpectralMoments[idof, i_wave_dir_mean, iHs, iorder] = np.trapz(np.trapz(moment, x=theta_rad, axis=0), x=omega, axis=0)

        self._pyHDB.has_EnergySpectralMoments = True

    def Compute_VF_approximation(self, ibody_force, ibody_motion, iforce, idof, w):
        """This functions computes the vector fitting approximation."""

        # Poles and residues.
        PR = self._pyHDB.bodies[ibody_force].poles_residues[36 * ibody_motion + 6 * idof + iforce]

        # Pole-residue approximation.
        H = np.zeros(w.shape[0], dtype = np.complex64)
        for iw in range(w.shape[0]):
            s = 1j * w[iw]

            # Real poles and residues.
            if (PR.nb_real_poles() > 0):
                real_poles = PR.real_poles()
                real_residues = PR.real_residues()
                for ireal in range(PR.nb_real_poles()):
                    H[iw] += real_residues[ireal] / (s - real_poles[ireal])

            # Complex poles and residues.
            if(PR.nb_cc_poles() > 0):
                cc_poles = PR.cc_poles()
                cc_residues = PR.cc_residues()
                for icc in range(PR.nb_cc_poles()):
                    H[iw] += cc_residues[icc] / (s - cc_poles[icc]) + np.conj(cc_residues[icc]) / (s - np.conj(cc_poles[icc]))

        return H

    def computeIRFfromTimeDomainVF(self, ibody_force, ibody_motion, iforce, idof, time):
        """This functions computes the impulse response function from the time-domain vector fitting approximation."""

        # Poles and residues.
        PR = self._pyHDB.bodies[ibody_force].poles_residues[36 * ibody_motion + 6 * idof + iforce]

        # Impulse response function from the pole-residue approximation.
        K = np.zeros(time.shape[0])
        for it in range(time.shape[0]):
            t = time[it]

            # Real poles and residues.
            if (PR.nb_real_poles() > 0):
                real_poles = PR.real_poles()
                real_residues = PR.real_residues()
                for ireal in range(PR.nb_real_poles()):
                    K[it] += real_residues[ireal] * np.exp(real_poles[ireal] * t)

            # Complex poles and residues.
            if(PR.nb_cc_poles() > 0):
                cc_poles = PR.cc_poles()
                cc_residues = PR.cc_residues()
                for icc in range(PR.nb_cc_poles()):
                    K[it] += 2. * (cc_residues[icc] * np.exp(cc_poles[icc] * t)).real

        return K

    def computeIRFKufromTimeDomainVF(self, ibody_force, ibody_motion, iforce, idof, time):
        """This functions computes the impulse response function Ku from the time-domain vector fitting approximation."""

        # Impulse response function from the pole-residue approximation.
        Ku = np.zeros(time.shape[0])
        if(idof >= 4): # Application of the matrix L.

            # Minus sign of the matrix L.
            epsilon = 1.
            if(idof == 4):
                epsilon = -1.

            # Application of the matrix for selecting the poles and residues.
            if(idof == 4): # Pitch.
                idof_coupling = 2
            else: # Yaw.
                idof_coupling = 1
            PR = self._pyHDB.bodies[ibody_force].poles_residues[36 * ibody_motion + 6 * idof_coupling + iforce]

            for it in range(time.shape[0]):
                t = time[it]

                # Real poles and residues.
                if (PR.nb_real_poles() > 0):
                    real_poles = PR.real_poles()
                    real_residues = PR.real_residues()
                    for ireal in range(PR.nb_real_poles()):
                        Ku[it] += (real_residues[ireal] / real_poles[ireal]) * (1. - np.exp(real_poles[ireal] * t))

                # Complex poles and residues.
                if(PR.nb_cc_poles() > 0):
                    cc_poles = PR.cc_poles()
                    cc_residues = PR.cc_residues()
                    for icc in range(PR.nb_cc_poles()):
                        Ku[it] += 2. * ((cc_residues[icc] / cc_poles[icc]) * (1. - np.exp(cc_poles[icc] * t))).real

            Ku *= epsilon

        return Ku

    def computeIRFfromFrequencyDomainVF(self):
        """This method computes the impulse response functions from the frequency-domain vector fitting approximation."""

        # Initialization of the discretization object.
        self._discretization.initialize(self._pyHDB)

        # Time.
        self._pyHDB.time = self.discretization.time
        self._pyHDB.dt = self.discretization.delta_time
        self._pyHDB.nb_time_samples = self.discretization.nb_time_sample

        # Wave frequency range.
        wmax = 10 * self._pyHDB.max_wave_freq
        w = np.linspace(0, wmax, 20 * self._pyHDB.nb_wave_freq, dtype=np.float64)

        # Computation.
        wt = np.einsum('i, j -> ij', w, self._pyHDB.time)  # w*t.
        cwt = np.cos(wt)  # cos(w*t).

        for body in self._pyHDB.bodies:

            irf_data = np.empty(0, dtype=np.float64)

            # Damping from the vector fitting approximation.
            Damping = np.zeros((6, 6 * self._pyHDB.nb_bodies, len(w)), dtype=np.float64)
            for ibody_motion in range(0, self._pyHDB.nb_bodies):
                for iforce in range(0, 6):
                    for idof in range(0, 6):
                        Damping[iforce, 6 * ibody_motion + idof, :] = self.Compute_VF_approximation(body.i_body, ibody_motion, iforce, idof, w).real

            # Computation of the IRF.
            ca = np.einsum('ijk, ij -> ijk', Damping, body._flags) # B(w).
            kernel = np.einsum('ijk, kl -> ijkl', ca, cwt) # B(w)*cos(wt).
            irf_data = (2 / np.pi) * np.trapz(kernel, x=w, axis=2) # (2/pi) * Int(B(w)*cos(wt), dw).

            body.irf = irf_data

    def computeIRFKufromFrequencyDomainVF(self):
        """This method computes the impulse response functions Ku from the vector fitting approximation."""

        # Wave frequency range.
        wmax = 10 * self._pyHDB.max_wave_freq
        w = np.linspace(1e-6, wmax, 20 * self._pyHDB.nb_wave_freq, dtype=np.float64)

        # Computation.
        wt = np.einsum('i, j -> ij', w, self._pyHDB.time)  # w*t.
        cwt = np.cos(wt)  # cos(w*t).

        for body in self._pyHDB.bodies:

            irf_data = np.empty(0, dtype=np.float64)

            # Added mass minus the infinite-frequency added mass from the vector fitting approximation.
            Added_mass = np.zeros((6, 6 * self._pyHDB.nb_bodies, len(w)), dtype=np.float64)
            for ibody_motion in range(0, self._pyHDB.nb_bodies):
                for iforce in range(0, 6):
                    for idof in range(0, 6):
                        PR = self._pyHDB.bodies[body.i_body].poles_residues[36 * ibody_motion + 6 * idof + iforce]
                        Added_mass[iforce, 6 * ibody_motion + idof, :] = -self.Compute_VF_approximation(body.i_body, ibody_motion, iforce, idof, w).imag / w

            # [A(inf) - A(w)]*L.
            Added_mass[:, 4, :] = -Added_mass[:, 2, :]
            Added_mass[:, 5, :] = Added_mass[:, 1, :]
            Added_mass[:, 0, :] = 0.
            Added_mass[:, 1, :] = 0.
            Added_mass[:, 2, :] = 0.
            Added_mass[:, 3, :] = 0.

            # Computation of the IRF.
            kernel = np.einsum('ijk, kl -> ijkl', Added_mass, cwt) # int([A(inf) - A(w)]*L*cos(wt), dw).
            irf_data = (2. / np.pi) * np.trapz(kernel, x=w, axis=2) # (2/pi) * int([A(inf) - A(w)]*L*cos(wt), dw).

            body.irf_ku = irf_data

    def ComparaisonConvolutionK(self, ibody_force, ibody_motion, iforce, idof):

        """This method compares the computation of the convolution of the IRF and a sine velocity by a direct integration and the recursive convolution."""

        # Body.
        body = self._pyHDB.bodies[ibody_force]

        # Time.
        time = self._pyHDB.time
        dt = time[1] - time[0]
        ntime = self._pyHDB.time.shape[0]

        # The velocity is prescribed.
        velocity = np.zeros(ntime)
        for it in range(0, ntime):
            velocity[it] = np.sin(time[it]) # v(t) = sin(wt) with w = 1 rad/s.

        # Direct evaluation.
        mu_direct = np.zeros(ntime)
        IRF = body.irf[iforce, 6 * ibody_motion + idof, :]
        for it in range(0, ntime):

            # Integration over [0, t].
            kernel = np.zeros(it + 1)
            tau = time[0:it + 1]

            for it_convolution in range(0, tau.shape[0]):
                kernel[it_convolution] = IRF[it - it_convolution] * velocity[it_convolution] # K(t-tau) * v(tau).
            mu_direct[it] = np.trapz(kernel, x=tau, axis=0) # int_0^t(K(tau) * v(t - tau), dtau)

        # Recursive convolution.
        mu_recursive = np.zeros(time.shape[0])
        PR = body.poles_residues[36 * ibody_motion + 6 * idof + iforce]
        n_poles = PR.nb_real_poles() + PR.nb_cc_poles()
        alpha = np.zeros(n_poles, dtype = np.complex64)
        beta = np.zeros(n_poles, dtype = np.complex64)
        gamma = np.zeros(n_poles, dtype = np.complex64)
        state = np.zeros(n_poles, dtype = np.complex64)

        # Real poles and residues.
        ipole = 0
        if (PR.nb_real_poles() > 0):
            real_poles = PR.real_poles()
            for ireal in range(PR.nb_real_poles()):
                p = real_poles[ireal]
                alpha[ipole] = np.exp(p * dt)
                beta[ipole] = (1. + (p * dt - 1.) * np.exp(p * dt)) / (p * p * dt)
                gamma[ipole] = (np.exp(p * dt) - (p * dt + 1.)) / (p * p * dt)
                ipole += 1

        # Complex poles and residues.
        if (PR.nb_cc_poles() > 0):
            cc_poles = PR.cc_poles()
            for icc in range(PR.nb_cc_poles()):
                p = cc_poles[icc]
                alpha[ipole] = np.exp(p * dt)
                beta[ipole] = (1. + (p * dt - 1.) * np.exp(p * dt)) / (p * p * dt)
                gamma[ipole] = (np.exp(p * dt) - (p * dt + 1.)) / (p * p * dt)
                ipole += 1

        for it in range(0, ntime):

            for ipole in range(0, n_poles):
                state[ipole] = alpha[ipole] * state[ipole] + gamma[ipole] * velocity[it]
                if(it != 0):
                    state[ipole] += beta[ipole] * velocity[it - 1]

            # Mu.
            ipole = 0
            if (PR.nb_real_poles() > 0):
                real_residues = PR.real_residues()
                for ireal in range(PR.nb_real_poles()):
                    mu_recursive[it] += (real_residues[ireal] * state[ipole]).real
                    ipole += 1
            if (PR.nb_cc_poles() > 0):
                cc_residues = PR.cc_residues()
                for icc in range(PR.nb_cc_poles()):
                    mu_recursive[it] += 2. * (cc_residues[icc] * state[ipole]).real
                    ipole += 1

        import matplotlib.pyplot as plt
        plt.plot(time, mu_direct, 'r-', label = "Direct")
        plt.plot(time, mu_recursive, 'b--', label="Recursive")
        plt.legend()
        plt.show()

    def ComparaisonConvolutionKu(self, ibody_force, ibody_motion, iforce, idof):

        """This method compares the computation of the convolution of the IRF Ku and a sine velocity by a direct integration and the recursive convolution."""

        # Body.
        body = self._pyHDB.bodies[ibody_force]

        # Time.
        time = self._pyHDB.time
        dt = time[1] - time[0]
        ntime = self._pyHDB.time.shape[0]

        # The position and velocity are prescribed.
        position = np.zeros(ntime)
        velocity = np.zeros(ntime)
        for it in range(0, ntime):
            position[it] = -np.cos(time[it]) # x(t) = -cos(wt) with w = 1 rad/s.
            velocity[it] = np.sin(time[it]) # v(t) = sin(wt) with w = 1 rad/s.

        # Direct evaluation.
        mu_direct = np.zeros(ntime)
        IRF = body.irf_ku[iforce, 6 * ibody_motion + idof, :]
        for it in range(0, ntime):

            # Integration over [0, t].
            kernel = np.zeros(it + 1)
            tau = time[0:it + 1]

            for it_convolution in range(0, tau.shape[0]):
                kernel[it_convolution] = IRF[it - it_convolution] * velocity[it_convolution] # K(t-tau) * v(tau).
            mu_direct[it] = np.trapz(kernel, x=tau, axis=0) # int_0^t(K(tau) * v(t - tau), dtau)

        # Recursive convolution.
        mu_recursive = np.zeros(time.shape[0])
        if (idof >= 4):  # Application of the matrix L.

            # Minus sign of the matrix L.
            epsilon = 1.
            if (idof == 4):
                epsilon = -1.

            # Application of the matrix for selecting the poles and residues.
            if (idof == 4):  # Pitch.
                idof_coupling = 2
            else:  # Yaw.
                idof_coupling = 1
            PR = body.poles_residues[36 * ibody_motion + 6 * idof_coupling + iforce]
            n_poles = PR.nb_real_poles() + PR.nb_cc_poles()
            alpha = np.zeros(n_poles, dtype = np.complex64)
            beta = np.zeros(n_poles, dtype = np.complex64)
            gamma = np.zeros(n_poles, dtype = np.complex64)
            state = np.zeros(n_poles, dtype = np.complex64)

            # Real poles and residues.
            ipole = 0
            if (PR.nb_real_poles() > 0):
                real_poles = PR.real_poles()
                for ireal in range(PR.nb_real_poles()):
                    p = real_poles[ireal]
                    alpha[ipole] = np.exp(p * dt)
                    beta[ipole] = (1. + (p * dt - 1.) * np.exp(p * dt)) / (p * p * dt)
                    gamma[ipole] = (np.exp(p * dt) - (p * dt + 1.)) / (p * p * dt)
                    ipole += 1

            # Complex poles and residues.
            if (PR.nb_cc_poles() > 0):
                cc_poles = PR.cc_poles()
                for icc in range(PR.nb_cc_poles()):
                    p = cc_poles[icc]
                    alpha[ipole] = np.exp(p * dt)
                    beta[ipole] = (1. + (p * dt - 1.) * np.exp(p * dt)) / (p * p * dt)
                    gamma[ipole] = (np.exp(p * dt) - (p * dt + 1.)) / (p * p * dt)
                    ipole += 1

            for it in range(0, ntime):

                for ipole in range(0, n_poles):
                    state[ipole] = alpha[ipole] * state[ipole] + gamma[ipole] * velocity[it]
                    if(it != 0):
                        state[ipole] += beta[ipole] * velocity[it - 1]

                # Mu.
                ipole = 0
                if (PR.nb_real_poles() > 0):
                    real_poles = PR.real_poles()
                    real_residues = PR.real_residues()
                    for ireal in range(PR.nb_real_poles()):
                        mu_recursive[it] += (real_residues[ireal] / real_poles[ireal]) * (position[it] - position[0]) - ((real_residues[ireal] / real_poles[ireal]) * state[ipole]).real
                        ipole += 1
                if (PR.nb_cc_poles() > 0):
                    cc_poles = PR.cc_poles()
                    cc_residues = PR.cc_residues()
                    for icc in range(PR.nb_cc_poles()):
                        mu_recursive[it] += 2. * (cc_residues[icc] / cc_poles[icc]).real * (position[it] - position[0]) - 2. * ((cc_residues[icc] / cc_poles[icc]) * state[ipole]).real
                        ipole += 1

            mu_recursive *= epsilon

        import matplotlib.pyplot as plt
        plt.plot(time, mu_direct, 'r-', label = "Direct")
        plt.plot(time, mu_recursive, 'b--', label="Recursive")
        plt.legend()
        plt.show()

    def Plot_RAO(self, ibody, iforce, iwave = 0):
        """This functions plots the RAOs."""

        # Data.
        data = self._pyHDB.bodies[ibody].RAO[iforce, :, iwave]

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Eigenfrequency.
        eigenfreq = self._pyHDB.bodies[ibody].Eigenfrequencies[iforce]

        # Plot.
        plot_db.plot_RAO_fig(data, self._pyHDB.wave_freq, ibody, iforce, beta, eigenfreq)

    def Plot_Diffraction(self, ibody, iforce, iwave = 0):
        """This functions plots the diffraction loads."""

        # Data.
        data = self._pyHDB.bodies[ibody].Diffraction[iforce, :, iwave]

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Plot.
        plot_db.plot_loads(data, self._pyHDB.wave_freq, 0, ibody, iforce, beta, False)

    def Plot_Diffraction_x_derivative(self, ibody, iforce, iwave = 0):
        """This functions plots the x-derivative of the diffraction loads."""

        # Data.
        data = self._pyHDB.bodies[ibody].Diffraction_x_derivative[iforce, :, iwave]

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Plot.
        plot_db.plot_loads(data, self._pyHDB.wave_freq, 0, ibody, iforce, beta, True)

    def Plot_Froude_Krylov(self, ibody, iforce, iwave = 0, **kwargs):
        """This functions plots the Froude-Krylov loads."""

        # Data.
        data = self._pyHDB.bodies[ibody].Froude_Krylov[iforce, :, iwave]

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Plots.
        plot_db.plot_loads(data, self._pyHDB.wave_freq, 1, ibody, iforce, beta, False, **kwargs)

    def Plot_Froude_Krylov_x_derivative(self, ibody, iforce, iwave = 0, **kwargs):
        """This functions plots the x-derivative of the Froude-Krylov loads."""

        # Data.
        data = self._pyHDB.bodies[ibody].Froude_Krylov_x_derivative[iforce, :, iwave]

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Plots.
        plot_db.plot_loads(data, self._pyHDB.wave_freq, 1, ibody, iforce, beta, True, **kwargs)

    def Plot_Excitation(self, ibody, iforce, iwave = 0, **kwargs):
        """This functions plots the excitation loads."""

        # Data.
        data = self._pyHDB.bodies[ibody].Diffraction[iforce, :, iwave] + self._pyHDB.bodies[ibody].Froude_Krylov[iforce, :, iwave]

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Plots.
        plot_db.plot_loads(data, self._pyHDB.wave_freq, 2, ibody, iforce, beta, False, **kwargs)

    def Plot_Excitation_x_derivative(self, ibody, iforce, iwave = 0, **kwargs):
        """This functions plots the x-derivative of the excitation loads."""

        # Data.
        data = self._pyHDB.bodies[ibody].Diffraction_x_derivative[iforce, :, iwave] + self._pyHDB.bodies[ibody].Froude_Krylov_x_derivative[iforce, :, iwave]

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Plots.
        plot_db.plot_loads(data, self._pyHDB.wave_freq, 2, ibody, iforce, beta, True, **kwargs)

    def Plot_Radiation_coeff(self, ibody_force, iforce, ibody_motion, idof):
        """This functions plots the added mass and damping coefficients."""

        # Data.
        body_force = self._pyHDB.bodies[ibody_force]

        # If w_min is too far from w = 0, w = 0 is not displayed.
        display_zero_frequency = False
        if body_force.Zero_Added_mass is not None and self._pyHDB.wave_freq[0] < 1.:
            display_zero_frequency = True

        if display_zero_frequency is True:
            data = np.zeros((self._pyHDB.nb_wave_freq + 2, 2), dtype=np.float64)  # 2 for added mass and damping coefficients, +2 for both the infinite and zero-frequency added mass.

            # Added mass.
            data[0, 0] = body_force.Zero_Added_mass[iforce, 6 * ibody_motion + idof]
            data[1:self._pyHDB.nb_wave_freq+1, 0] = body_force.Added_mass[iforce, 6 * ibody_motion + idof, :]
            data[self._pyHDB.nb_wave_freq+1, 0] = body_force.Inf_Added_mass[iforce, 6 * ibody_motion + idof]

            # Damping.
            data[0, 1] = 0.
            data[1:self._pyHDB.nb_wave_freq+1, 1] = body_force.Damping[iforce, 6 * ibody_motion + idof, :]

            # Wave frequency.
            w = np.zeros((self._pyHDB.wave_freq.shape[0] + 1))
            w[0] = 0
            w[1:] = self._pyHDB.wave_freq
        else:
            data = np.zeros((self._pyHDB.nb_wave_freq+1,2), dtype = np.float64) # 2 for added mass and damping coefficients, +1 for the infinite-frequency added mass.

            # Added mass.
            data[0:self._pyHDB.nb_wave_freq, 0] = body_force.Added_mass[iforce, 6 * ibody_motion + idof, :]
            data[self._pyHDB.nb_wave_freq, 0] = body_force.Inf_Added_mass[iforce, 6 * ibody_motion + idof]

            # Damping.
            data[0:self._pyHDB.nb_wave_freq, 1] = body_force.Damping[iforce, 6 * ibody_motion + idof, :]

            # Wave frequency.
            w = self._pyHDB.wave_freq

        # Plots.
        plot_db.plot_AB(data, w, ibody_force, iforce, ibody_motion, idof, False)

    def Plot_Radiation_coeff_x_derivative(self, ibody_force, iforce, ibody_motion, idof):
        """This functions plots the x-derivative of the added mass and damping coefficients."""

        if(self._pyHDB._has_x_derivatives):

            # Data.
            body_force = self._pyHDB.bodies[ibody_force]
            data = np.zeros((self._pyHDB.nb_wave_freq + 2, 2), dtype=np.float64)  # 2 for added mass and damping coefficients, +2 for both the infinite and zero-frequency added mass.

            # x-derivative of the added mass.
            data[0, 0] = body_force.Zero_Added_mass_x_derivative[iforce, 6 * ibody_motion + idof]
            data[1:self._pyHDB.nb_wave_freq+1, 0] = body_force.Added_mass_x_derivative[iforce, 6 * ibody_motion + idof, :]
            data[self._pyHDB.nb_wave_freq+1, 0] = body_force.Inf_Added_mass_x_derivative[iforce, 6 * ibody_motion + idof]

            # x-derivative of the Damping.
            data[0, 1] = 0.
            data[1:self._pyHDB.nb_wave_freq+1, 1] = body_force.Damping_x_derivative[iforce, 6 * ibody_motion + idof, :]

            # Wave frequency.
            w = np.zeros((self._pyHDB.wave_freq.shape[0] + 1))
            w[0] = 0
            w[1:] = self._pyHDB.wave_freq

            # Plots.
            plot_db.plot_AB(data, w, ibody_force, iforce, ibody_motion, idof, True)

    def Plot_IRF(self, ibody_force, iforce, ibody_motion, idof):
        """This function plots the impulse response functions without forward speed."""

        # Data.
        data = self._pyHDB.bodies[ibody_force].irf[iforce, 6 * ibody_motion + idof, :]

        # Time.
        time = self._pyHDB.time

        # Plots.
        plot_db.plot_irf(data, time, 0, ibody_force, iforce, ibody_motion, idof)

    def Plot_IRF_speed(self, ibody_force, iforce, ibody_motion, idof):
        """This function plots the impulse response functions with forward speed."""

        # Data.
        data = self._pyHDB.bodies[ibody_force].irf_ku[iforce, 6 * ibody_motion + idof , :]

        # Time.
        time = self._pyHDB.time

        # Plots.
        plot_db.plot_irf(data, time, 2, ibody_force, iforce, ibody_motion, idof)

        if(self._pyHDB._has_x_derivatives):

            if(self._pyHDB.bodies[ibody_force].irf_ku_x_derivative is not None):
                # Data.
                data = self._pyHDB.bodies[ibody_force].irf_ku_x_derivative[iforce, 6 * ibody_motion + idof, :]

                # Plots.
                plot_db.plot_irf(data, time, 1, ibody_force, iforce, ibody_motion, idof)

            if(self._pyHDB.bodies[ibody_force].irf_ku2 is not None):
                # Data.
                data = self._pyHDB.bodies[ibody_force].irf_ku2[iforce, 6 * ibody_motion + idof, :]

                # Plots.
                plot_db.plot_irf(data, time, 3, ibody_force, iforce, ibody_motion, idof)

    def Plot_Mesh(self, ibody = -1):
        """This function plots a mesh."""

        if(self._pyHDB.bodies[ibody].mesh is not None):

            # Data.
            mesh = self._pyHDB.bodies[ibody].mesh

            # From the body frame to the global frame.
            if(self._pyHDB.bodies[ibody].horizontal_position is not None):
                horizontal_position = self._pyHDB.bodies[ibody].horizontal_position
                mesh.rotate([0., 0., horizontal_position[2]])
                mesh.translate([horizontal_position[0], horizontal_position[1], 0.])

            # Plot.
            plot_db.Meshmagick_viewer(mesh)

    def Plot_Meshes(self):
        """This function plots all meshes."""

        if (self._pyHDB.bodies[0].mesh is not None): # If one mesh is present, other ones should also be.

            # Data.
            mesh = self._pyHDB.bodies[0].mesh
            if (self._pyHDB.bodies[0].horizontal_position is not None):
                horizontal_position = self._pyHDB.bodies[0].horizontal_position
                mesh.rotate([0., 0., horizontal_position[2]])
                mesh.translate([horizontal_position[0], horizontal_position[1], 0.])

            MultibodyMesh = mesh # Initialization by using the first body which always exists because they are several bodies.
            for id in range(1, self._pyHDB.nb_bodies): # Loop over all bodies except the first one.

                mesh = self._pyHDB.bodies[id].mesh

                # From the body frame to the global frame.
                if (self._pyHDB.bodies[id].horizontal_position is not None):
                    horizontal_position = self._pyHDB.bodies[id].horizontal_position
                    mesh.rotate([0., 0., horizontal_position[2]])
                    mesh.translate([horizontal_position[0], horizontal_position[1], 0.])

                # Merging.
                MultibodyMesh += mesh

            # Plot.
            plot_db.Meshmagick_viewer(MultibodyMesh)

    def Plot_Kochin_Elem(self, DifforRad, ibody, iforce, iwave, iw, **kwargs):
        """This functions plots the elementary (diffraction(0) or radiation (1)) kochin functions."""

        # Data.
        if DifforRad == 0: # Diffraction.
            data = self._pyHDB.kochin_diffraction[iwave, iw, :]
        else: # Radiation.
            data = self._pyHDB.kochin_radiation[6*ibody + iforce, iw, :]

        # Angular discretization.
        Angle = self._pyHDB.angle_kochin

        # Wave direction.
        beta = self._pyHDB.wave_dir_kochin[iwave]

        # Wave frequency.
        w = self._pyHDB.omega[iw]

        # Plots.
        plot_db.plotKochinElem(data, Angle, DifforRad, w, ibody, iforce, beta, **kwargs)

    def Plot_Kochin(self, iwave, iw, **kwargs):
        """This functions plots the total kochin functions."""

        # Data.
        data = self._pyHDB.kochin_total[iwave,iw,:]

        # Angular discretization.
        Angle = self._pyHDB.angle_kochin

        # Wave direction.
        beta = self._pyHDB.wave_dir_kochin[iwave]

        # Wave frequency.
        w = self._pyHDB.omega[iw]

        # Plots.
        plot_db.plotKochin(data, Angle, w, beta, 0, **kwargs)

    def Plot_Kochin_derive(self, iwave, iw, **kwargs):
        """This functions plots the total angular derivative kochin functions."""

        # Data.
        data = self._pyHDB.kochin_total_derivative[iwave,iw,:]

        # Angular discretization.
        Angle = self._pyHDB.angle_kochin

        # Wave direction.
        beta = self._pyHDB.wave_dir_kochin[iwave]

        # Wave frequency.
        w = self._pyHDB.omega[iw]

        # Plots.
        plot_db.plotKochin(data, Angle, w, beta, 1, **kwargs)

    def Plot_drift(self, ibody, iwave, iforce):
        """This functions plots the drift loads."""

        # Data.
        data = self._pyHDB.bodies[ibody].Wave_drift_force[iforce, :, iwave]

        # Wave frequencies.
        w = self._pyHDB.wave_freq

        # Wave direction.
        beta = self._pyHDB.wave_dir[iwave]

        # Plots.
        plot_db.plotDrift(data, w, beta, iforce)

    def Plot_vector_fitting(self, ibody_force, iforce, ibody_motion, idof):
        """This functions plots the vector fitting approximation and the frequency-domain IRF per coefficient."""

        # Vector fitting approximation for K.
        nw = self._pyHDB.nb_wave_freq
        w = np.zeros((nw + 1), dtype = np.float64)
        data = np.zeros((nw + 1, 2), dtype=np.complex64) # 2 for the IRF and the vector fitting approximation, +1 for the null frequency.

        # Added mass and damping coefficients.
        AddedMass = self._pyHDB.bodies[ibody_force].Added_mass[iforce, 6 * ibody_motion + idof, :]
        InfAddedMass = self._pyHDB.bodies[ibody_force].Inf_Added_mass[iforce, 6 * ibody_motion + idof]
        Damping = self._pyHDB.bodies[ibody_force].Damping[iforce, 6 * ibody_motion + idof, :]

        # Frequency-domain impulse response function.
        data[0, 0] = 0.
        data[1:nw+1, 0] = Damping + 1j * self._pyHDB.wave_freq * (AddedMass - InfAddedMass)

        # Vector fitting approximation.
        w[0] = 0
        w[1:nw + 1] = self._pyHDB.wave_freq
        data[:, 1] = self.Compute_VF_approximation(ibody_force, ibody_motion, iforce, idof, w)

        # Plots.
        plot_db.plot_VF(data, w, ibody_force, iforce, ibody_motion, idof, False)

        # Vector fitting approximation for KU.
        data = np.zeros((nw, 2), dtype=np.complex64) # 2 for the IRF and the vector fitting approximation, +1 for the null frequency.

        # Added mass and damping coefficients.
        AddedMass = self._pyHDB.bodies[ibody_force].Added_mass[iforce, 6 * ibody_motion + idof, :]
        InfAddedMass = self._pyHDB.bodies[ibody_force].Inf_Added_mass[iforce, 6 * ibody_motion + idof]
        Damping = self._pyHDB.bodies[ibody_force].Damping[iforce, 6 * ibody_motion + idof, :]

        # Frequency-domain impulse response function.
        data[:, 0] = (1j / self._pyHDB.wave_freq) * Damping - (AddedMass - InfAddedMass)

        # Vector fitting approximation.
        data[:, 1] = -(1 / (1j * self._pyHDB.wave_freq)) * self.Compute_VF_approximation(ibody_force, ibody_motion, iforce, idof, self._pyHDB.wave_freq)

        # Plots
        plot_db.plot_VF(data, self._pyHDB.wave_freq, ibody_force, iforce, ibody_motion, idof, True)

    def Plot_vector_fitting_array(self):
        """This functions plots the vector fitting approximation and the frequency-domain IRF per body."""

        # Wave frequencies.
        nw = self._pyHDB.nb_wave_freq
        w = np.zeros((nw + 1), dtype=np.float64)
        w[0] = 0
        w[1:nw + 1] = self._pyHDB.wave_freq

        # Data.
        for ibody_force in range(0, self._pyHDB.nb_bodies):
            for ibody_motion in range(0, self._pyHDB.nb_bodies):

                # VF approximation of the K.
                data = np.zeros((6, 6, nw + 1, 2), dtype=np.complex64)# 2 for the IRF and the vector fitting approximation, +1 for the null frequency.

                for iforce in range(0, 6):
                    for idof in range(0, 6):

                        # Added mass and damping coefficients.
                        AddedMass = self._pyHDB.bodies[ibody_force].Added_mass[iforce, 6 * ibody_motion + idof, :]
                        InfAddedMass = self._pyHDB.bodies[ibody_force].Inf_Added_mass[iforce, 6 * ibody_motion + idof]
                        Damping = self._pyHDB.bodies[ibody_force].Damping[iforce, 6 * ibody_motion + idof, :]

                        # Frequency-domain impulse response function.
                        data[iforce, idof, 0, 0] = 0.
                        data[iforce, idof, 1:nw+1, 0] = Damping + 1j * self._pyHDB.wave_freq * (AddedMass - InfAddedMass)

                        # Vector fitting approximation.
                        data[iforce, idof, :, 1] = self.Compute_VF_approximation(ibody_force, ibody_motion, iforce, idof, w)

                # Plots.
                plot_db.plot_VF_array(data, w, ibody_force, ibody_motion, False)

                # VF approximation of the KU.
                data = np.zeros((6, 6, nw, 2), dtype=np.complex64) # 2 for the IRF and the vector fitting approximation.

                for iforce in range(0, 6):
                    for idof in range(0, 6):
                        # Added mass and damping coefficients.
                        AddedMass = self._pyHDB.bodies[ibody_force].Added_mass[iforce, 6 * ibody_motion + idof, :]
                        InfAddedMass = self._pyHDB.bodies[ibody_force].Inf_Added_mass[iforce, 6 * ibody_motion + idof]
                        Damping = self._pyHDB.bodies[ibody_force].Damping[iforce, 6 * ibody_motion + idof, :]

                        # Frequency-domain impulse response function.
                        data[iforce, idof, :, 0] = (1j / self._pyHDB.wave_freq) * Damping - (AddedMass - InfAddedMass)

                        # Vector fitting approximation.
                        data[iforce, idof, :, 1] = -(1 / (1j * self._pyHDB.wave_freq)) \
                                                   * self.Compute_VF_approximation(ibody_force, ibody_motion, iforce, idof, self._pyHDB.wave_freq)

                # Plots.
                plot_db.plot_VF_array(data, self._pyHDB.wave_freq, ibody_force, ibody_motion, True)

        return

    def Write_Mesh(self, ibody=-1):
        """This method writes a mesh."""

        if self._pyHDB.bodies[ibody].mesh is not None:

            # Data.
            body = self._pyHDB.bodies[ibody]
            mesh = body.mesh

            # From the body frame to the global frame.
            if self._pyHDB.bodies[ibody].horizontal_position is not None:
                horizontal_position = self._pyHDB.bodies[ibody].horizontal_position
                mesh.rotate([0., 0., horizontal_position[2]])
                mesh.translate([horizontal_position[0], horizontal_position[1], 0.])

            # Write.
            if body.name is not None:
                write_OBJ(body.name + ".obj", mesh.vertices, mesh.faces)
            else:
                write_OBJ("Body_" + os.str(ibody + 1) + ".obj", mesh.vertices, mesh.faces)

        return

    def Write_Meshes(self):
        """This method writes all meshes."""

        for ibody in range(0, self._pyHDB.nb_bodies):  # Loop over all bodies except the first one.

            if self._pyHDB.bodies[ibody].mesh is not None:  # If one mesh is present, other ones should also be.

                # Data.
                body = self._pyHDB.bodies[ibody]
                mesh = body.mesh

                # From the body frame to the global frame.
                if self._pyHDB.bodies[ibody].horizontal_position is not None:
                    horizontal_position = self._pyHDB.bodies[ibody].horizontal_position
                    mesh.rotate([0., 0., horizontal_position[2]])
                    mesh.translate([horizontal_position[0], horizontal_position[1], 0.])

                # Write.
                if body.name is not None:
                    write_OBJ(body.name + ".obj", mesh.vertices, mesh.faces)
                else:
                    write_OBJ("Body_" + os.str(ibody + 1) + ".obj", mesh.vertices, mesh.faces)

    def Cutoff_scaling_IRF(self, tc, ibody_force, iforce, ibody_motion, idof, auto_apply=False):
        """This function applies a filter to the impule response functions without forward speed and plot the result.

        Parameters
        ----------
        float : tc.
            Cutting time.
        ibody_force : int.
            Index of the body where the radiation force is applied.
        int : i_force.
            Index of the index of the force of the current body.
        int : i_body_motion.
            Index of the body.
        int : i_dof.
            Index of the dof of the moving body.
        Bool : auto_apply, optional.
            Automatic application of the filtering, not if flase (default).
       """

        # Data.
        data = self._pyHDB.bodies[ibody_force].irf[iforce, 6 * ibody_motion + idof, :]

        # Time.
        time = self._pyHDB.time

        # Coeff.
        try:
            coeff = np.exp(-9.*time*time / (tc*tc))
        except:
            coeff = np.zeros(time.size)

        # Application of the filer.
        if auto_apply:
            bool = True
        else:
            # Plot.
            plot_db.plot_filering(data, time, 0, coeff, ibody_force, iforce, ibody_motion, idof)

            # input returns the empty string for "enter".
            yes = {'yes', 'y', 'ye', ''}
            no = {'no', 'n'}

            choice = input("Apply scaling (y/n) ? ").lower()
            if choice in yes:
                bool = True
            elif choice in no:
                bool = False
            else:
                stdout.write("Please respond with 'yes' or 'no'")

        if bool:
            self._pyHDB.bodies[ibody_force].irf[iforce, 6 * ibody_motion + idof, :] *= coeff

    def Cutoff_scaling_IRF_speed(self, tc, ibody_force, iforce, ibody_motion, idof, auto_apply=False):
        """This function applies a filter to the impule response functions with forward speed and plot the result.

        Parameters
        ----------
        float : tc.
            Cutting time.
        ibody_force : int.
            Index of the body where the radiation force is applied.
        int : i_force.
            Index of the index of the force of the current body.
        int : i_body_motion.
            Index of the body.
        int : i_dof.
            Index of the dof of the moving body.
        Bool : auto_apply, optional.
            Automatic application of the filtering, not if flase (default).
       """

        # Data.
        data = self._pyHDB.bodies[ibody_force].irf_ku[iforce, 6 * ibody_motion + idof, :]

        # Time.
        time = self._pyHDB.time

        # Coeff.
        try:
            coeff = np.exp(-9.*time*time / (tc*tc))
        except:
            coeff = np.zeros(time.size)

        # Application of the filer.
        if auto_apply:
            bool = True
        else:
            # Plot.
            plot_db.plot_filering(data, time, 1, coeff, ibody_force, iforce, ibody_motion, idof)

            # input returns the empty string for "enter".
            yes = {'yes', 'y', 'ye', ''}
            no = {'no', 'n'}

            choice = input("Apply scaling (y/n) ? ").lower()
            if choice in yes:
                bool = True
            elif choice in no:
                bool = False
            else:
                stdout.write("Please respond with 'yes' or 'no'")

        if bool:
            self._pyHDB.bodies[ibody_force].irf[iforce, 6 * ibody_motion + iforce, :] *= coeff

    def Update_radiation_mask(self):
        """This function asks the user to define the radiation coefficient which should be zeroed and update the radiation mask accordingly."""

        for ibody_force in range(0, self._pyHDB.nb_bodies):
            for ibody_motion in range(0, self._pyHDB.nb_bodies):

                # data.
                data = np.zeros((6, 6, self._pyHDB.nb_wave_freq), dtype=np.float64)
                for iforce in range(0, 6):
                    for idof in range(0, 6):
                        for iw in range(0, self._pyHDB.nb_wave_freq):
                            data[iforce, idof, iw] = np.linalg.norm(self._pyHDB.bodies[ibody_force].Damping[iforce, 6 * ibody_motion + idof, iw]
                                        + 1j * self._pyHDB.wave_freq[iw] * (self._pyHDB.bodies[ibody_force].Added_mass[iforce, 6 * ibody_motion + idof, iw]
                                        - self._pyHDB.bodies[ibody_force].Inf_Added_mass[iforce, 6 * ibody_motion + idof]))

                # Plot.
                plot_db.plot_AB_array(data, self._pyHDB.wave_freq, ibody_force, ibody_motion, self._pyHDB, False)

    def Update_radiation_mask_x_derivatives(self):
        """This function asks the user to define the radiation coefficient which should be zeroed and update the radiation mask accordingly."""

        if self._pyHDB._has_x_derivatives:

            for ibody_force in range(0, self._pyHDB.nb_bodies):
                for ibody_motion in range(0, self._pyHDB.nb_bodies):

                    # data.
                    data = np.zeros((6, 6, self._pyHDB.nb_wave_freq), dtype=np.float64)
                    for iforce in range(0, 6):
                        for idof in range(0, 6):
                            for iw in range(0, self._pyHDB.nb_wave_freq):
                                data[iforce, idof, iw] = np.linalg.norm(self._pyHDB.bodies[ibody_force].Damping_x_derivative[iforce, 6 * ibody_motion + idof, iw]
                                            + 1j * self._pyHDB.wave_freq[iw] * (self._pyHDB.bodies[ibody_force].Added_mass_x_derivative[iforce, 6 * ibody_motion + idof, iw]
                                            - self._pyHDB.bodies[ibody_force].Inf_Added_mass_x_derivative[iforce, 6 * ibody_motion + idof]))

                    # Plot.
                    plot_db.plot_AB_array(data, self._pyHDB.wave_freq, ibody_force, ibody_motion, self._pyHDB, True)

        else:
            print("No x-derivative.")

    def Plot_irf_array(self):
        """This method plots the impulse response functions per body."""

        for ibody_force in range(0, self._pyHDB.nb_bodies):
            for ibody_motion in range(0, self._pyHDB.nb_bodies):

                # Time.
                time = self._pyHDB.time

                # Data.
                data = np.zeros((6, 6, self._pyHDB.nb_time_samples), dtype=np.float64)
                for iforce in range(0, 6):
                    for idof in range(0, 6):
                        data[iforce, idof, :] = self._pyHDB.bodies[ibody_force].irf[iforce, 6 * ibody_motion + idof, :]
                # Plot.
                plot_db.plot_irf_array(data, time, ibody_force, ibody_motion, 0)

    def Plot_irf_speed_array(self):
        """This method plots the impulse response functions per body."""

        for ibody_force in range(0, self._pyHDB.nb_bodies):
            for ibody_motion in range(0, self._pyHDB.nb_bodies):

                # Time.
                time = self._pyHDB.time

                if self._pyHDB.bodies[ibody_force].irf_ku is not None:
                    # Data.
                    data = np.zeros((6, 6, self._pyHDB.nb_time_samples), dtype=np.float64)
                    for iforce in range(0, 6):
                        for idof in range(0, 6):
                            data[iforce, idof, :] = self._pyHDB.bodies[ibody_force].irf_ku[iforce, 6 * ibody_motion + idof, :]
                    # Plot.
                    plot_db.plot_irf_array(data, time, ibody_force, ibody_motion, 1)

                if self._pyHDB._has_x_derivatives:

                    if self._pyHDB.bodies[ibody_force].irf_ku_x_derivative is not None:
                        # Data.
                        data = np.zeros((6, 6, self._pyHDB.nb_time_samples), dtype=np.float64)
                        for iforce in range(0, 6):
                            for idof in range(0, 6):
                                data[iforce, idof, :] = self._pyHDB.bodies[ibody_force].irf_ku_x_derivative[iforce, 6 * ibody_motion + idof, :]

                        # Plot.
                        plot_db.plot_irf_array(data, time, ibody_force, ibody_motion, 2)

                    if self._pyHDB.bodies[ibody_force].irf_ku2 is not None:
                        # Data.
                        data = np.zeros((6, 6, self._pyHDB.nb_time_samples), dtype=np.float64)
                        for iforce in range(0, 6):
                            for idof in range(0, 6):
                                data[iforce, idof, :] = self._pyHDB.bodies[ibody_force].irf_ku2[iforce,6 * ibody_motion + idof, :]

                        # Plot.
                        plot_db.plot_irf_array(data, time, ibody_force, ibody_motion, 3)

    def export_hdb5(self, output_file = None):
        """This function writes the hydrodynamic database into a *.hdb5 file.

        Parameter
        ---------
        output_file : string, optional
            Name of the hdb5 output file.
        """

        if not self._is_initialized:

            print('========================')
            print('Intialize HDB5 database...')
            print('========================')

            self._initialize()

        print('========================')
        print('Writing HDB5 database...')
        print('========================')

        if output_file is None:
            hdb5_file = os.path.abspath('frydom.hdb5')
        else:
            # Verifying that the output file has the extension .hdb5.
            root, ext = os.path.splitext(output_file)
            if not ext == '.hdb5':
                raise IOError('Please register the output file with a .hdb5 extension.')

            hdb5_file = output_file

            if not os.path.isabs(output_file):
                hdb5_file = os.path.abspath(hdb5_file)

        # Writing all the data from _pyHDB.
        try:
            self._pyHDB.write_hdb5(hdb5_file)
        except IOError:
            raise IOError('Problem in writing HDB5 file at location %s' % hdb5_file)

        print('')
        print('-------> "%s" has been written.' % hdb5_file)
        print('')

    def read_hdb5(self, input_file = None):
        """This function loads a *.hdb5 file.

        Parameter
        ---------
        input_file : string, optional
            Name of the hdb5 input file.
        """

        if input_file is None :
            hdb5_file = os.path.abspath('frydom.hdb5')
        else:
            # Verifying that the output file has the extension .hdb5.
            root, ext = os.path.splitext(input_file)
            if not ext == '.hdb5':
                raise os.IOError('Please register the input file with a .hdb5 extension.')

            hdb5_file = input_file

            if not os.path.isabs(input_file):
                hdb5_file = os.path.abspath(hdb5_file)

        # Reading all the data from .hdb5 and creating a _pyHDB object.
        try:
            HDB5_reader.HDB5reader(self._pyHDB, hdb5_file)
        except IOError:
            raise IOError('Problem in reading HDB5 file at location %s' % hdb5_file)

        print('')
        print('-------> "%s" has been loaded.' % hdb5_file)
        print('')

    def write_info(self, input_file):
        """This method writes information about the hdb5."""

        self._pyHDB.write_info(input_file)

    def write_output_format(self, output_format):

        # Formats.
        WAMIT_formats = ["WAMIT", "Wamit", "wamit", "W", "w"]
        CSV_formats = ["CSV", "csv", "Csv", "XLSX" "XLS", "xlsx", "xls", "Xlsx", "Xls"]
        Diodore_formats = ["DIODORE", "Diodore", "diodore", "HDB", "hdb"]
        OrcaFlex_formats = ["OrcaFlex", "Orcaflex", "orcaflex"]
        formats_available = WAMIT_formats + CSV_formats + Diodore_formats + OrcaFlex_formats

        # Checking the output format.
        if output_format[0] in formats_available:

            # Which format.
            if output_format[0] in WAMIT_formats: # WAMIT.

                # Initialization of the BEM writer.
                self.BEM_writer = bem_writer.WAMIT_writer(self._pyHDB, output_format[1])

                # Hydrostatic ouput file.
                self.BEM_writer.write_hst()

                # Added mass and damping output file.
                self.BEM_writer.write_AB()

                # Excitation loads.
                self.BEM_writer.write_Fexc()

                # Mean wave drift loads.
                for body in self._pyHDB.bodies:
                    if body.has_Drift is not None:
                        self.BEM_writer.write_drift(body)

            elif output_format[0] in Diodore_formats:
                writer = DiodoreHDBWriter(self._pyHDB, output_format[1])

            elif output_format[0] in CSV_formats: # CSV

                # Initialization of the Excel writer.
                self.Excel_writer = CSV_writer.Excel_writer(self._pyHDB, output_format[1])

                # Mean wave drift loads.
                if self._pyHDB.has_Drift is not None:
                    self.Excel_writer.write_drift()

            elif output_format[0] in OrcaFlex_formats: # OrcaFlex.

                # Initialization of the BEM writer.
                self.BEM_writer = bem_writer.OrcaFlex_writer(self._pyHDB, output_format[1])

                # RAO.
                if self._pyHDB.has_RAO is not None:
                    self.BEM_writer.write_RAO()

        else:
            print("The available output formats are:")
            print(" - WAMIT")
            print(" - CSV")
            print(" - Diodore")
