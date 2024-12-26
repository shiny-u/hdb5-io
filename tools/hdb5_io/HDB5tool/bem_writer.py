#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# ==========================================================================
# FRyDoM - frydom-ee.org
#
# Copyright (c) Ecole Centrale de Nantes (LHEEA lab.) and D-ICE Engineering.
# All rights reserved.
#
# Use of this source code is governed by a GPLv3 license that can be found
# in the LICENSE file of FRyDoM.
#
# ==========================================================================

"""Module to writer the output data in the format of a frequency-domain solver."""

import os
import numpy as np
import cmath

from hdb5_io.HDB5tool.plot_db import Dof_name

class BEM_writer():

    """
    Class for writing ouput data in the format of a frequency-domain solver.
    """

    def __init__(self, pyHDB, folder_path, name_solver):

        # pyHDB.
        self._pyHDB = pyHDB

        # Creation of the output folder for storing the output files.
        self.output_folder = os.path.join(folder_path, name_solver)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def compute_S(self, vect):

        """This function computes the skew cross product matrix."""

        M = np.zeros((3, 3), dtype = np.float64)

        M[0, 1] = -vect[2]
        M[0, 2] = vect[1]
        M[1, 0] = vect[2]
        M[1, 2] = -vect[0]
        M[2, 0] = -vect[1]
        M[2, 1] = vect[0]

        return M

    def compute_T(self, vect):

        """This function computes the transport matrix for hydrodynamic databases."""

        Transport = np.zeros(3, dtype=np.float64)
        Transport[2] = vect[2] # Only the vertical component to stay in the local body frame.
        T = np.identity(6)
        T[0:3, 3:6] = -self.compute_S(Transport)

        return T

class WAMIT_writer(BEM_writer):

    """
        Class for writing WAMIT output files.
    """

    def __init__(self, pyHDB, folder_path):

        # Initialization of the mother class.
        BEM_writer.__init__(self, pyHDB, folder_path, "WAMIT")

    def write_hst(self):

        """This function writes the WAMIT hydrostatic output file."""

        Extension = ".hst"
        filename = os.path.join(self.output_folder, "Hydrodstatic_matrix" + Extension)
        nb_bodies = self._pyHDB.nb_bodies

        # Adaptation to the WAMIT format.
        mat_hs = np.zeros((6 * nb_bodies, 6 * nb_bodies), dtype = np.float64)
        for body in self._pyHDB.bodies:
            if(body._hydrostatic):

                # Evaluation of the displacement.
                if body.mesh:
                    inertia = body.mesh.eval_plain_mesh_inertias(rho_medium=self._pyHDB.rho_water)
                    displacement = inertia.mass / self._pyHDB.rho_water
                else:
                    print("warning : cannot compute inertia from mesh. Use mass=1 for adim.")
                    displacement = 1. / self._pyHDB.rho_water

                # Transport from the center of gravity to the WAMIT reference point.
                K_0 = np.copy(body._hydrostatic.matrix) # Initialization with the hydrostatic matrix at CoG.

                K_0[3, 3] += self._pyHDB.rho_water * self._pyHDB.grav * displacement * body.computation_point[2] # K44.
                K_0[4, 4] += self._pyHDB.rho_water * self._pyHDB.grav * displacement * body.computation_point[2] # K55.

                # Adimensionalization.
                K_0[2, 2] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 2)) # K33.
                K_0[2, 3] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 3)) # K34.
                K_0[3, 2] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 3)) # K43.
                K_0[2, 4] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 3)) # K35.
                K_0[4, 2] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 3)) # K53.
                K_0[3, 3] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 4)) # K44.
                K_0[3, 4] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 4)) # K45.
                K_0[4, 3] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 4)) # K54.
                K_0[4, 4] /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(self._pyHDB.normalization_length, 4)) # K55.

                # Storage.
                mat_hs[6 * body.i_body: 6 * (body.i_body + 1), 6 * body.i_body: 6 * (body.i_body + 1)] = K_0[:, :]

        # Writing the hydrostatic output file.
        with open(filename, 'w') as file:
            for irow in range(0, 6 * self._pyHDB.nb_bodies):
                for jcol in range(0, 6 * self._pyHDB.nb_bodies):
                    file.write("     %i     %i   %.6e\n" % (irow + 1, jcol + 1, mat_hs[irow, jcol]))

    def exponent_adim_AB(self, irow, jcol):

        """This function gives the exponent of the adimensionalization coefficient for the added mass and damping matrices."""

        if(irow <= 2 and jcol <= 2):
            k = 3
        elif((irow <= 2 and jcol >= 3) or (irow >= 3 and jcol <= 2)):
            k = 4
        elif(irow >= 3 and jcol >= 3):
            k = 5
        else:
            print("")
            print("exponent_adim_AB: the values of irow and jcol are not good:")
            print("irow = %i" % irow)
            print("jcol = %i" % jcol)

        return k

    def write_AB(self):

        """This function writes the WAMIT added mass and damping output file."""

        Extension = ".1"
        nw = self._pyHDB.nb_wave_freq
        nb_bodies = self._pyHDB.nb_bodies
        filename = os.path.join(self.output_folder, "Added_mass_Damping" + Extension)

        # Adaptation to the WAMIT format.
        with open(filename, 'w') as file:

            ###############################################################################
            #                   Infinite period - Zero frequency.
            ###############################################################################

            """TODO: The zero-frequecy added mass coefficients are not computed but they are equal to the added mass coefficient at the first wave frequency computed in the solver."""

            mat_A = np.zeros((6 * nb_bodies, 6 * nb_bodies), dtype=np.float64)
            for body_force in self._pyHDB.bodies:

                # Added mass matrix at the first wave frequency computed of the body at the center of gravity.
                A = np.copy(body_force.Added_mass[:, :, 0])

                # Transport from the center of gravity to the WAMIT reference point (center of the waterline area of each body)
                T_force = self.compute_T(body_force.computation_point)

                for body_motion in self._pyHDB.bodies:
                    T_motion = self.compute_T(body_motion.computation_point)
                    A[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)] = np.dot(np.transpose(T_force), np.dot(A[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)], T_motion))

                # Adimensionalization.
                for irow in range(0, 6):
                    for jcol in range(0, 6):
                        A[irow, jcol] /= (self._pyHDB.rho_water * pow(self._pyHDB.normalization_length, self.exponent_adim_AB(irow, jcol)))

                # Storage.
                mat_A[6 * body_force.i_body: 6 * (body_force.i_body + 1), :] = A[:, :]

            # Writing.
            for irow in range(0, 6 * nb_bodies):
                for jcol in range(0, 6 * nb_bodies):
                    file.write(" %.6e     %i     %i %.6e\n" % (-1, irow + 1, jcol + 1, mat_A[irow, jcol]))

            ###############################################################################
            #                   Zero period - Infinite frequency.
            ###############################################################################

            mat_A = np.zeros((6 * nb_bodies, 6 * nb_bodies), dtype=np.float64)
            for body_force in self._pyHDB.bodies:

                # Infinite added mass matrix of the body at the center of gravity.
                A = np.copy(body_force.Inf_Added_mass)

                # Transport from the center of gravity to the WAMIT reference point (center of the waterline area of each body)
                T_force = self.compute_T(body_force.computation_point)

                for body_motion in self._pyHDB.bodies:
                    T_motion = self.compute_T(body_motion.computation_point)
                    A[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)] = np.dot(np.transpose(T_force), np.dot(A[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)], T_motion))

                # Adimensionalization.
                for irow in range(0, 6):
                    for jcol in range(0, 6):
                        A[irow, jcol] /= (self._pyHDB.rho_water * pow(self._pyHDB.normalization_length, self.exponent_adim_AB(irow, jcol)))

                # Storage.
                mat_A[6 * body_force.i_body: 6 * (body_force.i_body + 1), :] = A[:, :]

            # Writing.
            for irow in range(0, 6 * nb_bodies):
                for jcol in range(0, 6 * nb_bodies):
                    # print mat_A[irow, jcol], Decimal(mat_A[irow, jcol])
                    file.write(" %.6e     %i     %i %.6e\n" % (0, irow + 1, jcol + 1, mat_A[irow, jcol]))

            ###############################################################################
            #                          Other periods.
            ###############################################################################

            for iw in range(0, nw):

                w = self._pyHDB.wave_freq[iw]
                mat_A = np.zeros((6 * nb_bodies, 6 * nb_bodies), dtype=np.float64)
                mat_B = np.zeros((6 * nb_bodies, 6 * nb_bodies), dtype=np.float64)

                for body_force in self._pyHDB.bodies:

                    # WAMIT: loads in the body coordinate system, defined from XBody.
                    # In this function, the point of computation is the center of the waterline area of each body, according to the the global frame axis.

                    # Nemoh: loads in the body frames which the direction of the axes matches those of the global frame.
                    # That represents XBody(3) = 0 and XBody(4) = 0 in WAMIT.

                    # Infinite added mass matrix of the body at the center of gravity.
                    A = np.copy(body_force.Added_mass[:, :, iw])
                    B = np.copy(body_force.Damping[:, :, iw])

                    # Transport from the center of gravity to the WAMIT reference point (center of the waterline area of each body)
                    T_force = self.compute_T(body_force.computation_point)

                    for body_motion in self._pyHDB.bodies:
                        T_motion = self.compute_T(body_motion.computation_point)
                        A[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)] = np.dot(np.transpose(T_force), np.dot(A[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)], T_motion))
                        B[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)] = np.dot(np.transpose(T_force), np.dot(B[:, 6 * body_motion.i_body: 6 * (body_motion.i_body + 1)], T_motion))

                    # Adimensionalization.
                    for irow in range(0, 6):
                        for jcol in range(0, 6):
                            A[irow, jcol] /= (self._pyHDB.rho_water * pow(self._pyHDB.normalization_length, self.exponent_adim_AB(irow, jcol)))
                            B[irow, jcol] /= (self._pyHDB.rho_water * w * pow(self._pyHDB.normalization_length, self.exponent_adim_AB(irow, jcol)))

                    # Storage.
                    mat_A[6 * body_force.i_body: 6 * (body_force.i_body + 1), :] = A[:, :]
                    mat_B[6 * body_force.i_body: 6 * (body_force.i_body + 1), :] = B[:, :]

                # Writing.
                for irow in range(0, 6 * nb_bodies):
                    for jcol in range(0, 6 * nb_bodies):
                        file.write(" %.6e     %i     %i %.6e %.6e\n" % (2*np.pi / w, irow + 1, jcol + 1, mat_A[irow, jcol], mat_B[irow, jcol]))

    def exponent_adim_Fexc(self, irow):

        """This function gives the exponent of the adimensionalization coefficient for the excitation loads."""

        if(irow <= 2):
            k = 2
        elif(irow >= 3):
            k = 3
        else:
            print("")
            print("exponent_adim_Fexc: the values of irow is not good:")
            print("irow = %i" % irow)

        return k

    def write_Fexc(self):

        """This function writes the WAMIT excitation load output file."""

        Extension = ".3"
        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_wave_dir
        nb_bodies = self._pyHDB.nb_bodies
        filename = os.path.join(self.output_folder, "Excitation_loads" + Extension)
        Wave_amplitude = 1.

        # Adaptation to the WAMIT format.
        with open(filename, 'w') as file:

            for iw in range(0, nw):

                w = self._pyHDB.wave_freq[iw]

                for ibeta in range(0, nbeta):

                    beta = np.degrees(self._pyHDB.wave_dir[ibeta]) # deg.
                    Vect_Fexc = np.zeros((6 * nb_bodies), dtype=np.complex64)

                    for body_force in self._pyHDB.bodies:

                        # WAMIT: loads in the body coordinate system, defined from XBody.
                        # In this function, the point of computation is the center of the waterline area of each body, according to the the global frame axis.

                        # Nemoh: loads in the body frames which the direction of the axes matches those of the global frame.
                        # That represents XBody(3) = 0 and XBody(4) = 0 in WAMIT.

                        # WAMIT: A*exp(j*omega*t)
                        # Nemoh: A*exp(-j*omega*t)
                        # Re_WAMIT = Re_Nemoh
                        # Im_WAMIT = -Im_Nemoh

                        # Excitation loads at the center of gravity.
                        Fexc = np.copy(body_force.Diffraction[:, iw, ibeta] + body_force.Froude_Krylov[:, iw, ibeta])

                        # # Rotation matrix for WAMIT05 validation test case.
                        # if(body_force.i_body == 1):
                        #     Rot_mat = np.zeros((3, 3), dtype = np.float64)
                        #     angle = np.pi/2.
                        #     Rot_mat[0, 0] = np.cos(angle)
                        #     Rot_mat[0, 1] = -np.sin(angle)
                        #     Rot_mat[1, 0] = np.sin(angle)
                        #     Rot_mat[1, 1] = np.cos(angle)
                        #     Rot_mat[2, 2] = 1
                        #     Rotation = np.zeros((6, 6), dtype = np.float64)
                        #     Rotation[0:3, 0:3] = np.copy(Rot_mat)
                        #     Rotation[3:6, 3:6] = np.copy(Rot_mat)
                        # else:
                        #     Rotation = np.identity(6)
                        # Fexc = np.dot(np.transpose(Rotation), Fexc)

                        # Transport from the center of gravity to the WAMIT reference point (center of the waterline area of each body)
                        T = self.compute_T(body_force.computation_point)

                        # Real part.
                        Fexc.real = np.dot(np.transpose(T), Fexc.real)

                        # Imaginary part.
                        Fexc.imag = -np.dot(np.transpose(T), Fexc.imag)

                        # Adimensionalization.
                        for irow in range(0, 6):
                            Fexc[irow] /= (self._pyHDB.rho_water * self._pyHDB.grav * Wave_amplitude * pow(self._pyHDB.normalization_length, self.exponent_adim_Fexc(irow)))

                        # Storage.
                        Vect_Fexc[6 * body_force.i_body: 6 * (body_force.i_body + 1)] = Fexc

                    # Writing.
                    for irow in range(0, 6 * nb_bodies):
                        file.write(" %.6e  %.6e     %i  %.6e  %.6e  %.6e  %.6e\n" % (2*np.pi / w, beta, irow + 1, abs(Vect_Fexc[irow]), np.degrees(cmath.phase(Vect_Fexc[irow]))
                                                                                     , Vect_Fexc[irow].real, Vect_Fexc[irow].imag))

    def write_drift(self, body):

        """This function writes the WAMIT mean wave drift load output file."""

        # TODO: Use self._pyHDB.Wave_drift_force instead of self._pyHDB.wave_drift.

        Extension = ".8"
        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_wave_dir
        filename = os.path.join(self.output_folder, body.name.strip() + "_Drift" + Extension)
        Wave_amplitude = 1.

        # Adaptation to the WAMIT format.
        with open(filename, 'w') as file:

            for iw in range(0, nw):

                w = self._pyHDB.wave_freq[iw]

                for ibeta in range(0, nbeta):

                    beta = np.degrees(self._pyHDB.wave_dir[ibeta]) # deg.

                    # for key, mode in self.wave_drift.modes.items():
                    for key in Dof_name:

                        # iforce.
                        if (key == 'surge'):
                            iforce = 0
                        elif (key == 'sway'):
                            iforce = 1
                        elif (key == 'heave'):
                            iforce = 2
                        elif (key == 'roll'):
                            iforce = 3
                        elif (key == 'pitch'):
                            iforce = 4
                        else: # Yaw.
                            iforce = 5

                        if(iforce == 0 or iforce == 1 or iforce == 5):

                            # Mean wave drift loads.
                            data = body.Wave_drift_force[iforce, iw, ibeta]

                            # Adimensionalization.
                            if(iforce == 0 or iforce == 1): # Surge and sway.
                                data /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(Wave_amplitude, 2) * self._pyHDB.normalization_length)
                            else: # Yaw.
                                data /= (self._pyHDB.rho_water * self._pyHDB.grav * pow(Wave_amplitude, 2) * pow(self._pyHDB.normalization_length, 2))

                            # Writing.
                            file.write(" %.6e  %.6e  %.6e     %i  %.6e  %.6e  %.6e  %.6e\n" % (2 * np.pi / w, beta, beta, iforce + 1, abs(data),
                                                                                               -np.degrees(cmath.phase(data)), data.real, data.imag))

class OrcaFlex_writer(BEM_writer):

    """
        Class for writing OrcaFlex output files.
    """

    def __init__(self, pyHDB, folder_path):

        # Initialization of the mother class.
        BEM_writer.__init__(self, pyHDB, folder_path, "OrcaFlex")

    def write_RAO(self):

        """This method writes the OrcaFlex RAO input file."""

        nw = self._pyHDB.nb_wave_freq
        nbeta = self._pyHDB.nb_wave_dir
        nb_bodies = self._pyHDB.nb_bodies
        Extension = ".txt"
        filename = os.path.join(self.output_folder, "RAO" + Extension)

        # Adaptation to the WAMIT format.
        with open(filename, 'w') as file:

            # Header.
            file.write("OrcaFlex Displacement RAO Start\n")

            for body_force in self._pyHDB.bodies:
                file.write("Draught " + body_force.name + "\n")

                for ibeta in range(0, nbeta):
                    beta = np.degrees(self._pyHDB.wave_dir[ibeta]) # deg.
                    file.write("Direction " + str(beta) + "\n")
                    file.write("WFR XA XP YA YP ZA ZP RXA RXP RYA RYP RZA RZP\n")

                    for iw in range(0, nw):
                        w = self._pyHDB.wave_freq[iw]
                        RAO = body_force.RAO[:, iw, ibeta]
                        RAO_abs = np.abs(RAO)
                        RAO_deg = np.angle(RAO, deg=True)
                        file.write("%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n"
                                   % (w, RAO_abs[0], RAO_deg[0], RAO_abs[1], RAO_deg[1], RAO_abs[2], RAO_deg[2],
                                      np.degrees(RAO_abs[3]), RAO_deg[3], np.degrees(RAO_abs[4]), RAO_deg[4],
                                      np.degrees(RAO_abs[5]), RAO_deg[5]))

            # Ender.
            file.write("OrcaFlex Displacement RAO End")