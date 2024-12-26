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

import numpy as np

import hdb5_io.HDB5tool.pyHDB as pyHDB
import hdb5_io.HDB5tool.body_db as body_db

class Merger(object):
    """
        Class for merging two pyHDB of the same problem, only the wave freqency ranges are different.
    """

    def __init__(self, pyHDB_1, pyHDB_2):

        """
            Constructor of the class Merger.
        """

        # By definition, self._pyHDB_1 has the lowest frequency range and self._pyHDB_2 the highest one.
        if(pyHDB_1.max_wave_freq < pyHDB_2.min_wave_freq):
            self._pyHDB_1 = pyHDB_1
            self._pyHDB_2 = pyHDB_2
        elif(pyHDB_2.max_wave_freq < pyHDB_1.min_wave_freq):
            self._pyHDB_1 = pyHDB_2
            self._pyHDB_2 = pyHDB_1
        else: # Check that the two wave frequency ranges are disjoint.
            print("Wave frequency ranges are not disjoint.")
            print("This case is not taken into account by hdb5merge.")
            print("Range of the first *.hdb5: [%f , %f]" % (pyHDB_1.min_wave_freq, pyHDB_1.max_wave_freq))
            print("Range of the second *.hdb5: [%f , %f]" % (pyHDB_2.min_wave_freq, pyHDB_2.max_wave_freq))
            exit()

    def merge_version(self, pyHDB_out):

        """
            This method merges the version of the two pyHDB.
        """

        pyHDB_out.version = self._pyHDB_1.version_max

    def merge_commit_hash(self, pyHDB_out):

        """
            This method merges the commit hashes of the two pyHDB.
        """

        pyHDB_out.commit_hash = self._pyHDB_1.commit_hash

    def merge_numerical_parameters(self, pyHDB_out):

        """
            This method merges the expert numerical parameters of the two pyHDB.
        """

        #TODO: Should they be really identical?

        # Presence of expert numerical parameters.
        assert (self._pyHDB_1.has_expert_parameters == self._pyHDB_2.has_expert_parameters)
        pyHDB_out.has_expert_parameters = self._pyHDB_1.has_expert_parameters

        # Surface integration order.
        assert (self._pyHDB_1.surface_integration_order == self._pyHDB_2.surface_integration_order)
        pyHDB_out.surface_integration_order = self._pyHDB_1.surface_integration_order

        # Infinite-depth Green's function.
        assert (self._pyHDB_1.green_function == self._pyHDB_2.green_function)
        pyHDB_out.green_function = self._pyHDB_1.green_function

        # Crmax.
        assert (self._pyHDB_1.crmax == self._pyHDB_2.crmax)
        pyHDB_out.crmax = self._pyHDB_1.crmax

    def merge_environment(self, pyHDB_out):

        """
            This method merges the environmental data of the two pyHDB.
        """

        # Gravity acceleration.
        assert (self._pyHDB_1.grav == self._pyHDB_2.grav)
        pyHDB_out.grav = self._pyHDB_1.grav

        # Water density.
        assert (self._pyHDB_1.rho_water == self._pyHDB_2.rho_water)
        pyHDB_out.rho_water = self._pyHDB_1.rho_water

        # Normalisation length.
        assert (self._pyHDB_1.normalization_length == self._pyHDB_2.normalization_length)
        pyHDB_out.normalization_length = self._pyHDB_1.normalization_length

        # Water depth.
        assert (self._pyHDB_1.depth == self._pyHDB_2.depth)
        pyHDB_out.depth = self._pyHDB_1.depth

        # Number of bodies.
        assert (self._pyHDB_1.nb_bodies == self._pyHDB_2.nb_bodies)
        pyHDB_out.nb_bodies = self._pyHDB_1.nb_bodies

        # Solver.
        assert (self._pyHDB_1.solver == self._pyHDB_2.solver)
        pyHDB_out.solver = self._pyHDB_1.solver

        # Wave mesure point.
        assert (self._pyHDB_1.x_wave_measure == self._pyHDB_2.x_wave_measure)
        assert (self._pyHDB_1.y_wave_measure == self._pyHDB_2.y_wave_measure)
        pyHDB_out.x_wave_measure = self._pyHDB_1.x_wave_measure
        pyHDB_out.y_wave_measure = self._pyHDB_1.y_wave_measure

    def merge_discretization(self, pyHDB_out):

        """
            This method merges the discretization data of the two pyHDB.
        """

        # Wave frequency discretization.
        pyHDB_out.nb_wave_freq = self._pyHDB_1.nb_wave_freq + self._pyHDB_2.nb_wave_freq
        pyHDB_out.min_wave_freq = self._pyHDB_1.min_wave_freq # By definition pyHDB_1 has the lowest frequency range.
        pyHDB_out.max_wave_freq = self._pyHDB_2.max_wave_freq # By definition pyHDB_2 has the highest frequency range.
        pyHDB_out.wave_freq = np.concatenate([self._pyHDB_1.wave_freq, self._pyHDB_2.wave_freq], axis=0)

        # Wave direction discretization.
        assert (self._pyHDB_1.nb_wave_dir == self._pyHDB_2.nb_wave_dir)
        pyHDB_out.nb_wave_dir = self._pyHDB_1.nb_wave_dir
        assert (self._pyHDB_1.min_wave_dir == self._pyHDB_2.min_wave_dir)
        pyHDB_out.min_wave_dir = self._pyHDB_1.min_wave_dir
        assert (self._pyHDB_1.max_wave_dir == self._pyHDB_2.max_wave_dir)
        pyHDB_out.max_wave_dir = self._pyHDB_1.max_wave_dir
        pyHDB_out.wave_dir = self._pyHDB_1.wave_dir

        # Time samples.
        assert (self._pyHDB_1.nb_time_samples == self._pyHDB_2.nb_time_samples)
        pyHDB_out.nb_time_samples = self._pyHDB_1.nb_time_samples
        assert (self._pyHDB_1.dt == self._pyHDB_2.dt)
        pyHDB_out.dt = self._pyHDB_1.dt
        pyHDB_out.time = self._pyHDB_1.time

    def merge_symmetries(self, pyHDB_out):

        """
            This method merges the symmetry data of the two pyHDB.
        """

        assert (self._pyHDB_1.bottom_sym == self._pyHDB_2.bottom_sym)
        pyHDB_out.bottom_sym = self._pyHDB_1.bottom_sym
        assert (self._pyHDB_1.xoz_sym == self._pyHDB_2.xoz_sym)
        pyHDB_out.xoz_sym = self._pyHDB_1.xoz_sym
        assert (self._pyHDB_1.yoz_sym == self._pyHDB_2.yoz_sym)
        pyHDB_out.yoz_sym = self._pyHDB_1.yoz_sym

    def merge_VF(self, pyHDB_out):

        """
            This method merges the VF data of the two pyHDB.
        """

        assert (self._pyHDB_1.max_order == self._pyHDB_2.max_order)
        pyHDB_out.max_order = self._pyHDB_1.max_order
        assert (self._pyHDB_1.relaxed == self._pyHDB_2.relaxed)
        pyHDB_out.relaxed = self._pyHDB_1.relaxed
        assert (self._pyHDB_1.tolerance == self._pyHDB_2.tolerance)
        pyHDB_out.tolerance = self._pyHDB_1.tolerance
        assert (self._pyHDB_1.has_VF == self._pyHDB_2.has_VF)
        pyHDB_out.has_VF = self._pyHDB_1.has_VF

    def merge_excitation(self, body_1, body_2, body_out, pyHDB_out):

        """
            This method merges the excitation loads of the two pyHDB.
        """

        # Froude-Kylov loads.
        assert (body_out.Froude_Krylov.shape[0] == body_1.Froude_Krylov.shape[0] == body_2.Froude_Krylov.shape[0]) # Dof.
        assert (body_out.Froude_Krylov.shape[2] == body_1.Froude_Krylov.shape[2] == body_2.Froude_Krylov.shape[2]) # Wave directions.
        body_out.Froude_Krylov = np.concatenate([body_1.Froude_Krylov, body_2.Froude_Krylov], axis=1) # Wave frequencies.

        # x-derivative of the Froude-Kylov loads.
        if(pyHDB_out._has_x_derivatives):
            assert (body_out.Froude_Krylov_x_derivative.shape[0] == body_1.Froude_Krylov_x_derivative.shape[0] == body_2.Froude_Krylov_x_derivative.shape[0]) # Dof.
            assert (body_out.Froude_Krylov_x_derivative.shape[2] == body_1.Froude_Krylov_x_derivative.shape[2] == body_2.Froude_Krylov_x_derivative.shape[2]) # Wave directions.
            body_out.Froude_Krylov_x_derivative = np.concatenate([body_1.Froude_Krylov_x_derivative, body_2.Froude_Krylov_x_derivative], axis=1) # Wave frequencies.

        # Diffraction loads.
        assert (body_out.Diffraction.shape[0] == body_1.Diffraction.shape[0] == body_2.Diffraction.shape[0]) # Dof.
        assert (body_out.Diffraction.shape[2] == body_1.Diffraction.shape[2] == body_2.Diffraction.shape[2]) # Wave directions.
        body_out.Diffraction = np.concatenate([body_1.Diffraction, body_2.Diffraction], axis=1) # Wave frequencies.

        # x_derivative of the diffraction loads.
        if (pyHDB_out._has_x_derivatives):
            assert (body_out.Diffraction_x_derivative.shape[0] == body_1.Diffraction_x_derivative.shape[0] == body_2.Diffraction_x_derivative.shape[0]) # Dof.
            assert (body_out.Diffraction_x_derivative.shape[2] == body_1.Diffraction_x_derivative.shape[2] == body_2.Diffraction_x_derivative.shape[2]) # Wave directions.
            body_out.Diffraction_x_derivative = np.concatenate([body_1.Diffraction_x_derivative, body_2.Diffraction_x_derivative], axis=1) # Wave frequencies.

    def merge_radiation(self, body_1, body_2, body_out, pyHDB_out):

        """
            This method merges the radiation quantities of the two pyHDB.
        """

        # Infinite-frequency added mass.
        assert (np.all(body_1.Inf_Added_mass == body_2.Inf_Added_mass))
        body_out.Inf_Added_mass = body_1.Inf_Added_mass

        # x-derivative of the infinite-frequency added mass.
        if (pyHDB_out._has_x_derivatives):
            assert (np.all(body_1.Inf_Added_mass_x_derivative == body_2.Inf_Added_mass_x_derivative))
            body_out.Inf_Added_mass_x_derivative = body_1.Inf_Added_mass_x_derivative

        # Zero-frequency added mass.
        assert (np.all(body_1.Zero_Added_mass == body_2.Zero_Added_mass))
        body_out.Zero_Added_mass = body_1.Zero_Added_mass

        # x-derivative of the zero-frequency added mass.
        if (pyHDB_out._has_x_derivatives):
            assert (np.all(body_1.Zero_Added_mass_x_derivative == body_2.Zero_Added_mass_x_derivative))
            body_out.Zero_Added_mass_x_derivative = body_1.Zero_Added_mass_x_derivative

        # Radiation mask.
        assert (np.all(body_1.Radiation_mask == body_2.Radiation_mask))
        body_out.Radiation_mask = body_1.Radiation_mask

        # Added mass.
        assert (body_out.Added_mass.shape[0] == body_1.Added_mass.shape[0] == body_2.Added_mass.shape[0]) # i_force.
        assert (body_out.Added_mass.shape[1] == body_1.Added_mass.shape[1] == body_2.Added_mass.shape[1]) # i_motion.
        body_out.Added_mass = np.concatenate([body_1.Added_mass, body_2.Added_mass], axis=2) # Wave frequencies.

        # x-derivative of the added mass.
        if (pyHDB_out._has_x_derivatives):
            assert (body_out.Added_mass_x_derivative.shape[0] == body_1.Added_mass_x_derivative.shape[0] == body_2.Added_mass_x_derivative.shape[0]) # i_force.
            assert (body_out.Added_mass_x_derivative.shape[1] == body_1.Added_mass_x_derivative.shape[1] == body_2.Added_mass_x_derivative.shape[1]) # i_motion.
            body_out.Added_mass_x_derivative = np.concatenate([body_1.Added_mass_x_derivative, body_2.Added_mass_x_derivative], axis=2) # Wave frequencies.

        # Damping.
        assert (body_out.Damping.shape[0] == body_1.Damping.shape[0] == body_2.Damping.shape[0]) # i_force.
        assert (body_out.Damping.shape[1] == body_1.Damping.shape[1] == body_2.Damping.shape[1]) # i_motion.
        body_out.Damping = np.concatenate([body_1.Damping, body_2.Damping], axis=2) # Wave frequencies.

        # x-derivative of the damping.
        if (pyHDB_out._has_x_derivatives):
            assert (body_out.Damping_x_derivative.shape[0] == body_1.Damping_x_derivative.shape[0] == body_2.Damping_x_derivative.shape[0])  # i_force.
            assert (body_out.Damping_x_derivative.shape[1] == body_1.Damping_x_derivative.shape[1] == body_2.Damping_x_derivative.shape[1])  # i_motion.
            body_out.Damping_x_derivative = np.concatenate([body_1.Damping_x_derivative, body_2.Damping_x_derivative], axis=2)  # Wave frequencies.

        # Impulse response functions without forward speed.
        assert (body_1.irf == body_2.irf)
        if(body_1.irf is not None and body_2.irf is not None):
            body_out.irf = np.zeros((6, 6 * pyHDB_out.nb_bodies, pyHDB_out.nb_time_samples), dtype=np.float)
            assert (np.all(body_1.irf == body_2.irf))
            body_out.irf = body_1.irf

        # Impulse response functions with forward speed.
        assert (body_1.irf_ku == body_2.irf_ku)
        if (body_1.irf_ku is not None and body_2.irf_ku is not None):
            body_out.irf_ku = np.zeros((6, 6 * pyHDB_out.nb_bodies, pyHDB_out.nb_time_samples), dtype=np.float)
            assert (np.all(body_1.irf_ku == body_2.irf_ku))
            body_out.irf_ku = body_1.irf_ku

        # Poles and residues.
        if(pyHDB_out.has_VF):
            print("The vector-fitting approximation of the hdb with the lowest wave frequency range is used. "
                  "Please check the quality of the vector-fitting approximation.\n")
            body_out.poles_residues = body_1.poles_residues

    def merge_hydrostatics(self, body_1, body_2, body_out):

        """
            This method merges the hydrostatics data of the two pyHDB.
        """

        if(body_1.hydrostatic is not None and body_2.hydrostatic is not None):
            body_out.activate_hydrostatic()
            assert (np.all(body_1.hydrostatic.matrix == body_2.hydrostatic.matrix))
            body_out.hydrostatic.matrix = body_1.hydrostatic.matrix

    def merge_mass_matrix(self, body_1, body_2, body_out):

        """
            This method merges the inertial data of the two pyHDB.
        """

        if (body_1.inertia is not None and body_2.inertia is not None):
            body_out.activate_inertia()
            assert (body_1.inertia.mass == body_2.inertia.mass)
            assert (np.all(body_1.inertia.matrix33 == body_2.inertia.matrix33))
            body_out.inertia.matrix = body_1.inertia.matrix

    def merge_mooring_matrix(self, body_1, body_2, body_out):

        """
            This method merges the mooring data of the two pyHDB.
        """

        if (body_1.mooring is not None and body_2.mooring is not None):
            body_out.activate_mooring()
            assert (np.all(body_1.mooring == body_2.mooring))
            body_out.mooring = body_1.mooring

    def merge_extra_linear_damping_matrix(self, body_1, body_2, body_out):

        """
            This method merges the extra linear damping data of the two pyHDB.
        """

        if (body_1.extra_damping is not None and body_2.extra_damping is not None):
            body_out.activate_extra_damping()
            assert (np.all(body_1.extra_damping == body_2.extra_damping))
            body_out.extra_damping = body_1.extra_damping

    def merge_RAO(self, body_1, body_2, body_out, pyHDB_out):

        """
            This method merges the RAO of the two pyHDB.
        """

        body_out.RAO = np.zeros((6, pyHDB_out.nb_wave_freq, pyHDB_out.nb_wave_dir), dtype=np.complex)
        assert (body_out.RAO.shape[0] == body_1.RAO.shape[0] == body_2.RAO.shape[0]) # Dof.
        assert (body_out.RAO.shape[2] == body_1.RAO.shape[2] == body_2.RAO.shape[2]) # Wave directions.
        body_out.RAO = np.concatenate([body_1.RAO, body_2.RAO], axis=1) # Wave frequencies.

    def merge_Eigenfrequencies(self, body_1, body_2, body_out):

        """
            This method merges the eigenfrequencies of the two pyHDB.
        """

        body_out.Eigenfrequencies = np.zeros((6), dtype=np.float)
        assert (np.all(body_1.Eigenfrequencies == body_2.Eigenfrequencies))
        body_out.Eigenfrequencies = body_1.Eigenfrequencies

    def merge_bodies(self, pyHDB_out):

        """
            This method merges the body data of the two pyHDB.
        """

        for ibody in range(0, pyHDB_out.nb_bodies):

            # Input bodies.
            body_1 = self._pyHDB_1.bodies[ibody]
            body_2 = self._pyHDB_2.bodies[ibody]

            # Index.
            assert (body_1.i_body == body_2.i_body)

            # Body_out.
            if (pyHDB_out.version == 2.0):
                body_out = body_db.BodyDB(body_1.i_body, pyHDB_out.nb_bodies, pyHDB_out.nb_wave_freq, pyHDB_out.nb_wave_dir, body_1.mesh)
            else:
                body_out = body_db.BodyDB(body_1.i_body, pyHDB_out.nb_bodies, pyHDB_out.nb_wave_freq, pyHDB_out.nb_wave_dir)

            # Name.
            assert (body_1.name == body_2.name)
            body_out.name = body_1.name

            # Mesh.
            if (body_1.mesh is not None and body_2.mesh is not None):
                assert (body_1.mesh.nb_vertices == body_2.mesh.nb_vertices)
                assert (body_1.mesh.nb_faces == body_2.mesh.nb_faces)
                body_out.mesh = body_1.mesh

            # Horizontal position in world (m, m, deg).
            assert (np.all(body_1.horizontal_position == body_2.horizontal_position))
            body_out.horizontal_position = body_1.horizontal_position

            # Computation point in body frame.
            assert (np.all(body_1.computation_point == body_2.computation_point))
            body_out.computation_point = body_1.computation_point

            # Wave reference point in body frame.
            assert (np.all(body_1.wave_reference_point_in_body_frame == body_2.wave_reference_point_in_body_frame))
            body_out.wave_reference_point_in_body_frame = body_1.wave_reference_point_in_body_frame

            # Point.
            assert (np.all(body_1.point == body_2.point))
            body_out.point = body_1.point

            # Motion mask.
            assert (np.all(body_1.Motion_mask == body_2.Motion_mask))
            body_out.Motion_mask = body_1.Motion_mask

            # Force mask.
            assert (np.all(body_1.Force_mask == body_2.Force_mask))
            body_out.Force_mask = body_1.Force_mask

            # x-derivatives.
            assert (self._pyHDB_1._has_x_derivatives == self._pyHDB_2._has_x_derivatives)
            pyHDB_out._has_x_derivatives = self._pyHDB_1._has_x_derivatives
            if(pyHDB_out._has_x_derivatives):
                nw = pyHDB_out.nb_wave_freq
                nbeta = pyHDB_out.nb_wave_dir
                body_out.Froude_Krylov_x_derivative = np.zeros((6, nw, nbeta), dtype=np.complex)
                body_out.Diffraction_x_derivative = np.zeros((6, nw, nbeta), dtype=np.complex)
                body_out.Added_mass_x_derivative = np.zeros((6, 6 * pyHDB_out.nb_bodies, nw), dtype=np.float)
                body_out.Inf_Added_mass_x_derivative = np.zeros((6, 6 * pyHDB_out.nb_bodies), dtype=np.float)
                body_out.Zero_Added_mass_x_derivative = np.zeros((6, 6 * pyHDB_out.nb_bodies), dtype=np.float)
                body_out.Damping_x_derivative = np.zeros((6, 6 * pyHDB_out.nb_bodies, nw), dtype=np.float)

            # Excitation loads.
            assert (self._pyHDB_1._has_froude_krylov == self._pyHDB_2._has_froude_krylov)
            pyHDB_out._has_froude_krylov = self._pyHDB_1._has_froude_krylov
            self.merge_excitation(body_1, body_2, body_out, pyHDB_out)

            # Added mass and damping coefficients, impulse response functions and poles and residues of the VF.
            assert (self._pyHDB_1._has_infinite_added_mass == self._pyHDB_2._has_infinite_added_mass)
            pyHDB_out._has_infinite_added_mass = self._pyHDB_1._has_infinite_added_mass
            self.merge_radiation(body_1, body_2, body_out, pyHDB_out)

            # Hydrostatics.
            self.merge_hydrostatics(body_1, body_2, body_out)

            # Mass matrix.
            self.merge_mass_matrix(body_1, body_2, body_out)

            # Mooring matrix.
            self.merge_mooring_matrix(body_1, body_2, body_out)

            # Extra linear damping matrix.
            self.merge_extra_linear_damping_matrix(body_1, body_2, body_out)

            # RAO.
            assert(self._pyHDB_1.has_RAO == self._pyHDB_2.has_RAO)
            pyHDB_out.has_RAO = self._pyHDB_1.has_RAO
            if(pyHDB_out.has_RAO):
                self.merge_RAO(body_1, body_2, body_out, pyHDB_out)

            # Eigenfrequencies.
            assert (self._pyHDB_1.has_Eigenfrequencies == self._pyHDB_2.has_Eigenfrequencies)
            pyHDB_out.has_Eigenfrequencies = self._pyHDB_1.has_Eigenfrequencies
            if (pyHDB_out.has_Eigenfrequencies):
                self.merge_Eigenfrequencies(body_1, body_2, body_out)

            # Add body to pyHDB.
            pyHDB_out.append(body_out)

    def merge_wave_field(self, pyHDB_out):

        """
            This method merges the wave field data of the two pyHDB.
        """

        assert (self._pyHDB_1.has_wave_field == self._pyHDB_2.has_wave_field)
        pyHDB_out.has_wave_field = self._pyHDB_1.has_wave_field

    def merge_Kochin(self, pyHDB_out):

        """
            This method merges the wave field data of the two pyHDB.
        """

        # Angular discretization.
        assert (self._pyHDB_1.min_angle_kochin == self._pyHDB_2.min_angle_kochin)
        pyHDB_out.min_angle_kochin = self._pyHDB_1.min_angle_kochin
        assert (self._pyHDB_1.max_angle_kochin == self._pyHDB_2.max_angle_kochin)
        pyHDB_out.max_angle_kochin = self._pyHDB_1.max_angle_kochin
        assert (self._pyHDB_1.nb_angle_kochin == self._pyHDB_2.nb_angle_kochin)
        pyHDB_out.nb_angle_kochin = self._pyHDB_1.nb_angle_kochin
        pyHDB_out.angle_kochin = self._pyHDB_1.angle_kochin

        # Wave directions.
        assert (self._pyHDB_1.nb_dir_kochin == self._pyHDB_2.nb_dir_kochin)
        pyHDB_out.nb_dir_kochin = self._pyHDB_1.nb_dir_kochin
        assert (self._pyHDB_1.min_dir_kochin == self._pyHDB_2.min_dir_kochin)
        pyHDB_out.min_dir_kochin = self._pyHDB_1.min_dir_kochin
        assert (self._pyHDB_1.max_dir_kochin == self._pyHDB_2.max_dir_kochin)
        pyHDB_out.max_dir_kochin = self._pyHDB_1.max_dir_kochin
        pyHDB_out.wave_dir_kochin = self._pyHDB_1.wave_dir_kochin

        # Parameters.
        ntheta = pyHDB_out.nb_angle_kochin
        nw = pyHDB_out.nb_wave_freq
        nbeta = pyHDB_out.nb_dir_kochin
        nbodies = pyHDB_out.nb_bodies

        # Diffraction Kochin functions.
        pyHDB_out.kochin_diffraction = np.zeros((nbeta, nw, ntheta), dtype=np.complex)
        assert (pyHDB_out.kochin_diffraction.shape[0] == self._pyHDB_1.kochin_diffraction.shape[0] == self._pyHDB_2.kochin_diffraction.shape[0]) # Wave directions.
        assert (pyHDB_out.kochin_diffraction.shape[2] == self._pyHDB_1.kochin_diffraction.shape[2] == self._pyHDB_2.kochin_diffraction.shape[2]) # Angular discretization.
        pyHDB_out.kochin_diffraction = np.concatenate([self._pyHDB_1.kochin_diffraction, self._pyHDB_2.kochin_diffraction], axis=1) # Wave frequencies.

        # Angular discretization of the diffraction Kochin functions.
        pyHDB_out.kochin_diffraction_derivative = np.zeros((nbeta, nw, ntheta), dtype=np.complex)
        assert (pyHDB_out.kochin_diffraction_derivative.shape[0] == self._pyHDB_1.kochin_diffraction_derivative.shape[0] == self._pyHDB_2.kochin_diffraction_derivative.shape[0]) # Wave directions.
        assert (pyHDB_out.kochin_diffraction_derivative.shape[2] == self._pyHDB_1.kochin_diffraction_derivative.shape[2] == self._pyHDB_2.kochin_diffraction_derivative.shape[2]) # Angular discretization.
        pyHDB_out.kochin_diffraction_derivative = np.concatenate([self._pyHDB_1.kochin_diffraction_derivative, self._pyHDB_2.kochin_diffraction_derivative], axis=1) # Wave frequencies.

        # Radiation Kochin functions.
        pyHDB_out.kochin_radiation = np.zeros((6 * nbodies, nw, ntheta), dtype=np.complex)
        assert (pyHDB_out.kochin_radiation.shape[0] == self._pyHDB_1.kochin_radiation.shape[0] == self._pyHDB_2.kochin_radiation.shape[0]) # Bodies and dof.
        assert (pyHDB_out.kochin_radiation.shape[2] == self._pyHDB_1.kochin_radiation.shape[2] == self._pyHDB_2.kochin_radiation.shape[2]) # Angular discretization.
        pyHDB_out.kochin_radiation = np.concatenate([self._pyHDB_1.kochin_radiation, self._pyHDB_2.kochin_radiation], axis=1) # Wave frequencies.

        # Angular discretization of the radiation Kochin functions.
        pyHDB_out.kochin_radiation_derivative = np.zeros((6 * nbodies, nw, ntheta), dtype=np.complex)
        assert (pyHDB_out.kochin_radiation_derivative.shape[0] == self._pyHDB_1.kochin_radiation_derivative.shape[0] == self._pyHDB_2.kochin_radiation_derivative.shape[0]) # Bodies and dof.
        assert (pyHDB_out.kochin_radiation_derivative.shape[2] == self._pyHDB_1.kochin_radiation_derivative.shape[2] == self._pyHDB_2.kochin_radiation_derivative.shape[2]) # Angular discretization.
        pyHDB_out.kochin_radiation_derivative = np.concatenate([self._pyHDB_1.kochin_radiation_derivative, self._pyHDB_2.kochin_radiation_derivative], axis=1) # Wave frequencies.

    def merge_drift(self, pyHDB_out):

        """
            This method merges the drift data of the two pyHDB.
        """

        # sym_x.
        assert (self._pyHDB_1.sym_x == self._pyHDB_2.sym_x)
        pyHDB_out.sym_x = self._pyHDB_1.sym_x

        # sym_y.
        assert (self._pyHDB_1.sym_y == self._pyHDB_2.sym_y)
        pyHDB_out.sym_y = self._pyHDB_1.sym_y

        # Kochin angular step.
        assert (self._pyHDB_1.kochin_step == self._pyHDB_2.kochin_step)
        pyHDB_out.kochin_step = self._pyHDB_1.kochin_step

        # Kochin functions and their angular derivatives.
        assert (self._pyHDB_1.has_kochin == self._pyHDB_2.has_kochin)
        pyHDB_out.has_kochin = self._pyHDB_1.has_kochin
        if(pyHDB_out.has_kochin):
            self.merge_Kochin(pyHDB_out)

        # Mean wave drift loads.
        assert (self._pyHDB_1.has_Drift == self._pyHDB_2.has_Drift)
        pyHDB_out.has_Drift = self._pyHDB_1.has_Drift
        if (pyHDB_out.has_Drift):
            pyHDB_out.Wave_drift_force = np.zeros((6, pyHDB_out.nb_wave_freq, pyHDB_out.nb_wave_dir), dtype=np.float)
            assert (pyHDB_out.Wave_drift_force.shape[0] == self._pyHDB_1.Wave_drift_force.shape[0] == self._pyHDB_2.Wave_drift_force.shape[0]) # Dof.
            assert (pyHDB_out.Wave_drift_force.shape[2] == self._pyHDB_1.Wave_drift_force.shape[2] == self._pyHDB_2.Wave_drift_force.shape[2]) # Wave directions.
            pyHDB_out.Wave_drift_force = np.concatenate([self._pyHDB_1.Wave_drift_force, self._pyHDB_2.Wave_drift_force], axis=1) # Wave frequencies.

    def merge(self):

        """
            This method merges two pyHDB and returns the merged pyHDB.
        """

        # Initialization of the merged pyHDB.
        pyHDB_out = pyHDB.pyHDB()

        # Version.
        self.merge_version(pyHDB_out)

        # Commit hash.
        self.merge_commit_hash(pyHDB_out)

        # Expert numerical parameters.
        self.merge_numerical_parameters(pyHDB_out)

        # Environnement.
        self.merge_environment(pyHDB_out)

        # Discretization.
        self.merge_discretization(pyHDB_out)

        # Symmetries.
        self.merge_symmetries(pyHDB_out)

        # Vector fitting.
        self.merge_VF(pyHDB_out)

        # Bodies.
        self.merge_bodies(pyHDB_out)

        # Wave field.
        self.merge_wave_field(pyHDB_out)

        # Mean wave drift loads.
        self.merge_drift(pyHDB_out)

        return pyHDB_out

