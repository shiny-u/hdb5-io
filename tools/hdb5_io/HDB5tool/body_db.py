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

"""Module to create a body database for frydom hydrodynamic database."""

import numpy as np

from hdb5_io.HDB5tool.hydrostatic_db import HydrostaticDB
from hdb5_io.HDB5tool.inertia import Inertia


class BodyDB(object):

    """
        Class for writing the body data into the *.hdb5 file.
    """

    def __init__(self, i_body, nb_bodies, nw, nbeta, mesh = None):

        """
        Constructor of the class BodyDB.

        Parameters
        ----------
        i_body : int
            Index of the body.
        nb_bodies : int
            Number of bodies.
        nw : int
            Number of wave frequencies.
        nbeta : int
            Number of wave directions.
        mesh : Array of floats
            Mesh of the body.
        """

        # Index.
        self.i_body = i_body

        # Horizontal position in world (m, m, deg).
        self.horizontal_position = None

        # Computation point in body frame.
        self.computation_point = np.empty((1, 3), dtype=float)

        # Wave reference point in body frame.
        self.wave_reference_point_in_body_frame = None

        # Added mass matrices (6 dof so 6 rows x all the columns x all the frequencies).
        self.Added_mass = np.zeros((6, 6 * nb_bodies, nw), dtype = np.float64)

        # x-derivative of the added mass matrices (6 dof so 6 rows x all the columns x all the frequencies).
        self.Added_mass_x_derivative = None

        # Infinite-frequency added mass matrices.
        self.Inf_Added_mass = None

        # x-derivative of the infinite-frequency added mass matrices.
        self.Inf_Added_mass_x_derivative = None

        # Zero-frequency added mass matrices.
        self.Zero_Added_mass = None

        # x-derivative of the zero-frequency added mass matrices.
        self.Zero_Added_mass_x_derivative = None

        # Damping matrices (6 dof so 6 rows x all the columns x all the frequencies).
        self.Damping = np.zeros((6, 6 * nb_bodies, nw), dtype=np.float64)

        # x-derivative of the damping matrices (6 dof so 6 rows x all the columns x all the frequencies).
        self.Damping_x_derivative = None

        # Diffraction loads (6 dof so 6 rows x all the frequencies).
        self.Diffraction = np.zeros((6, nw, nbeta), dtype=np.complex64)

        # x-derivative of the diffraction loads (6 dof so 6 rows x all the frequencies).
        self.Diffraction_x_derivative = None

        # Froude-Krylov loads (6 dof so 6 rows x all the frequencies).
        self.Froude_Krylov = np.zeros((6, nw, nbeta), dtype=np.complex64)

        # x-derivative of the Froude-Krylov loads (6 dof so 6 rows x all the frequencies).
        self.Froude_Krylov_x_derivative = None

        # Mesh in the body frame.
        self.mesh = mesh

        # Body name (body mesh name until version 2).
        self.name = None

        # Force mask.
        self.Force_mask = np.zeros(6,dtype = np.int64)

        # Motion mask.
        self.Motion_mask = np.zeros(6,dtype = np.int64)

        # Radiation mask
        self.Radiation_mask = np.ones((6, 6 * nb_bodies), dtype = bool)

        # Product n*dS for the computation of the Froude-Krylov loads.
        self._nds = None

        # Point of computation of the moments (for Nemoh only).
        self.point = np.zeros((3,3),dtype = np.float64)

        # Impulse response functions without forward speed.
        self.irf = None

        # Impulse response functions proportional to the forward speed and without x-derivative.
        self.irf_ku = None

        # Impulse response functions proportional to the forward speed and with x-derivatives.
        self.irf_ku_x_derivative = None

        # Impulse response functions proportional to the the square of the forward speed.
        self.irf_ku2 = None

        # Flags (?).
        self._flags = np.ones((6, 6 * nb_bodies), dtype=bool)

        # Hydrostatics.
        self._hydrostatic = None

        # Inertia matrix.
        self._inertia = None

        # Mooring matrix.
        self._mooring = None

        # Linear extra damping matrix.
        self._extra_damping = None

        # RAO.
        self.RAO = None

        # Eigenfrequencies.
        self.Eigenfrequencies = None

        # Poles and residues.
        self.poles_residues = None

        # Energy spectral moments.
        self.EnergySpectralMoments = None

        # Diodore data.
        self.cog = None
        self.cob = None
        self.underwater_volume = None

        self.position = np.zeros(3)

        # Mean wave drift loads.
        self.has_Drift = False
        self.Wave_drift_force = None
        self.sym_x = False # Sym x.
        self.sym_y = False # Sym y.
        self.kochin_step = None # Kochin function angular step.

        return

    def _compute_nds(self):
        """Computes the term n dS for each force mode of the body."""

        self._nds = np.zeros((6, self.mesh.nb_faces), dtype=np.float64)

        areas = self.mesh.faces_areas
        normals = self.mesh.faces_normals
        centers = self.mesh.faces_centers

        for i in range(0,6):

            # Direction.
            direction = np.zeros(3)
            if (i == 0 or i == 3):
                direction[0] = 1 # ex.
            elif (i == 1 or i == 4):
                direction[1] = 1 # ey.
            elif (i == 2 or i == 5):
                direction[2] = 1 # ez.

            # n*ds.
            if i <= 2: # Force.
                self._nds[i, :] = areas * np.einsum('ij, j -> i', normals, direction)
            else: # Moment.
                am = centers - self.point[i-3,:]
                vel = np.cross(direction, am)
                self._nds[i, :] = areas * (normals * vel).sum(axis=1)

    def get_nds(self, iforce):
        """Get the array of ndS . direction vector for the specified force mode.

        Parameters
        ----------
        iforce : int
            Force mode index.

        Returns
        -------
        np.ndarray
            (n_faces) Array of ndS quantities.
        """

        if self._nds is None:
            self._compute_nds()

        return self._nds[iforce, :]

    def radiation_damping(self,iwcut):

        """This function gives the damping coefficients.

        Returns
        -------
        Array of floats
            Damping coefficients.
        """

        if iwcut is None:
            ca = self.Damping
        else:
            ca = self.Damping[:, :, :iwcut]

        return np.einsum('ijk, ij -> ijk', ca, self._flags)

    def radiation_damping_x_derivative(self,iwcut):

        """This function gives the x-derivative of the damping coefficients.

        Returns
        -------
        Array of floats
            Damping coefficients.
        """

        if iwcut is None:
            ca = self.Damping_x_derivative
        else:
            ca = self.Damping_x_derivative[:, :, :iwcut]

        return np.einsum('ijk, ij -> ijk', ca, self._flags)

    def radiation_added_mass(self, iwcut):

        """This function gives the added-mass coefficients.

        Returns
        -------
        Array of floats
            Added-mass coefficients.
        """

        if iwcut is None:
            cm = self.Added_mass
        else:
            cm = self.Added_mass[:, :, :iwcut]
        return np.einsum('ijk, ij -> ijk', cm, self._flags)

    def radiation_added_mass_x_derivative(self, iwcut):

        """This function gives the x-derivative of the added-mass coefficients."""

        if iwcut is None:
            cm = self.Added_mass_x_derivative
        else:
            cm = self.Added_mass_x_derivative[:, :, :iwcut]
        return np.einsum('ijk, ij -> ijk', cm, self._flags)

    def infinite_added_mass(self):

        """This function gives the infinite-frequency added mass coefficients.

        Returns
        -------
        Array of floats
            Infinite-frequency added mass coefficients.
        """

        if self.Inf_Added_mass is None:
            return
        else:
            return self.Inf_Added_mass * self._flags

    def infinite_added_mass_x_derivative(self):

        """This function gives the x-derivative of the infinite-frequency added mass coefficients.

        Returns
        -------
        Array of floats
            x-derivative of the infinite-frequency added mass coefficients.
        """

        if self.Inf_Added_mass_x_derivative is None:
            return
        else:
            return self.Inf_Added_mass_x_derivative * self._flags

    def zero_added_mass(self):

        """This function gives the zero-frequency added mass coefficients.

        Returns
        -------
        Array of floats
            Zero-frequency added mass coefficients.
        """

        if self.Zero_Added_mass is None:
            return
        else:
            return self.Zero_Added_mass * self._flags

    def zero_added_mass_x_derivative(self):

        """This function gives the x-derivative of the zero-frequency added mass coefficients.

        Returns
        -------
        Array of floats
            x-derivative of the zero-frequency added mass coefficients.
        """

        if self.Zero_Added_mass_x_derivative is None:
            return
        else:
            return self.Zero_Added_mass_x_derivative * self._flags

    @property
    def hydrostatic(self):

        """This function gives the hydrostatic data of the body.

        Returns
        -------
        HydrostaticDB
            Hydrostatic data of the body.
        """

        return self._hydrostatic

    def activate_hydrostatic(self):

        """This function initializes the hydrostatic parameters."""

        if(self._hydrostatic is None):
            self._hydrostatic = HydrostaticDB()

    @property
    def inertia(self):

        """This function gives the inertia data of the body.

        Returns
        -------
        Inertia
            inertia data of the body.
        """

        return self._inertia

    def activate_inertia(self):

        """This function initializes the inertia matrix."""

        if(self._inertia is None):
            self._inertia = Inertia()

    @property
    def mooring(self):

        """This function gives the mooring data of the body.

        Returns
        -------
        Mooring
            mooring data of the body.
        """

        return self._mooring

    def activate_mooring(self):

        """This function initializes the mooring matrix."""

        if (self._mooring is None):
            self._mooring = np.zeros((6, 6))

    @mooring.setter
    def mooring(self, value):

        """This function sets the mooring data of the body.

        Parameter
        ---------
        Mooring
            mooring data of the body.
        """

        self._mooring = value

    @property
    def extra_damping(self):

        """This function gives the extra linear damping data of the body.

        Returns
        -------
        Extra linear damping
            extra linear damping data of the body.
        """

        return self._extra_damping

    def activate_extra_damping(self):

        """This function initializes the extra linear damping matrix."""

        if (self._extra_damping is None):
            self._extra_damping = np.zeros((6, 6))

    @extra_damping.setter
    def extra_damping(self, value):

        """This function sets the extra linear damping data of the body.

        Parameter
        ---------
        Extra linear damping
            extra linear damping data of the body.
        """

        self._extra_damping = value

    def set_extra_roll_damping(self, value):
        """This method sets the extra roll damping."""

        self._extra_damping[3, 3] = value

    def extra_roll_damping(self):
        """This method returns the extra roll damping."""

        return self._extra_damping[3, 3]