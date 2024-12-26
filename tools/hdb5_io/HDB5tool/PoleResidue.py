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

"""Module to handling the poles and residues of the vector fitting."""

import numpy as np

class PoleResidue(object):

    """
        Class for handling the poles and the residues of the vector fitting.
    """

    def __init__(self):

        self.real_pole = []
        self.real_residue = []

        self.cc_pole = []
        self.cc_residue = []

    def add_real_pole_residue(self, pole, residue):

        """Add real poles and residues."""

        assert(pole.shape[0] == residue.shape[0])
        for i in range(pole.shape[0]):
            self.real_pole.append(pole[i])
            self.real_residue.append(residue[i])

        assert(len(self.real_pole) == len(self.real_residue))

    def add_cc_pole_residue(self, pole, residue):

        """Add complex poles and residues."""

        assert (pole.shape[0] == residue.shape[0])
        for i in range(pole.shape[0]):
            self.cc_pole.append(pole[i])
            self.cc_residue.append(residue[i])

        assert (len(self.cc_pole) == len(self.cc_residue))

    def order(self):
        """Order of the vector fitting."""
        return len(self.real_pole) + 2 * len(self.cc_pole)

    def nb_real_poles(self):
        """Order of the vector fitting."""
        return len(self.real_pole)

    def nb_cc_poles(self):
        """Order of the vector fitting."""
        return len(self.cc_pole)

    def real_poles(self):
        """Getter of the vector of real poles."""
        real_pole_tmp = np.zeros(self.nb_real_poles())
        for i in range(self.nb_real_poles()):
            real_pole_tmp[i] = self.real_pole[i]

        return real_pole_tmp

    def real_residues(self):
        """Getter of the vector of real residues."""
        real_residue_tmp = np.zeros(self.nb_real_poles())
        for i in range(self.nb_real_poles()):
            real_residue_tmp[i] = self.real_residue[i]

        return real_residue_tmp

    def cc_poles(self):
        """Getter of the vector of comples poles."""
        cc_pole_tmp = np.zeros(self.nb_cc_poles(), dtype = np.complex64)
        for i in range(self.nb_cc_poles()):
            cc_pole_tmp[i] = self.cc_pole[i]

        return cc_pole_tmp

    def cc_residues(self):
        """Getter of the vector of complex residues."""
        cc_residue_tmp = np.zeros(self.nb_cc_poles(), dtype = np.complex64)
        for i in range(self.nb_cc_poles()):
            cc_residue_tmp[i] = self.cc_residue[i]

        return cc_residue_tmp