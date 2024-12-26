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

import re
import sys, os
from math import *

import numpy as np
import hdb5_io.HDB5tool.body_db as body_db
from hdb5_io.HDB5tool.inertia import Inertia


class DiodoreReader:
    __float_pattern = r'[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[Ee][+-]?\d+)?'
    __one_value_pattern_base = r'\[%(TAG)s\]\s*(.+)'
    __vector_pattern_base = r'\[%(TAG)s\]\s+(.+?)\['
    __matrix3D_pattern_base = r'\[%(TAG)s\]\s*([+-]?(?:\d+\.\d*|\d*\.\d+)(?:[Ee][+-]?\d+)?)(.+?)\['
    __structure_pattern = r'\[%(TAG)s\](.+)\[STRUCTURE_'
    __coupling_structure_pattern = r'\[HYDRODYNAMIC_COUPLING_BETWEEN_STRUCTURES\]\s*%(TAG1)s\s*\[ON\]\s*%(TAG2)s\s*' \
                                   r'\[Coupled_Added_mass_Radiation_Damping\](.+)\[(HYDRODYNAMIC|STRUCTURE)'

    def __init__(self, hdb_file: str):
        with open(hdb_file) as f:
            self._data = f.read()
            self._data += '\n[STRUCTURE_END]'

    def get_val(self, TAG: str, _type):
        return _type(re.search(self.__one_value_pattern_base % {"TAG": TAG}, self._data).groups()[0])

    def get_vector(self, TAG: str):
        res = re.search(self.__vector_pattern_base % {"TAG": TAG}, self._data, re.DOTALL).groups()[0]
        return np.asarray(list(map(float, res.split())))

    def get_matrix(self, TAG: str, shape: tuple):
        res = re.search(self.__vector_pattern_base % {"TAG": TAG}, self._data, re.DOTALL).groups()[0]
        return np.asarray(list(map(float, res.split()))).reshape(shape)

    def get_3Dmatrix(self, TAG: str, periods: np.ndarray, headings: np.ndarray):
        mat_3D = np.zeros((headings.size, periods.size, 6))

        for idx, heading in enumerate(headings):
            TAG_tmp = "{}_{:0>3}".format(TAG, idx + 1)

            pattern = self.__matrix3D_pattern_base % ({"TAG": TAG_tmp})

            res = re.search(pattern, self._data, re.DOTALL).groups()
            assert float(res[0]) == heading

            mat = np.asarray(list(map(float, res[1].split()))).reshape((periods.size, 7))
            assert np.allclose(mat[:, 0], periods)

            mat_3D[idx, :, :] = mat[:, 1:]

        return mat_3D

    def get_added_mass_damp(self, TAG: str, periods: np.ndarray, headings: np.ndarray):
        mat_3D = np.zeros((headings.size, periods.size, 6))

        for idx, heading in enumerate(headings):
            TAG_tmp = "{}_{}".format(TAG, idx + 1)

            res = re.search(self.__vector_pattern_base % {"TAG": TAG_tmp}, self._data, re.DOTALL).groups()[0]

            mat = np.asarray(list(map(float, res.split()))).reshape((periods.size, 7))
            assert np.allclose(mat[:, 0], periods)

            mat_3D[idx, :, :] = mat[:, 1:]

        return mat_3D

    def get_structure(self, TAG: str):
        res = re.search(self.__structure_pattern % {"TAG": TAG}, self._data, re.DOTALL).groups()[0]
        return DiodoreStructureReader(res)

    def get_coupling_structure(self, TAG1: str, TAG2: str):
        res = re.search(self.__coupling_structure_pattern % {"TAG1": TAG1, "TAG2": TAG2}, self._data, re.DOTALL).groups()[0]
        return DiodoreStructureReader(res)


class DiodoreStructureReader(DiodoreReader):

    def __init__(self, data: str):
        self._data = data
        self._data += '\n[STRUCTURE_END]'


def str2bool(s: str):
    if s.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif s.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        print("error : boolean value must be in : ['true', '1', 't', 'y', 'yes'] or ['false', '0', 'f', 'n', 'no']")
        exit(1)

def set_time_discretization(pyHDB):

    dw = np.abs(np.mean(np.diff(pyHDB.omega)))
    pyHDB._final_time = np.pi / (dw)
    pyHDB._delta_time = 0.05

    pyHDB._nb_time_sample = int(pyHDB._final_time / pyHDB._delta_time) + 1
    pyHDB._delta_time = pyHDB._final_time / float(pyHDB._nb_time_sample -1)

    pyHDB.time = np.linspace(start=0, stop=pyHDB._final_time, num=pyHDB._nb_time_sample)
    pyHDB.dt = pyHDB._delta_time
    pyHDB.nb_time_samples = pyHDB._nb_time_sample

    return


def reader_args(args, database):

    hdbfile = args.diodore_input[0]
    reader(hdbfile, database)
    return


def reader(hdbfile, database):

    database._pyHDB.rho_water = 1025.0
    database._pyHDB.grav = 9.81
    database._pyHDB.depth = 600

    reader = DiodoreReader(hdbfile)

    print("Read DIODORE HDB file : %s" % hdbfile)

    try:
        soft = reader.get_val("SOFT", str)
    except:
        print("error : File format. Unable to get 'SOFT' from {}. File is not a DIODORE HDB file.")
        exit(1)

    database._pyHDB.solver = soft

    version = reader.get_val("VERSION", str)
    print("version : %s" % version)

    is_hydro_para = str2bool(reader.get_val("HYDRO_PARA", str))
    is_detail_exc = str2bool(reader.get_val("DETAIL_EXC", str))
    is_rao = str2bool(reader.get_val("RAO", str))
    is_drift_force = str2bool(reader.get_val("DRIFT_FORCE", str))
    is_qtf = str2bool(reader.get_val("QTF", str))

    # Wave frequencies
    n_periods = reader.get_val("PERIODS_NUMBER", int)
    periods = reader.get_vector("List_calculated_periods")
    assert periods.size == n_periods

    omega = 2. * pi / periods
    omega = np.flip(omega, 0)

    database._pyHDB.nb_wave_freq = omega.size
    database._pyHDB.min_wave_freq = np.min(omega)
    database._pyHDB.max_wave_freq = np.max(omega)
    database._pyHDB.wave_freq = omega

    # Wave headings
    n_headings = reader.get_val("HEADINGS_NUMBER", int)
    headings = reader.get_vector("List_calculated_headings")
    assert headings.size == n_headings

    database._pyHDB.nb_wave_dir = headings.size
    database._pyHDB.min_wave_dir = np.min(headings)
    database._pyHDB.max_wave_dir = np.max(headings)
    database._pyHDB.wave_dir = headings * pi / 180.

    database._pyHDB.nb_bodies = reader.get_val("STRUCTURES_NUMBER", int)

    # Symetries

    database._pyHDB.bottom_sym = False
    database._pyHDB.xoz_sym = False
    database._pyHDB.yoz_sym = False
    database._pyHDB.sym_x = False
    database._pyHDB.sym_y = False
    database._pyHDB.discrete_frequency = database._pyHDB.wave_freq

    for i_body in range(database._pyHDB.nb_bodies):

        body = body_db.BodyDB(i_body, database._pyHDB.nb_bodies, database._pyHDB.nb_wave_freq, database._pyHDB.nb_wave_dir)

        body.name = reader.get_val("STRUCTURE_{:0>2}".format(i_body+1), str).strip()

        structure = reader.get_structure("STRUCTURE_{:0>2}".format(i_body+1))

        body.cog = structure.get_vector("CENTER_OF_GRAVITY") # FIXME : vérifier par rarpport à que repère (body frame / world frame)
        body.cob = structure.get_vector("CENTER_OF_BUOYANCY") # FIXME : vérifier par rarpport à que repère (body frame / world frame)
        body.computation_point = body.cog
        body.wave_reference_point_in_body_frame = - body.cog # FIXME : vérifier par rarpport à que repère (body frame / world frame)

        mass_matrix = structure.get_matrix("Mass_Inertia_matrix", (6, 6))
        body.activate_inertia()
        body.inertia.matrix = mass_matrix

        hst_matrix = structure.get_matrix("Hydrostatic_matrix", (6, 6))
        body.activate_hydrostatic()
        body.hydrostatic.matrix = hst_matrix

        moor_matrix = structure.get_matrix("Stiffness_matrix_of_the_mooring_system", (6, 6))
        body.activate_mooring()
        body.mooring = moor_matrix

        #EFM_MOD = reader.get_3Dmatrix("INCIDENCE_EFM_MOD", periods, headings)
        #EFM_PH = reader.get_3Dmatrix("INCIDENCE_EFM_PH", periods, headings)
        #EFM = EFM_MOD * np.exp(1j * EFM_PH)

        if is_detail_exc:
            EFM_FK_MOD = structure.get_3Dmatrix("INCIDENCE_EFM_FFK_MOD", periods, headings)
            EFM_FK_PH = structure.get_3Dmatrix("INCIDENCE_EFM_FFK_PH", periods, headings)
            EFM_FK = EFM_FK_MOD * np.exp(1j * (EFM_FK_PH + 0.5*pi))
            EFM_FK = np.flip(EFM_FK, 1)
            body.Froude_Krylov = np.transpose(EFM_FK, (2, 1, 0))

            EFM_DIFF_MOD = structure.get_3Dmatrix("INCIDENCE_EFM_DIFF_MOD", periods, headings)
            EFM_DIFF_PH = structure.get_3Dmatrix("INCIDENCE_EFM_DIFF_PH", periods, headings)
            EFM_DIFF = EFM_DIFF_MOD * np.exp(1j * (EFM_DIFF_PH + 0.5*pi))
            EFM_DIFF = np.flip(EFM_DIFF, 1)
            body.Diffraction = np.transpose(EFM_DIFF, (2, 1, 0))

        if is_rao:
            RAO_MOD = structure.get_3Dmatrix("INCIDENCE_RAO_MOD", periods, headings)
            RAO_MOD[:, :, 3:] *= np.pi / 180.  # deg -> rad for rotation
            RAO_PH = structure.get_3Dmatrix("INCIDENCE_RAO_PH", periods, headings)
            RAO = RAO_MOD * np.exp(1j * (RAO_PH + 90.) * np.pi / 180.)
            RAO = np.flip(RAO, 1)
            body.RAO = np.transpose(RAO, (2, 1, 0))
        else:
            body.RAO = None

        if is_hydro_para:
            ADD_MASS = structure.get_added_mass_damp("ADDED_MASS_LINE", periods, np.arange(1, 7, 1))
            ADD_MASS = np.flip(ADD_MASS, 1)
            body.Added_mass[:, i_body * 6:(i_body + 1)*6, :] = np.moveaxis(ADD_MASS, -1, 1)

            DAMP = structure.get_added_mass_damp("DAMPING_TERM", periods, np.arange(1, 7, 1))
            DAMP = np.flip(DAMP, 1)
            body.Damping[:, i_body * 6:(i_body + 1) * 6, :] = np.moveaxis(DAMP, -1, 1)

        if is_drift_force and is_hydro_para:
            # FIXME : adapter la structure de wave drift force de pyHDB pour permettre de charger un effort de dérive par corps
            DFM = reader.get_3Dmatrix("INCIDENCE_DFM", periods, headings)
            DFM = np.transpose(DFM, (2, 1, 0))
            DFM = np.flip(DFM, 1)

            body.Wave_drift_force = np.zeros((6, database._pyHDB.nb_wave_freq, database._pyHDB.nb_wave_dir), dtype=np.float64)
            for i_beta in range(database._pyHDB.nb_wave_dir):
                body.Wave_drift_force[0, :, i_beta] = DFM[0, :, i_beta]
                body.Wave_drift_force[1, :, i_beta] = DFM[1, :, i_beta]
                body.Wave_drift_force[5, :, i_beta] = DFM[5, :, i_beta]

            body.has_Drift = True
        else:
            body.has_drift = False

        database._pyHDB.append(body)

    # Coupling terms in Added mass and damping
    if is_hydro_para and database._pyHDB.nb_bodies > 1:
        for i_body_1, body_1 in enumerate(database._pyHDB.bodies):
            for i_body_2, body_2 in enumerate(database._pyHDB.bodies):
                if i_body_1 != i_body_2:
                    coupling_structure = reader.get_coupling_structure(body_1.name, body_2.name) # ORIGINAL
                    #coupling_structure = reader.get_coupling_structure(body_2.name, body_1.name) # FIXME : il semble devoir intervertir les noms par rapport à la version original pour des coefficients cohérents
                    ADD_MASS = coupling_structure.get_added_mass_damp("ADDED_MASS_LINE", periods, np.arange(1, 7, 1))
                    ADD_MASS = np.flip(ADD_MASS, 1)
                    body_2.Added_mass[:, i_body_1 * 6:(i_body_1 + 1)*6, :] = np.moveaxis(ADD_MASS, -1, 1)

                    DAMP = coupling_structure.get_added_mass_damp("DAMPING_TERM", periods, np.arange(1, 7, 1))
                    DAMP = np.flip(DAMP, 1)
                    body_2.Damping[:, i_body_1 * 6:(i_body_1 + 1) * 6, :] = np.moveaxis(DAMP, -1, 1)

    # Infinite added masses

    set_time_discretization(database._pyHDB)

    if is_hydro_para:
        database._pyHDB.eval_impulse_response_function()
        database._pyHDB.eval_infinite_added_mass()

    if is_qtf:
        print("warning : QTF are present in DIODORE HDB file but will not be saved in hdb5 file")

    database._pyHDB.has_RAO = is_rao
    database._is_initialized = True

    return


def main():
    # Pour exemple de fonctionnement

    hdbfile = "CUBE1Seul.HDB"

    reader = DiodoreReader(hdbfile)

    # Lecture d'une valeur seule
    soft = reader.get_val("SOFT", str)

    n_periods = reader.get_val("PERIODS_NUMBER", int)
    n_headings = reader.get_val("HEADINGS_NUMBER", int)

    # Lecture d'un vecteur
    headings = reader.get_vector("List_calculated_headings")
    assert headings.size == n_headings

    periods = reader.get_vector("List_calculated_periods")
    assert periods.size == n_periods

    # Lecture d'un matrice
    mass_matrix = reader.get_matrix("Mass_Inertia_matrix", (6, 6))
    hst_matrix = reader.get_matrix("Hydrostatic_matrix", (6, 6))
    moor_matrix = reader.get_matrix("Stiffness_matrix_of_the_mooring_system", (6, 6))

    # Excitation
    EFM_MOD = reader.get_3Dmatrix("INCIDENCE_EFM_MOD", periods, headings)
    EFM_PH = reader.get_3Dmatrix("INCIDENCE_EFM_PH", periods, headings)
    EFM = EFM_MOD * np.exp(1j * EFM_PH)

    # Froude-Krylov
    EFM_FK_MOD = reader.get_3Dmatrix("INCIDENCE_EFM_FFK_MOD", periods, headings)
    EFM_FK_PH = reader.get_3Dmatrix("INCIDENCE_EFM_FFK_PH", periods, headings)
    EFM_FK = EFM_FK_MOD * np.exp(1j * EFM_FK_PH)

    # Diffraction
    EFM_DIFF_MOD = reader.get_3Dmatrix("INCIDENCE_EFM_DIFF_MOD", periods, headings)
    EFM_DIFF_PH = reader.get_3Dmatrix("INCIDENCE_EFM_DIFF_PH", periods, headings)
    EFM_DIFF = EFM_DIFF_MOD * np.exp(1j * EFM_DIFF_PH)

    # RAO
    RAO_MOD = reader.get_3Dmatrix("INCIDENCE_RAO_MOD", periods, headings)
    RAO_PH = reader.get_3Dmatrix("INCIDENCE_RAO_PH", periods, headings)
    RAO = RAO_MOD * np.exp(1j * RAO_PH)

    # Wave Drift
    DFM = reader.get_3Dmatrix("INCIDENCE_DFM", periods, headings)

    # Added Mass
    ADD_MASS = reader.get_added_mass_damp("ADDED_MASS_LINE", periods, np.arange(1, 7, 1))
    DAMP = reader.get_added_mass_damp("DAMPING_TERM", periods, np.arange(1, 7, 1))


if __name__ == '__main__':
    main()
