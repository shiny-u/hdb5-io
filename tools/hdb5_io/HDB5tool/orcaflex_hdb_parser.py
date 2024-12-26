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

import yaml
import numpy as np
import hdb5_io.HDB5tool.body_db as body_db


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

def get_by_name(reader, name):

    for data in reader:
        if data["Name"] == name:
            return data

    return None


def reader_args(args, database):
    hdbfile = args.orcaflex_input[0]
    if len(args.orcaflex_input) > 1:
        structure_names = args.orcaflex_input[1:]
    else:
        structure_names = None
    reader(hdbfile, database, structure_names)

    return


def reader(hdbfile, database, structure_names):

    with open(hdbfile, 'r') as f:
        reader = yaml.safe_load(f)

    # General
    unit = reader["General"]["UnitsSystem"]
    if unit == "SI":
        unit_mass = 1000. # tons
        unit_force = 1000. # kN
        unit_time = 1. # seconds
    else:
        print("error : General unit unknown. Must be SI")
        # US unit not implemented
        exit(1)

    database._pyHDB.solver = "OrcaFlex"

    # Environment
    environment = reader["Environment"]
    database._pyHDB.rho_water = environment["Density"] * 1000
    database._pyHDB.grav = 9.81
    database._pyHDB.depth = environment["WaterDepth"]

    # Group
    if structure_names is None:
        structure_names = reader["Groups"]["Structure"].keys()
    nb_bodies = len(structure_names)
    database._pyHDB.nb_bodies = nb_bodies

    # List multibody Groups (if present)

    multibody = {}

    for structure_name in structure_names:
        vessel = get_by_name(reader["Vessels"], structure_name)
        multibody[vessel["VesselType"]] = None

    if "MultibodyGroups" in reader:
        for group in reader["MultibodyGroups"]:
            group_name = group["Name"]
            for i_body, body in enumerate(group["Bodies"]):
                body_name = body["Name"]
                body_type = body["VesselType"]
                multibody[body_type] = {"group": group_name, "id": i_body+1}
        multibody_mat = np.full((i_body+1, i_body+1), False)
    else:
        multibody_mat = None

    # Get structure

    for i_structure, structure_name in enumerate(structure_names):

        vessel = get_by_name(reader["Vessels"], structure_name)
        vessel_type = get_by_name(reader["VesselTypes"], vessel["VesselType"])
        vessel_type_name = vessel["VesselType"]
        draught = get_by_name(vessel_type["Draughts"], vessel["Draught"])

        if vessel_type["RAOPhaseUnitsConvention"] == "degrees":
            c_phase = np.pi / 180.
        else:
            c_phase = 1.

        # Waves frequencies
        waves_freq = np.array([])

        if multibody[vessel_type_name] is None:
            for data in draught["FrequencyDependentAddedMassAndDamping"]:
                freq = data["AMDPeriodOrFrequency"]
                if freq > 1E-06:
                    waves_freq = np.append(waves_freq, freq)
        else:
            group = get_by_name(reader["MultibodyGroups"], multibody[vessel_type_name]["group"])
            for data in group["MultibodyAddedMassAndDamping"]:
                freq = data["AMDPeriodOrFrequency"]
                if freq > 1E-6:
                    waves_freq = np.append(waves_freq, freq)

        if vessel_type["WavesReferredToBy"] == "period (s)":
            waves_freq = 2*np.pi/waves_freq

        nb_wave_freq = waves_freq.size

        # Waves directions
        waves_dir = np.array([])
        for data in draught["LoadRAOs"]["RAOs"]:
            waves_dir = np.append(waves_dir, data["RAODirection"])

        waves_dir *= np.pi / 180.

        nb_wave_dir = waves_dir.size

        # BEM Body
        body = body_db.BodyDB(i_structure, nb_bodies, nb_wave_freq, nb_wave_dir)

        body.name = structure_name
        body.cog = draught["CentreOfMass"]
        body.wave_reference_point_in_body_frame = draught["LoadRAOs"]["RAOOrigin"]

        # Inertia
        body.activate_inertia()
        body.inertia.mass = float(draught["Mass"]) * unit_mass
        mat33 = unit_mass * np.array(draught["MomentOfInertiaTensorX, MomentOfInertiaTensorY, MomentOfInertiaTensorZ"], dtype=float)
        body.inertia.diagonal = np.diagonal(mat33)
        body.inertia.non_diagonal = np.array([mat33[0, 1], mat33[0, 2], mat33[1, 2]])

        # Excitation
        i_dir = 0
        for data in draught["LoadRAOs"]["RAOs"]:
            mat = np.array(data["RAOPeriodOrFrequency, RAOSurgeAmp, RAOSurgePhase, RAOSwayAmp, RAOSwayPhase, RAOHeaveAmp, RAOHeavePhase, RAORollAmp, RAORollPhase, RAOPitchAmp, RAOPitchPhase, RAOYawAmp, RAOYawPhase"], dtype=float)
            for i_dof in range(6):
                body.Froude_Krylov[i_dof, :, i_dir] = unit_force * mat[:, 1+2*i_dof] * np.exp(-1j * mat[:, 2+2*i_dof] * c_phase)
            i_dir += 1

        # RAO
        body.RAO = np.empty((6, nb_wave_freq, nb_wave_dir), dtype=complex)
        i_dir = 0
        for data in draught["DisplacementRAOs"]["RAOs"]:
            mat = np.array(data["RAOPeriodOrFrequency, RAOSurgeAmp, RAOSurgePhase, RAOSwayAmp, RAOSwayPhase, RAOHeaveAmp, RAOHeavePhase, RAORollAmp, RAORollPhase, RAOPitchAmp, RAOPitchPhase, RAOYawAmp, RAOYawPhase"], dtype=float)
            for i_dof in range(6):
                body.RAO[i_dof, :, i_dir] = mat[:, 1+2*i_dof] * np.exp(-1j * mat[:, 2+2*i_dof] * c_phase)
            i_dir += 1
        database._pyHDB.has_RAO = True

        # Wave Drift
        body.Wave_drift_force = np.empty((6, nb_wave_freq, nb_wave_dir), dtype=float)
        i_dir = 0
        for data in draught["WaveDrift"]["RAOs"]:
            mat = unit_force * np.array(data["RAOPeriodOrFrequency, RAOSurgeAmp, RAOSwayAmp, RAOHeaveAmp, RAORollAmp, RAOPitchAmp, RAOYawAmp"], dtype=float)
            for i_dof in range(6):
                body.Wave_drift_force[i_dof, :, i_dir] = mat[:, 1+i_dof]
            i_dir += 1
        body.has_Drift = True

        # Group dependent data

        if multibody[vessel_type_name] is None:
            # Buoyancy
            body.cob = draught["CentreOfBuoyancy"]
            body.computation_point = draught["ReferenceOrigin"]
            # Hydrostatics
            body.activate_hydrostatic()
            body.hydrostatic.matrix = unit_force * np.array(draught["HydrostaticStiffnessz, HydrostaticStiffnessRx, HydrostaticStiffnessRy"], dtype=float)
            # Added Mass and Damping
            i_data = 0
            for data in draught["FrequencyDependentAddedMassAndDamping"]:
                freq = data["AMDPeriodOrFrequency"]
                if freq > 1E-06:
                    body.Added_mass[:, :, i_data] = unit_force * np.array(data["AddedMassMatrixX, AddedMassMatrixY, AddedMassMatrixZ, AddedMassMatrixRx, AddedMassMatrixRy, AddedMassMatrixRz"], dtype=float).transpose()
                    body.Damping[:, :, i_data] = unit_force * np.array(data["DampingX, DampingY, DampingZ, DampingRx, DampingRy, DampingRz"], dtype=float).transpose()
                    i_data += 1
        else:
            i_body_data = multibody[vessel_type_name]["id"]
            group = get_by_name(reader["MultibodyGroups"], multibody[vessel_type_name]["group"])
            body_data = group["Bodies"][i_body_data-1]
            # Buoyancy
            body_cob = body_data["CentreOfBuoyancy"]
            body.computation_point =  body_data["ReferenceOrigin"]
            # Hydrostatics
            body.activate_hydrostatic()
            body.hydrostatic.matrix = unit_force * np.array(body_data["HydrostaticStiffnessz, HydrostaticStiffnessRx, HydrostaticStiffnessRy"], dtype=float)
            # Added Mass and Damping
            i_data = 0
            for data in group["MultibodyAddedMassAndDamping"]:
                freq = data["AMDPeriodOrFrequency"]
                if freq > 1E-6:
                    for mat_data in data["Matrices"]:
                        if i_body_data == mat_data["Row"]:
                            i_col = mat_data["Column"]-1
                            try:
                                body.Added_mass[:, 6*i_col:6*(i_col+1), i_data] = unit_force * np.array(mat_data["AddedMassX, AddedMassY, AddedMassZ, AddedMassRx, AddedMassRy, AddedMassRz"], dtype=float)
                                body.Damping[:, 6*i_col:6*(i_col+1), i_data] = unit_force * np.array(mat_data["DampingX, DampingY, DampingZ, DampingRx, DampingRy, DampingRz"], dtype=float)
                                multibody_mat[i_structure, i_col] = True
                            except:
                                print("error : reading added mass and damping for freq: {}, row: {}, col: {}".format(freq, i_body_data, i_col))
                                exit(1)
                    i_data += 1

        # TODO : ajouter matrice amortissement supplémentaire

        # Sorting
        body.Added_mass = body.Added_mass[:, :, np.argsort(waves_freq)]

        body.Damping = body.Damping[:, :, np.argsort(waves_freq)]

        body.Froude_Krylov = body.Froude_Krylov[:, np.argsort(waves_freq), :]
        body.Froude_Krylov = body.Froude_Krylov[:, :, np.argsort(waves_dir)]

        body.RAO = body.RAO[:, np.argsort(waves_freq), :]
        body.RAO = body.RAO[:, :, np.argsort(waves_dir)]

        body.Wave_drift_force = body.Wave_drift_force[:, np.argsort(waves_freq), :]
        body.Wave_drift_force = body.Wave_drift_force[:, :, np.argsort(waves_dir)]

        waves_freq = waves_freq[np.argsort(waves_freq)]
        waves_dir = waves_dir[np.argsort(waves_dir)]

        # Set wave frequencies
        database._pyHDB.nb_wave_freq = nb_wave_freq
        database._pyHDB.min_wave_freq = np.min(waves_freq)
        database._pyHDB.max_wave_freq = np.max(waves_freq)
        database._pyHDB.wave_freq = waves_freq

        database._pyHDB.discrete_frequency = database._pyHDB.wave_freq

        # Set wave direction
        database._pyHDB.nb_wave_dir = nb_wave_dir
        database._pyHDB.min_wave_dir = np.min(waves_dir)
        database._pyHDB.max_wave_dir = np.max(waves_dir)
        database._pyHDB.wave_dir = waves_dir

        # Mask
        body.Motion_mask = np.ones(6, dtype = np.int64)
        body.Force_mask = np.ones(6, dtype = np.int64)

        # Add Body
        database._pyHDB.append(body)

    # Complete extra-diagonal added mass / damping matrix if not defined in yml file

    if multibody_mat is not None:

        n = multibody_mat.shape[0]

        for i_row in range(n):
            for i_col in range(n):

                is_defined = multibody_mat[i_row, i_col]

                if not is_defined:

                    n_data = database._pyHDB.bodies[i_row].Added_mass[:, :, :].shape[2]

                    for i_data in range(n_data):

                        database._pyHDB.bodies[i_row].Added_mass[:, 6*i_col:6*(i_col+1), i_data] =\
                            database._pyHDB.bodies[i_col].Added_mass[:, 6*i_row:6*(i_row+1), i_data].transpose()

                        database._pyHDB.bodies[i_row].Damping[:, 6*i_col:6*(i_col+1), i_data] =\
                            database._pyHDB.bodies[i_col].Damping[:, 6*i_row:6*(i_row+1), i_data].transpose()

    # Infinite added mass and impulse response function
    set_time_discretization(database._pyHDB)
    database._pyHDB.eval_impulse_response_function()
    database._pyHDB.eval_infinite_added_mass()

    database._is_initialized = True

    return
