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

import os
import numpy as np
from math import *
import datetime

import hdb5_io.HDB5tool.body_db as body_db

def bool2str(value):
    if value:
        return "Y"
    else:
        return "N"


class DiodoreHDBWriter(object):

    def __init__(self, pyHDB, hdb_file):

        self._pyHDB = pyHDB

        self._set_periods()
        self._set_headings()

        with open(hdb_file, 'w') as writer:

            self._write_header(writer)

            self._write_periods(writer)
            self._write_headings(writer)

            nb_bodies = self._pyHDB.nb_bodies

            writer.write("[STRUCTURES_NUMBER]       %i\n" % nb_bodies)

            for i in range(nb_bodies):
                body = self._pyHDB.bodies[i]
                self._write_structure(writer, body)

            if nb_bodies > 1:
                for i in range(nb_bodies):
                    body_1 = self._pyHDB.bodies[i]
                    for j in range(i+1, nb_bodies):
                        body_2 = self._pyHDB.bodies[j]
                        self._write_added_mass_damping_body_coupling(writer, body_1, body_2)

        return

    def _write_header(self, writer):

        date = datetime.datetime.now()

        writer.write("[SOFT]              DIODORE\n")
        writer.write("[VERSION]           4.1.5\n")
        writer.write("[Date]              %s\n" % date.strftime("%X %a %b %d %Y"))
        writer.write("[INPUT_FILE]\n")
        writer.write("[Locally_At]\n")
        writer.write("[UNIT]\n")
        writer.write("[FORWARD_SPEED]     %5.2f\n" % 0.0)
        writer.write("[HYDRO_PARA]        %s\n" % bool2str(True))
        writer.write("[DETAIL_EXC]        %s\n" % bool2str(False))
        writer.write("[RAO]               %s\n" % bool2str(self._pyHDB.has_RAO))
        writer.write("[DRIFT_FORCE]       %s\n" % bool2str(self._pyHDB.bodies[0].Wave_drift_force is not None))
        writer.write("[QTF]               %s\n" % bool2str(False))
        writer.write("[PERIODS_NUMBER]    %i\n" % self._pyHDB.nb_wave_freq)
        writer.write("[INTER_PERIODS_NB]  %i\n" % self.nb_interp_periods)
        writer.write("[HEADINGS_NUMBER]   %i\n" % self._pyHDB.nb_wave_dir)
        writer.write("[LOWEST_HEADING]    %5.2f\n" % self._pyHDB.min_wave_dir)
        writer.write("[HIGHEST_HEADING]   %5.2f\n" % self._pyHDB.max_wave_dir)

        return

    def _write_periods(self, writer):

        writer.write("[List_calculated_periods]\n")
        for value in self.periods:
            writer.write("%8.3f\n" % value)

        return

    def _write_headings(self, writer):

        writer.write("[List_calculated_headings]\n")
        for value in self.headings:
            writer.write("%8.3f\n" % value)

        return

    def _write_structure(self, writer, body):

        # Name
        self._write_name(writer, body)

        # Underwater volume
        self._write_underwater_volume(writer, body)

        # Center of buoyancy
        self._write_center_of_buoyancy(writer, body)

        # Center of gravity
        self._write_center_of_gravity(writer, body)

        # Mass Inertia
        self._write_inertia(writer, body)

        # Hydrostatic
        self._write_hydrostatic(writer, body)

        # Stiffness
        if body.mooring is None:
            body.mooring = np.zeros((6, 6))
        self._write_stiffness(writer, body)

        # Blank line
        writer.write("\n")

        # Excitation force
        self._write_excitation_force(writer, body)

        # Response Amplitude Operator
        if self._pyHDB.has_RAO:
            self._write_response_amplitude_operator(writer, body)
            #if self.nb_interp_periods > 0:
            #    self._write_inter_rao(writer, body)

        # Drift force and moment
        if body.Wave_drift_force is not None:
            self._write_drift_force(writer, body)

        # Added mass and damping
        writer.write("[Added_mass_Radiation_Damping]\n")
        self._write_added_mass(writer, body)
        self._write_damping(writer, body)

        return

    def _write_name(self, writer, body):
        writer.write("[STRUCTURE_%02d]     %s\n" % (body.i_body+1, body.name))
        return

    def _write_underwater_volume(self, writer, body):
        writer.write("[UNDERWATER_VOLUME]      %8.4f\n" % body.underwater_volume)
        return

    def _write_center_of_buoyancy(self, writer, body):
        writer.write("[CENTER_OF_BUOYANCY]      %8.4f    %8.4f    %8.4f\n" % (body.cob[0], body.cob[1], body.cob[2]))
        return

    def _write_center_of_gravity(self, writer, body):
        writer.write("[CENTER_OF_GRAVITY]      %8.4f    %8.4f    %8.4f\n" % (body.cog[0], body.cog[1], body.cog[2]))
        return

    def _write_inertia(self, writer, body):
        writer.write("[Mass_Inertia_matrix]\n")
        for i in range(6):
            for j in range(6):
                writer.write("%.7E  " % body.inertia.matrix[i, j])
            writer.write("\n")
        return

    def _write_hydrostatic(self, writer, body):
        writer.write("[Hydrostatic_matrix]\n")
        for i in range(6):
            for j in range(6):
                writer.write("%.7E  " % body.hydrostatic.matrix[i, j])
            writer.write("\n")
        return

    def _write_stiffness(self, writer, body):
        writer.write("[Stiffness_matrix_of_the_mooring_system]\n")
        for i in range(6):
            for j in range(6):
                writer.write("%.7E  " % body.mooring[i, j])
            writer.write("\n")
        return

    def _write_excitation_force(self, writer, body):

        excitation_force = self.get_excitation_force(body)
        excitation_force_mod = np.abs(excitation_force)
        excitation_force_phase = np.angle(excitation_force, deg=True)

        writer.write("[EXCITATION_FORCES_AND_MOMENTS]\n")

        for i in range(self._pyHDB.nb_wave_dir):
            writer.write("[INCIDENCE_EFM_MOD_%03d]    %8.3f\n" % (i+1, self.headings[i]))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % excitation_force_mod[i, j, k])
                writer.write("\n")

        for i in range(self._pyHDB.nb_wave_dir):
            writer.write("[INCIDENCE_EFM_PH_%03d]     %8.3f\n" % (i+1, self.headings[i]))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % excitation_force_phase[i, j, k])
                writer.write("\n")

        return

    def _write_response_amplitude_operator(self, writer, body):

        rao = self.get_rao(body)
        rao_mod = np.abs(rao)
        rao_phase = np.angle(rao, deg=True)

        writer.write("[RAO]\n")

        for i in range(self._pyHDB.nb_wave_dir):
            writer.write("[INCIDENCE_RAO_MOD_%03d]      %8.3f\n" % (i+1, self.headings[i]))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % rao_mod[i, j, k])
                writer.write("\n")

        for i in range(self._pyHDB.nb_wave_dir):
            writer.write("[INCIDENCE_RAO_PH_%03d]       %8.3f\n" % (i+1, self.headings[i]))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % rao_phase[i, j, k])
                writer.write("\n")

        return

    def _write_inter_rao(self, writer, body):

        rao = self.get_rao(body)
        self.rao_mod_interp = np.abs(rao)
        self.rao_phase_interp = np.angle(rao, deg=True)

        writer.write("[INTER_RAO]\n")

        for i in range(self._pyHDB.nb_wave_dir):
            writer.write("[INCIDENCE_INTER_RAO_MOD_%03d]        %8.3f\n" % (i+1, self.headings[i]))
            for j in range(self.nb_interp_periods):
                writer.write("%8.3f  " % self.inter_periods[j])
                for k in range(6):
                    writer.write("%.7E  " % self.rao_mod_interp[i, j, k])
                writer.write("\n")

        for i in range(self._pyHDB.nb_wave_dir):
            writer.write("[INCIDENCE_INTER_RAO_PH_%03d]         %8.3f\n" % (i+1, self.headings[i]))
            for j in range(self.nb_interp_periods):
                writer.write("%8.3f  " % self.inter_periods[j])
                for k in range(6):
                    writer.write("%.7E  " % self.rao_phase_interp[i, j, k])
                writer.write("\n")

        return

    def _write_drift_force(self, writer, body):

        drift_force = body.Wave_drift_force

        drift_force = np.transpose(drift_force, (2, 1, 0))
        drift_force = np.flip(drift_force, 1)

        writer.write("[DRIFT_FORCES_AND_MOMENTS]\n")

        for i in range(self._pyHDB.nb_wave_dir):
            writer.write("[INCIDENCE_DFM_%03d]       %8.3f\n" % (i+1, self.headings[i]))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % drift_force[i, j, k])
                writer.write("\n")

        return

    def _write_added_mass(self, writer, body):

        i_body = body.i_body

        added_mass = body.Added_mass[:, i_body*6:(i_body+1)*6, :]
        added_mass = np.moveaxis(added_mass, 0, -1)
        added_mass = np.flip(added_mass, 1)

        for i in range(6):
            writer.write("[ADDED_MASS_LINE_%1i]\n" % (i+1))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % added_mass[k, j, i])
                writer.write("\n")

        return

    def _write_damping(self, writer, body):

        i_body = body.i_body

        damping = body.Damping[:, i_body*6:(i_body+1)*6, :]
        damping = np.moveaxis(damping, 0, -1)
        damping = np.flip(damping, 1)

        for i in range(6):
            writer.write("[DAMPING_TERM_%1i]\n" % (i+1))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % damping[k, j, i])
                writer.write("\n")

        return

    def _write_added_mass_damping_body_coupling(self, writer, body_1, body_2):

        i_body_1 = body_1.i_body
        i_body_2 = body_2.i_body

        # Influence of body_2 -> body_1

        added_mass = body_1.Added_mass[:, i_body_2*6:(i_body_2+1)*6, :]
        added_mass = np.moveaxis(added_mass, 0, -1)
        added_mass = np.flip(added_mass, 1)

        damping = body_1.Damping[:, i_body_2*6:(i_body_2+1)*6, :]
        damping = np.moveaxis(damping, 0, -1)
        damping = np.flip(damping, 1)

        writer.write("[HYDRODYNAMIC_COUPLING_BETWEEN_STRUCTURES] %s\n" % body_2.name)
        writer.write("[ON] %s\n" % body_1.name)

        writer.write("[Coupled_Added_mass_Radiation_Damping]\n")

        for i in range(6):
            writer.write("[ADDED_MASS_LINE_%1i]\n" % (i+1))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % added_mass[k, j, i])
                writer.write("\n")

        for i in range(6):
            writer.write("[DAMPING_TERM_%1i]\n" % (i+1))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % damping[k, j, i])
                writer.write("\n")

        # Influence of body_1 -> body_2

        added_mass = body_2.Added_mass[:, i_body_1*6:(i_body_1+1)*6, :]
        added_mass = np.moveaxis(added_mass, 0, -1)
        added_mass = np.flip(added_mass, 1)

        damping = body_2.Damping[:, i_body_1*6:(i_body_1+1)*6, :]
        damping = np.moveaxis(damping, 0, -1)
        damping = np.flip(damping, 1)

        writer.write("[HYDRODYNAMIC_COUPLING_BETWEEN_STRUCTURES] %s " % body_1.name)
        writer.write("[ON] %s\n" % body_2.name)

        writer.write("[Coupled_Added_mass_Radiation_Damping]\n")

        for i in range(6):
            writer.write("[ADDED_MASS_LINE_%1i]\n" % (i+1))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % added_mass[k, j, i])
                writer.write("\n")

        for i in range(6):
            writer.write("[DAMPING_TERM_%1i]\n" % (i+1))
            for j in range(self._pyHDB.nb_wave_freq):
                writer.write("%8.3f  " % self.periods[j])
                for k in range(6):
                    writer.write("%.7E  " % damping[k, j, i])
                writer.write("\n")

        return

    def get_excitation_force(self, body):

        excitation_force = body.Froude_Krylov
        excitation_force += body.Diffraction
        excitation_force = np.transpose(excitation_force, (2, 1, 0))
        excitation_force = np.flip(excitation_force, 1)

        return excitation_force

    def get_rao(self, body):

        rao = body.RAO
        rao = np.transpose(rao, (2, 1, 0))
        rao = np.flip(rao, 1)

        return rao

    def _set_periods(self):

        self.periods = 2. * pi / self._pyHDB.omega
        self.periods = self.periods[np.argsort(self.periods)]

        self.nb_interp_periods = self.periods.size
        self.inter_periods = self.periods

        return

    def _set_headings(self):

        self.headings = self._pyHDB.wave_dir * 180. / pi

        return




















