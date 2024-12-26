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
import csv
import numpy as np
import xlsxwriter


str2idof = {
    "surge": 0,
    "sway" : 1,
    "heave": 2,
    "roll" : 3,
    "pitch": 4,
    "yaw": 5
}


class CSV_writer():

    """
    Class for writing ouput data in a CSV file.
    """

    def __init__(self, pyHDB, folder_path):

        # pyHDB.
        self._pyHDB = pyHDB

        # Creation of the output folder for storing the output files.
        self.output_folder = os.path.join(folder_path, "CSV")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def write_drift(self):
        """This function writes the mean wave drift load coefficients in the csv file."""

        Extension = ".csv"
        filename = os.path.join(self.output_folder, "HDB" + Extension)

        with open(filename, 'w') as csvfile:

            nw = self._pyHDB.nb_wave_freq
            nbeta = self._pyHDB.nb_wave_dir

            writer = csv.writer(csvfile, delimiter = '\t')
            Row_1 = ["Frequency (rad/s)", "Direction (deg)"]
            writer.writerow(Row_1)
            Row_2 = [None]
            for ibeta in range(0, nbeta):
                Row_2 += [str(np.degrees(self._pyHDB.wave_dir[ibeta]))]
            writer.writerow(Row_2)

class Excel_writer():

    """
    Class for writing ouput data in a Excel file.
    """

    def __init__(self, pyHDB, folder_path):

        # pyHDB.
        self._pyHDB = pyHDB

        # Creation of the output folder for storing the output files.
        self.output_folder = os.path.join(folder_path, "XLSX")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        return

    def write_sheet_drift(self, workbook, body, dof):
        """This function writes the mean wave drift load coefficient for a degree of freedom."""

        idof = str2idof[dof]

        nw = self._pyHDB.nb_wave_freq
        nbeta = int(self._pyHDB.nb_wave_dir)

        # Create a new worksheet.
        worksheet = workbook.add_worksheet(body.name.strip() + "_" + dof)

        # Cell formats.
        cell_format_frequency = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': 1})
        cell_format_direction = workbook.add_format({'align': 'left', 'valign': 'vcenter', 'bold': 1})
        cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        # Frequencies - Caption.
        worksheet.set_column('A:A', 20) # Widen the first column.
        worksheet.merge_range("A1:A2", "Frequency (rad/s)", cell_format_frequency) # Merge the two first lines of the first column.

        # Frequencies - Data.
        for iw in range(0, nw):
            worksheet.write(iw + 2, 0, self._pyHDB.wave_freq[iw], cell_format)

        # Wave directions - Caption.
        worksheet.merge_range(0, 1, 0, nbeta, "Direction (deg)", cell_format_direction) # Merge the first line for the wave direction caption.
        for ibeta in range(0, nbeta):
            worksheet.write(1, ibeta + 1, np.degrees(self._pyHDB.wave_dir[ibeta]), cell_format) # deg.

        # Wave directions - Data.
        for ibeta in range(0, nbeta):
            for iw in range(0, nw):
                worksheet.write(iw + 2, ibeta + 1, body.Wave_drift_force[idof, iw, ibeta])

        return

    def write_drift(self):
        """This function writes the mean wave drift load coefficients in the Excel file."""

        Extension = ".xlsx"
        filename = os.path.join(self.output_folder, "Drift_coefficients" + Extension)
        workbook = xlsxwriter.Workbook(filename)

        for body in self._pyHDB.bodies:
            # Surge.
            self.write_sheet_drift(workbook, body, "surge")
            # Sway.
            self.write_sheet_drift(workbook, body, "sway")
            # Yaw.
            self.write_sheet_drift(workbook, body, "yaw")

        # Closing.
        workbook.close()

        return