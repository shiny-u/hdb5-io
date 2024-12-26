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

"""Module to plot the hydrodynamic database."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

import meshmagick.MMviewer
from hdb5_io.HDB5tool.WaveSpectrum import JonswapWaveSpectrum


Dof_notation = [r'x', r'y', r'z', r'\phi', r'\theta', r'\psi']
Dof_name = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def plot_loads(data, w, DiffOrFKOrExc, ibody, iforce, beta, x_derivative, show=True, save=False, filename="Loads.png"):
    """Plots the diffraction or Froude-Krylov or excitation response function of a given modes set.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: diffraction or Froude-Krylov loads.
    w : Array of floats.
        Wave frequencies.
    DiffOrFKOrExc : int.
        0 for diffraction loads, 1 for Froude-Krylov loads, 2 for excitation loads.
    ibody : int.
        The index of the body.
    iforce : int.
        The index of the body's force mode.
    beta : float.
        Wave direction in radians.
    """

    # Labels and title.
    xlabel = r'$\omega$' + ' $(rad/s)$'
    if (DiffOrFKOrExc == 0):  # Diffraction loads.

        # Amplitude.
        if (iforce <= 2):
            if(x_derivative):
                ylabel1 = r'$|\partial F_{Diff}^{%s}/\partial x(\omega, \beta)|$' % Dof_notation[iforce]
            else:
                ylabel1 = r'$|F_{Diff}^{%s}(\omega, \beta)|$' % Dof_notation[iforce]
        else:
            if (x_derivative):
                ylabel1 = r'$|\partial M_{Diff}^{%s}/\partial x(\omega, \beta)|$' % Dof_notation[iforce]
            else:
                ylabel1 = r'$|M_{Diff}^{%s}(\omega, \beta)|$' % Dof_notation[iforce]

        # Phase.
        if (iforce <= 2):
            if (x_derivative):
                ylabel2 = r'$Arg\left[\partial F_{Diff}^{%s}/\partial x(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
            else:
                ylabel2 = r'$Arg\left[F_{Diff}^{%s}(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
        else:
            if (x_derivative):
                ylabel2 = r'$Arg\left[\partial F_{Diff}^{%s}/\partial x(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
            else:
                ylabel2 = r'$Arg\left[F_{Diff}^{%s}(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]

        # Title.
        if (x_derivative):
            title = r'x-derivative of the diffraction loads in %s of body %u for a wave of direction %.1f deg' % \
                    (Dof_name[iforce], ibody + 1, np.degrees(beta))
        else:
            title = r'Diffraction loads in %s of body %u for a wave of direction %.1f deg' % \
                    (Dof_name[iforce], ibody + 1, np.degrees(beta))
    elif (DiffOrFKOrExc == 1):  # Froude-Krylov loads.

        # Amplitude.
        if (iforce <= 2):
            if (x_derivative):
                ylabel1 = r'$|\partial F_{FK}^{%s}/\partial x(\omega, \beta)|$' % Dof_notation[iforce]
            else:
                ylabel1 = r'$|F_{FK}^{%s}(\omega, \beta)|$' % Dof_notation[iforce]
        else:
            if (x_derivative):
                ylabel1 = r'$|\partial M_{FK}^{%s}/\partial x(\omega, \beta)|$' % Dof_notation[iforce]
            else:
                ylabel1 = r'$|M_{FK}^{%s}(\omega, \beta)|$' % Dof_notation[iforce]

        # Phase.
        if (iforce <= 2):
            if (x_derivative):
                ylabel2 = r'$Arg\left[\partial F_{FK}^{%s}/\partial x(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
            else:
                ylabel2 = r'$Arg\left[F_{FK}^{%s}(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
        else:
            if (x_derivative):
                ylabel2 = r'$Arg\left[\partial F_{FK}^{%s}/\partial x(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
            else:
                ylabel2 = r'$Arg\left[F_{FK}^{%s}(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]

        # Title.
        if (x_derivative):
            title = r'x-derivative of the Froude-Krylov loads in %s of body %u for a wave of direction %.1f deg' % \
                    (Dof_name[iforce], ibody + 1, np.degrees(beta))
        else:
            title = r'Froude-Krylov loads in %s of body %u for a wave of direction %.1f deg' % \
                    (Dof_name[iforce], ibody + 1, np.degrees(beta))

    elif (DiffOrFKOrExc == 2):  # Excitation loads.

        # Amplitude.
        if (iforce <= 2):
            if (x_derivative):
                ylabel1 = r'$|\partial F_{Exc}^{%s}/\partial x(\omega, \beta)|$' % Dof_notation[iforce]
            else:
                ylabel1 = r'$|F_{Exc}^{%s}(\omega, \beta)|$' % Dof_notation[iforce]
        else:
            if (x_derivative):
                ylabel1 = r'$|\partial M_{Exc}^{%s}/\partial x(\omega, \beta)|$' % Dof_notation[iforce]
            else:
                ylabel1 = r'$|M_{Exc}^{%s}(\omega, \beta)|$' % Dof_notation[iforce]

        # Phase.
        if (iforce <= 2):
            if (x_derivative):
                ylabel2 = r'$Arg\left[\partial F_{Exc}^{%s}/\partial x(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
            else:
                ylabel2 = r'$Arg\left[F_{Exc}^{%s}(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
        else:
            if (x_derivative):
                ylabel2 = r'$Arg\left[\partial F_{Exc}^{%s}/\partial x(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]
            else:
                ylabel2 = r'$Arg\left[F_{Exc}^{%s}(\omega,\beta)\right] (deg)$' % Dof_notation[iforce]

        # Title.
        if (x_derivative):
            title = r'x-derivative of the excitation loads in %s of body %u for a wave of direction %.1f deg' % \
                    (Dof_name[iforce], ibody + 1, np.degrees(beta))
        else:
            title = r'Excitation loads in %s of body %u for a wave of direction %.1f deg' % \
                    (Dof_name[iforce], ibody + 1, np.degrees(beta))

    # Units.
    if (iforce <= 2):
        if (x_derivative):
            ylabel1 += r' $(N/m^2)$'
        else:
            ylabel1 += r' $(N/m)$' # Because it is divided by the wave amplitude.
    else:
        if (x_derivative):
            ylabel1 += r' $(N.m/m^2)$'
        else:
            ylabel1 += r' $(N.m/m)$' # Because it is divided by the wave amplitude.

    # Plots.
    if (save == False):
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(w, np.absolute(data), linestyle="-", linewidth=2)
    plt.ylabel(ylabel1, fontsize=18)
    if (save == False):
        plt.title(title, fontsize=20)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(data, deg=True), linestyle="-", linewidth=2)
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_AB(data, w, ibody_force, iforce, ibody_motion, idof, x_derivative, show=True, save=False, filename="AB.png"):
    """Plots the radiation coefficients of a given modes set.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: added mass and damping coefficients.
    w : Array of floats.
        Wave frequencies.
    ibody_force : int
        Index of the body where the radiation force is applied.
    iforce : int
        Index of the local body's force mode.
    ibody_motion : int
        Index of the body having a motion.
    idof : int
        Index of the local body's radiation mode (motion).
    """

    # Label.
    xlabel = r'$\omega$' + ' $(rad/s)$'
    if(x_derivative):
        ylabel1 = r'$\partial A_{%s}/\partial x(\omega)$' % (
                Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
        ylabel2 = r'$\partial B_{%s}/\partial x(\omega)$' % (
                Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
    else:
        ylabel1 = r'$A_{%s}(\omega)$' % (
                Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
        ylabel2 = r'$B_{%s}(\omega)$' % (
                Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))

    if (iforce <= 2):
        force_str = 'force'
        if (idof <= 2):  # Translation.
            if (x_derivative):
                ylabel1 += r' $(kg/m)$'
                ylabel2 += r' $(kg/(m\,s)$'
            else:
                ylabel1 += r' $(kg)$'
                ylabel2 += r' $(kg/s)$'
            motion_str = 'translation'
        else:  # Rotation.
            if (x_derivative):
                ylabel1 += r' $(kg)$'
                ylabel2 += r' $(kg/s)$'
            else:
                ylabel1 += r' $(kg\,m)$'
                ylabel2 += r' $(kg\,m/s)$'
            motion_str = 'rotation'
    else:
        force_str = 'moment'
        if (idof <= 2):  # Translation.
            if (x_derivative):
                ylabel1 += r' $(kg)$'
                ylabel2 += r' $(kg/s)$'
            else:
                ylabel1 += r' $(kg\,m)$'
                ylabel2 += r' $(kg\,m/s)$'
            motion_str = 'translation'
        else:  # Rotation.
            if (x_derivative):
                ylabel1 += r' $(kg\,m)$'
                ylabel2 += r' $(kg\,m/s)$'
            else:
                ylabel1 += r' $(kg\,m^2)$'
                ylabel2 += r' $(kg\,m^2/s)$'
            motion_str = 'rotation'

    if (x_derivative):
        title = r"x-derivative of the radiation coefficients giving the %s in %s of body %u " \
                r"for a %s in %s of body %u" \
                % (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1)
    else:
        title = r"Radiation coefficients giving the %s in %s of body %u " \
                r"for a %s in %s of body %u" \
                % (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1)

    plt.close()
    if (save == False):
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))

    # Added mass.
    plt.subplot(2, 1, 1)
    plt.plot(w, data[:len(w), 0], linestyle="-", linewidth=2)
    plt.plot(w[-1], data[-1, 0], marker="+", color="red", markersize=10)
    plt.ylabel(ylabel1, fontsize=18)
    if (save == False):
        plt.title(title, fontsize=20)
    plt.grid()

    # Damping.
    plt.subplot(2, 1, 2)
    plt.plot(w, data[0:len(w), 1], linestyle="-", linewidth=2)
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    # Show and save.
    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_AB_array(data, w, ibody_force, ibody_motion, pyHDB, XDerivatives):
    """Plots ALL the radiation coefficients of a body.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: Combinaison of both added mass and damping coefficients.
    w : Array of floats.
        Wave frequencies.
    ibody_force : int
        Index of the body where the radiation force is applied.
    ibody_motion : int
        Index of the body having a motion.
    """

    # Title.
    title = r"Combinaison of radiation coefficients of body %u generated by a motion of body %u" % (
    ibody_force + 1, ibody_motion + 1) \
            + "\n " + \
            r"$H_{%s}(j\omega) = |B_{%s}(\omega) + j\omega[A_{%s}(\omega) - A_{%s}^{\infty}]|$" \
            % (str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1),
               str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1))
    if(XDerivatives):
        title = r"Combinaison of radiation coefficients of body %u generated by a motion of body %u" % (
            ibody_force + 1, ibody_motion + 1) \
                + "\n " + \
                r"$\partial H_{%s}/\partial x(j\omega) = |\partial B_{%s}/\partial x(\omega) + j\omega[\partial A_{%s}/\partial x(\omega) - \partial A_{%s}^{\infty}/\partial x]|$" \
                % (str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1),
                   str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1))


    # Definition of the figure.
    plt.close()
    fig, axes = plt.subplots(6, 6, figsize=(16, 8.5))

    # Plot.
    for iforce in range(0, 6):
        for idof in range(0, 6):
            labelA = r'$H_{%s} (\omega)$' % (
                        Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            if (XDerivatives):
                labelA = r'$\partial H_{%s}/\partial x(\omega)$' % (
                        Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            axes[iforce, idof].plot(w, data[iforce, idof, :], linestyle="-", linewidth=2)
            axes[iforce, idof].grid()
            axes[iforce, idof].set_title(labelA)

            # Automatic pre-filling of the radiation mask (can be change by clicking on the plots).
            if (pyHDB.bodies[ibody_force].Radiation_mask[iforce, 6 * ibody_motion + idof] == True
                    and (pyHDB.bodies[ibody_force].Force_mask[iforce] == 0 or pyHDB.bodies[ibody_force].Motion_mask[
                        idof] == 0)):
                pyHDB.bodies[ibody_force].Radiation_mask[iforce, 6 * ibody_motion + idof] = False
                axes[iforce, idof].set_facecolor("grey")

            # If a radiation mask has already been defined.
            if (pyHDB.bodies[ibody_force].Radiation_mask[iforce, 6 * ibody_motion + idof] == False):
                axes[iforce, idof].set_facecolor("grey")

    # What to do if a mouse click is performed.
    def onclick(event):

        # Change the boolean of Radiation_mask.
        iplot = 0
        for iforce in range(0, 6):
            for idof in range(0, 6):
                if event.inaxes == axes[iforce, idof]:

                    # Inverse the radiation mask and update the background color of the plot accordingly.
                    if (pyHDB.bodies[ibody_force].Radiation_mask[iforce, 6 * ibody_motion + idof] == True):
                        pyHDB.bodies[ibody_force].Radiation_mask[iforce, 6 * ibody_motion + idof] = False
                        event.canvas.figure.get_axes()[iplot].set_facecolor('grey')
                    else:
                        pyHDB.bodies[ibody_force].Radiation_mask[iforce, 6 * ibody_motion + idof] = True
                        event.canvas.figure.get_axes()[iplot].set_facecolor('white')

                    # Application of the modification.
                    event.canvas.draw()

                iplot = iplot + 1

    # Even of a mouse click.
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Title.
    plt.suptitle(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(top=0.88)

    # Show the plot.
    plt.show()


def plot_irf(data, time, IRFtype, ibody_force, iforce, ibody_motion, idof, show=True, save=False,
             filename="IRF.png"):
    """Plots the impulse response function of a given modes set.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: impulse response functions.
    time : Array of floats.
        Time.
    IRFtype : int
        Type of IRF: K0 (0), Kua (1), Kub (2), Ku2 (3) (cf doc Helios).
    ibody_force : int
        Index of the body where the radiation force is applied.
    iforce : int
        Index of the local body's force mode.
    ibody_motion : int
        Index of the body having a motion.
    idof : int
        Index of the local body's raditation mode (motion).
    """

    # Labels.
    if (IRFtype == 0):  # Without forward speed.
        ylabel = r'$K_{%s}$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(
            ibody_motion + 1))
    elif(IRFtype == 2):  # Proportional to the forward speed velocity, no x-derivatives.
        ylabel = r'$Ku_{%s}$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(
            ibody_motion + 1))
    elif (IRFtype == 1):  # Proportional to the forward speed velocity, with x-derivatives.
        ylabel = r'$Ku_{%s}^{x-diff}$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(
            ibody_motion + 1))
    elif (IRFtype == 3):  # Proportional to the square of the forward speed velocity.
        ylabel = r'$Ku^2_{%s}$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(
            ibody_motion + 1))

    if (iforce <= 2):
        force_str = 'force'
        if (idof <= 2):  # Translation.
            ylabel += r' $(kg/s^2)$'
            motion_str = 'translation'
        else:  # Rotation.
            ylabel += r' $(kg\,m/s^2)$'
            motion_str = 'rotation'
    else:
        force_str = 'moment'
        if (idof <= 2):  # Translation.
            ylabel += r' $(kg\,m/s^2)$'
            motion_str = 'translation'
        else:  # Rotation.
            ylabel += r' $(kg\,m^2/s^2)$'
            motion_str = 'rotation'

    # Plots.
    if (save == False):
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))
    plt.plot(time, data)
    plt.xlabel(r'$t$' + ' $(s)$', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)  # TODO: mettre une unite
    if (save == False):
        if (IRFtype == 0):  # Without forward speed.
            plt.title('Impulse response function of the radiation %s in %s of body %u for a %s in %s of body %u' %
                      (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1),
                      fontsize=20)
        elif (IRFtype == 2): # Proportional to the forward speed velocity, no x-derivatives.
            plt.title('Impulse response function proportional to the forward speed and without x-derivatives\n of the radiation %s in %s of body %u for a %s in %s of body %u' %
                (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1),
                fontsize=20)
        elif (IRFtype == 1): # Proportional to the forward speed velocity, with x-derivatives.
            plt.title('Impulse response function proportional to the forward speed and with x-derivatives\n of the radiation %s in %s of body %u for a %s in %s of body %u' %
                (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1),
                fontsize=20)
        elif (IRFtype == 3): # Proportional to the square of the forward speed velocity.
            plt.title('Impulse response function proportional to the square of the forward speed\n of the radiation %s in %s of body %u for a %s in %s of body %u' %
                (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1),
                fontsize=20)
    plt.grid()

    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_irf_array(data, time, ibody_force, ibody_motion, IRFtype):
    """Plots ALL the impulse response functions of a body.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: Combinaison impulse response functions.
    time : Array of floats.
        Time.
    ibody_force : int
        Index of the body where the radiation force is applied.
    ibody_motion : int
        Index of the body having a motion.
    """

    # Title.
    title = r"Combinaison of impulse response functions of body %u generated by a motion of body %u" % (ibody_force + 1, ibody_motion + 1)

    # Definition of the figure.
    plt.close()
    fig, axes = plt.subplots(6, 6, figsize=(16, 8.5))

    # Plot.
    for iforce in range(0, 6):
        for idof in range(0, 6):
            if(IRFtype == 0):
                labelA = r'$K_{%s} (t)$' % ( Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            elif(IRFtype == 1):
                labelA = r'$Ku_{%s} (t)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            elif (IRFtype == 2):
                labelA = r'$Ku_{%s}^{x-diff} (t)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            else:
                labelA = r'$Ku2_{%s} (t)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            axes[iforce, idof].plot(time, data[iforce, idof, :], linestyle="-", linewidth=2)
            axes[iforce, idof].grid()
            axes[iforce, idof].set_title(labelA)

    # Title.
    plt.suptitle(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(top=0.88)

    # Show the plot.
    plt.show()


def plot_filering(data, time, SpeedOrNot, coeff, ibody_force, iforce, ibody_motion, idof, **kwargs):
    """This function plots the filtered impulse response functions.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: impulse response functions with and without filetering.
    time : Array of floats.
        Time.
    SpeedOrNot : int
        IRF with forward speed (1) or not (0).
    coeff : Array of floats.
        Filerting.
    SpeedOrNot : int
        IRF with forward speed (1) or not (0).
    ibody_force : int
        Index of the body where the radiation force is applied.
    iforce : int
        Index of the local body's force mode.
    ibody_motion : int
        Index of the body having a motion.
    idof : int
        Index of the local body's raditation mode (motion).
    kwargs: optional
        Arguments that are to be used by pyplot.
    """

    # Labels.
    if (iforce <= 2):
        force_str = 'force'
        if (idof <= 2):  # Translation.
            motion_str = 'translation'
        else:  # Rotation.
            motion_str = 'rotation'
    else:
        force_str = 'moment'
        if (idof <= 2):  # Translation.
            motion_str = 'translation'
        else:  # Rotation.
            motion_str = 'rotation'

    if (SpeedOrNot == 0):  # Without forward speed.
        ylabel = r'$K_{%s}(t)$' % (str(iforce + 1) + str(idof + 1))
    else:  # With forward speed.
        ylabel = r'$Ku_{%s}(t)$' % (str(iforce + 1) + str(idof + 1))

    plt.figure()
    plt.plot(time, data, label="Without filetering")
    plt.plot(time, data * coeff, label="With filering")
    plt.xlabel(r'$t$' + ' $(s)$', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)  # TODO: mettre une unite
    if (SpeedOrNot == 0):  # Without forward speed.
        plt.title('Impulse response function of the %s in %s of body %u for a %s in %s of body %u' %
                  (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1),
                  fontsize=20)
    else:  # With forward speed.
        plt.title('Impulse response function with forward speed of the %s in %s of body %u for a %s in %s of body %u' %
                  (force_str, Dof_name[iforce], ibody_force + 1, motion_str, Dof_name[idof], ibody_motion + 1),
                  fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()


def plot_loads_all_wave_dir(data, w, DiffOrFKOrExc, ibody, iforce, beta,
                            show=True, save=False, is_period=False, filename="Loads.png"):

    """Plots the diffraction or Froude-Krylov or excitation response functions.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: diffraction or Froude-Krylov loads.
    w : Array of floats.
        Wave frequencies.
    DiffOrFKOrExc : int.
        0 for diffraction loads, 1 for Froude-Krylov loads, 2 for excitation loads.
    ibody : int.
        The index of the body.
    iforce : int.
        The index of the body's force mode.
    beta : float.
        Wave directions in radians.
    """

    # Labels and title.
    if is_period:
        xlabel = r'$T$'+'$ (s)$'
        xvar = 'T'
    else:
        xlabel = r'$\omega$'+' $(rad/s)$'
        xvar = '\omega'

    if(DiffOrFKOrExc == 0): # Diffraction loads.

        # Amplitude.
        if(iforce <= 2):
            ylabel1 = r'$|F_{Diff}^{%s}({%s}, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)
        else:
            ylabel1 = r'$|M_{Diff}^{%s}({%s}, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)

        # Phase.:
        if(iforce <= 2):
            ylabel2 = r'$Arg\left[F_{Diff}^{%s}({%s},\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)
        else:
            ylabel2 = r'$Arg\left[F_{Diff}^{%s}({%s},\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)

        # Title.
        title = r'Diffraction loads in %s of body %u' % \
                (Dof_name[iforce], ibody + 1)

    elif(DiffOrFKOrExc == 1): # Froude-Krylov loads.

        # Amplitude.
        if (iforce <= 2):
            ylabel1 = r'$|F_{FK}^{%s}({%s}, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)
        else:
            ylabel1 = r'$|M_{FK}^{%s}({%s}, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)

        # Phase.
        if (iforce <= 2):
            ylabel2 = r'$Arg\left[F_{FK}^{%s}({%s},\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)
        else:
            ylabel2 = r'$Arg\left[F_{FK}^{%s}({%s},\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)

        # Title.
        title = r'Froude-Krylov loads in %s of body %u' % \
                (Dof_name[iforce], ibody + 1)

    elif(DiffOrFKOrExc == 2): # Excitation loads.

        # Amplitude.
        if (iforce <= 2):
            ylabel1 = r'$|F_{Exc}^{%s}({%s}, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)
        else:
            ylabel1 = r'$|M_{Exc}^{%s}({%s}, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)

        # Phase.
        if (iforce <= 2):
            ylabel2 = r'$Arg\left[F_{Exc}^{%s}({%s},\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)
        else:
            ylabel2 = r'$Arg\left[F_{Exc}^{%s}({%s},\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1), xvar)

        # Title.
        title = r'Excitation loads in %s of body %u' % \
                (Dof_name[iforce], ibody + 1)

    # Units.
    if (iforce <= 2):
        ylabel1 += r' $(N/m)$'
    else:
        ylabel1 += r' $(N)$'

    # Colors.
    colors = cm.jet(np.linspace(1, 0, beta.shape[0]))

    # Plots.
    if (save == False):
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))

    # Amplitude.
    plt.subplot(2, 1, 1)
    for ibeta in range(0, beta.shape[0]):
        if is_period:
            plt.plot(2.*np.pi/w, np.absolute(data[:, ibeta]), linestyle="-", linewidth = 2, label = str(beta[ibeta])+" deg", color = colors[ibeta])
        else:
            plt.plot(w, np.absolute(data[:, ibeta]), linestyle="-", linewidth = 2, label = str(beta[ibeta])+" deg", color = colors[ibeta])

    plt.ylabel(ylabel1, fontsize=18)
    if (save == False):
        plt.title(title, fontsize=20)
    plt.grid()
    plt.legend()

    # Phase.
    plt.subplot(2, 1, 2)
    for ibeta in range(0, beta.shape[0]):
        if is_period:
            plt.plot(2.*np.pi/w, np.angle(data[:, ibeta], deg=True),linestyle="-", linewidth = 2, color = colors[ibeta])
        else:
            plt.plot(w, np.angle(data[:, ibeta], deg=True),linestyle="-", linewidth = 2, color = colors[ibeta])
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    # Show and save.
    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_AB_multiple_coef(data, w, ibody_force, iforce, ibody_motion, show = True, save = False, filename = "AB.png"):
    """Plots the radiation coefficients of a given modes set.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: added mass and damping coefficients.
    w : Array of floats.
        Wave frequencies.
    ibody_force : int
        Index of the body where the radiation force is applied.
    iforce : int
        Index of the local body's force mode.
    ibody_motion : int
        Index of the body having a motion.
    """

    # Dimension for taking into account or not the VF approximation.
    if(data.shape[2] == 2):
        VF = True
    else:
        VF = False

    # Labels.
    xlabel = r'$\omega$' + ' $(rad/s)$'
    ylabel1 = r'$A(\omega)$'
    ylabel2 = r'$B(\omega)$'

    # Colors.
    colors = cm.jet(np.linspace(0.9, 0, 6))

    # Definition of the figure.
    plt.close()
    if (save == False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8.5))
    else:
        if(VF):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6)) # Smaller figures because the legend is longer.
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))

    # Added mass coefficients.
    legend_plot = [[] for i in range(6)]
    ax1bis = ax1.twinx()
    for idof in range(0, 6):
        labelA = r'$A_{%s}$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
        if (iforce <= 2):
            if(idof <= 2): # Translation
                labelA = labelA + r' $(kg)$'
            else: # Rotation.
                labelA = labelA + r' $(kg\,m)$'
        else:
            if(idof <= 2): # Translation.
                labelA = labelA + r' $(kg\,m)$'
            else: # Rotation.
                labelA = labelA + r' $(kg\,m^2)$'
        if(idof <= 2): # Translation.
            legend_plot[idof] = ax1.plot(w, data[:len(w), idof, 0], linestyle="-", linewidth=2, label=labelA, color=colors[idof])
            ax1.plot(w[-1], data[-1, idof, 0], marker="+", markersize=10, mew=3, color=colors[idof])
            if(VF):
                ax1.plot(w, data[:len(w), idof, 1], linestyle="--", linewidth=1, label=labelA, color=colors[idof])
        else: # Rotation.
            legend_plot[idof] = ax1bis.plot(w, data[:len(w), idof, 0], linestyle="-", linewidth=2, label=labelA, color=colors[idof])
            ax1bis.plot(w[-1], data[-1, idof, 0], marker="+", markersize=10, mew=3, color=colors[idof])
            if (VF):
                ax1bis.plot(w, data[:len(w), idof, 1], linestyle="--", linewidth=1, label=labelA, color=colors[idof])
    # Units.
    if (iforce <= 2):
        ylabel1_ax1 = ylabel1 + r' $(kg)$' # Translation.
        ylabel1_ax1bis = ylabel1 + r' $(kg\,m)$' # Rotation.
    else:
        ylabel1_ax1 = ylabel1 + r' $(kg\,m)$' # Translation.
        ylabel1_ax1bis = ylabel1 + r' $(kg\,m^2)$' # Rotation.
    if (VF):
        ax1.set_ylabel(ylabel1_ax1, fontsize=15)
        ax1bis.set_ylabel(ylabel1_ax1bis, fontsize=15)
    else:
        ax1.set_ylabel(ylabel1_ax1, fontsize=18)
        ax1bis.set_ylabel(ylabel1_ax1bis, fontsize=18)

    # Legend.
    legend = []
    for idof in range(0, 6):
        legend += legend_plot[idof]
    labs = [l.get_label() for l in legend]
    if (VF):
        ax1.legend(legend, labs, fontsize=10, ncol=2)
    else:
        ax1.legend(legend, labs, fontsize=12, ncol=2)
    ax1.grid()

    # Damping coefficients.
    legend_plot = [[] for i in range(6)]
    ax2bis = ax2.twinx()
    for idof in range(0, 6):
        labelB = r'$B_{%s}$' % (Dof_notation[iforce]+"_"+str(ibody_force+1) + Dof_notation[idof]+"_"+str(ibody_motion+1))
        if (iforce <= 2):
            if(idof <= 2): # Translation.
                labelB = labelB + r' $(kg/s)$'
            else: # Rotation.
                labelB = labelB + r' $(kg\,m/s)$'
        else:
            if(idof <= 2): # Translation.
                labelB = labelB + r' $(kg\,m/s)$'
            else: # Rotation.
                labelB = labelB + r' $(kg\,m^2/s)$'
        if (idof <= 2): # Translation.
            legend_plot[idof] = ax2.plot(w, data[0:len(w),6 + idof, 0], linestyle="-", linewidth = 2, label = labelB, color = colors[idof])
            if (VF):
                ax2.plot(w, data[0:len(w), 6 + idof, 1], linestyle="--", linewidth=1, label=labelB, color=colors[idof])
        else: # Rotation.
            legend_plot[idof] = ax2bis.plot(w, data[0:len(w), 6 + idof, 0], linestyle="-", linewidth=2, label = labelB, color=colors[idof])
            if (VF):
                ax2bis.plot(w, data[0:len(w), 6 + idof, 1], linestyle="--", linewidth=1, label=labelB, color=colors[idof])
    # Units.
    if (iforce <= 2):
        ylabel2_ax2 = ylabel2 + r' $(kg/s)$' # Translation.
        ylabel2_ax2bis = ylabel2 + r' $(kg\,m/s)$' # Rotation.
    else:
        ylabel2_ax2 = ylabel2 + r' $(kg\,m/s)$' # Translation.
        ylabel2_ax2bis = ylabel2 + r' $(kg\,m^2/s)$' # Rotation.
    if (VF):
        ax2.set_ylabel(ylabel2_ax2, fontsize=15)
        ax2bis.set_ylabel(ylabel2_ax2bis, fontsize=15)
        ax2.set_xlabel(xlabel, fontsize=15)
    else:
        ax2.set_ylabel(ylabel2_ax2, fontsize=18)
        ax2bis.set_ylabel(ylabel2_ax2bis, fontsize=18)
        ax2.set_xlabel(xlabel, fontsize=18)
    ax2.grid()

    # Legend.
    legend = []
    for idof in range(0, 6):
        legend += legend_plot[idof]
    labs = [l.get_label() for l in legend]
    if (VF):
        ax2.legend(legend, labs, fontsize=10, ncol=2)
    else:
        ax2.legend(legend, labs, fontsize=12, ncol=2)

    if (show == True):
        plt.show()
    if(save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_irf_multiple_coef(data, time, SpeedOrNot, ibody_force, iforce, ibody_motion, show = True, save = False, filename = "IRF.png"):
    """Plots the impulse response function of a given modes set.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: impulse response functions.
    time : Array of floats.
        Time.
    SpeedOrNot : int
        IRF with forward speed (1) or not (0).
    ibody_force : int
        Index of the body where the radiation force is applied.
    iforce : int
        Index of the local body's force mode.
    ibody_motion : int
        Index of the body having a motion.
    """

    # Labels.
    if (iforce <= 2):
        force_str = 'force'
    else:
        force_str = 'moment'

    if (SpeedOrNot == 0): # Without forward speed.
        ylabel = r'$K(t)$'
    else: # With forward speed.
        ylabel = r'$Ku(t)$'

    # Colors.
    colors = cm.jet(np.linspace(0.9, 0, 6))

    # Definition of the figure.
    plt.close()
    if (save == False):
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8.5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6.5))

    # Plots.
    legend_plot = [[] for i in range(6)]
    ax1bis = ax1.twinx()
    for idof in range(0, 6):
        if (SpeedOrNot == 0): # Without forward speed.
            unit = r'$K_{%s}$' % (Dof_notation[iforce]+"_"+str(ibody_force+1) + Dof_notation[idof]+"_"+str(ibody_motion+1))
        else: # With forward speed.
            unit = r'$Ku_{%s}$' % (Dof_notation[iforce]+"_"+str(ibody_force+1) + Dof_notation[idof]+"_"+str(ibody_motion+1))
        if (iforce <= 2):
            if (idof <= 2): # Translation.
                unit += r' $(kg/s^2)$'
            else: # Rotation.
                unit += r' $(kg\,m/s^2)$'
        else:
            if (idof <= 2): # Translation.
                unit += r' $(kg\,m/s^2)$'
            else: # Rotation.
                unit += r' $(kg\,m^2/s^2)$'
        if(idof <= 2): # Translation.
            legend_plot[idof] = ax1.plot(time, data[:, idof], linestyle="-", linewidth=2, label=unit, color=colors[idof])
        else: # Rotation.
            legend_plot[idof] = ax1bis.plot(time, data[:, idof], linestyle="-", linewidth=2, label=unit, color=colors[idof])

    # Units.
    if (iforce <= 2):
        ylabel_ax1 = ylabel + r' $(kg/s^2)$' # Translation.
        ylabel_ax1bis = ylabel + r' $(kg\,m/s^2)$' # Rotation.
    else:
        ylabel_ax1 = ylabel + r' $(kg\,m/s^2)$' # Translation.
        ylabel_ax1bis = ylabel + r' $(kg\,m^2/s^2)$' # Rotation.
    ax1.set_ylabel(ylabel_ax1, fontsize=18)
    ax1bis.set_ylabel(ylabel_ax1bis, fontsize=18)

    # Legend.
    legend = []
    for idof in range(0, 6):
        legend += legend_plot[idof]
    labs = [l.get_label() for l in legend]
    ax1.legend(legend, labs, fontsize=12, ncol=2)
    ax1.grid()

    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_RAO_fig(data, w, ibody, iforce, beta, eigenfreq, show = True, save = False, filename = "RAO.png"):
    """Plots the RAO of a given modes set

    Parameters
    ----------
    data : Array of floats.
        Data to plot: RAO.
    w : Array of floats.
        Wave frequencies.
    ibody : int.
        The index of the body.
    iforce : int.
        The index of the body's force mode.
    beta : float.
        Wave direction in degrees.
    kwargs: optional.
        Arguments that are to be used by pyplot
    """

    # Labels and title.
    xlabel = r'$\omega$' + ' $(rad/s)$'
    ylabel1 = r'$|RAO(\omega, \beta)|$'
    ylabel2 = r'$Arg\left[RAO(\omega,\beta)\right] (deg)$'
    title = r'RAO in %s of body %u for a wave of direction %.1f deg' % \
            (Dof_name[iforce], ibody + 1, np.degrees(beta))

    # Units.
    if iforce <= 2:
        ylabel1 += r' $(m/m)$'
    else:
        ylabel1 += r' $(deg/m)$'

    # Plots.
    if (save == False):  # The size is smaller for the generation of automatic report because the title is not including.
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))

    # Amplitude.
    plt.subplot(2, 1, 1)
    if iforce >= 3: # Convertion of the RAO in roll, pitch and yaw in degrees.
        plt.plot(w, np.degrees(np.absolute(data)), linestyle="-", linewidth=2)
    else:
        plt.plot(w, np.absolute(data), linestyle="-", linewidth=2)
    if(eigenfreq >= 0.): # Otherwise the natural frequency is not defined (negative or zero stiffness coefficient).
        plt.axvline(x = eigenfreq, linewidth = 2, linestyle = "--", color = 'r')
    plt.ylabel(ylabel1, fontsize=18)
    if (save == False):  # The title is not necessary for the generation of automatic report.
        plt.title(title, fontsize=20)
    plt.grid()

    # Phase.
    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(data, deg=True), linestyle="-", linewidth=2)
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    # Show and save.
    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def plot_RAO_all_wave_dir(data, w, beta, ibody, iforce, eigenfreq, rads_or_period, show = True, save = False, filename = "RAO.png"):
    """Plots the RAO for all wave directions for a body and a degree of freedom.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: RAO.
    w : Array of floats.
        Wave frequencies.
    beta : float.
        Wave directions in radians.
    ibody : int.
        The index of the body.
    iforce : int.
        The index of the body's force mode.
    kwargs: optional.
        Arguments that are to be used by pyplot
    """

    # Labels and title.
    if rads_or_period is True: # Wave frequency.
        xlabel = r'$\omega$ $(rad/s)$'
        ylabel1 = r'$|RAO_{%s}(\omega, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1))
        ylabel2 = r'$Arg\left[RAO_{%s}(\omega,\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1))
    else: # Wave period.
        xlabel = r'$T$ $(s)$'
        ylabel1 = r'$|RAO_{%s}(T, \beta)|$' % (Dof_notation[iforce] + "_" + str(ibody + 1))
        ylabel2 = r'$Arg\left[RAO_{%s}(T,\beta)\right] (deg)$' % (Dof_notation[iforce] + "_" + str(ibody + 1))

    title = r'RAO in %s of body %u' % (Dof_name[iforce], ibody + 1)

    # Units.
    if iforce <= 2:
        ylabel1 += r' $(m/m)$'
    else:
        ylabel1 += r' $(deg/m)$'

    # Plots.
    if (save == False):  # The size is smaller for the generation of automatic report because the title is not including.
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))

    # Colors.
    colors = cm.jet(np.linspace(1, 0, beta.shape[0]))

    # Abscissa.
    if rads_or_period is True: # Wave frequency.
        iw_min = 0
        Abscissa = w
        EigenAbscissa = eigenfreq
    else: # Wave period.
        # Not to have too high wave periods.
        iw_min = 0
        ##CC for i in range(0, w.shape[0]):
        ##CC     if 2 * np.pi / w[i] > 40:  # Wave period maximum.
        ##CC        iw_min = i + 1  # For having the wave period lower than 40 m.
        Abscissa = 2 * np.pi / w[iw_min:]
        if eigenfreq <= w[iw_min]:
            EigenAbscissa = -1
        else:
            EigenAbscissa = 2 * np.pi / eigenfreq

    # Amplitude.
    plt.subplot(2, 1, 1)
    for ibeta in range(0, beta.shape[0]):
        if iforce >= 3: # Convertion of the RAO in roll, pitch and yaw in degrees.
            plt.plot(Abscissa, np.degrees(np.absolute(data[iw_min:, ibeta])), linestyle="-", linewidth=2, label = str(beta[ibeta])+" deg", color = colors[ibeta])
        else:
            plt.plot(Abscissa, np.absolute(data[iw_min:, ibeta]), linestyle="-", linewidth=2, label=str(beta[ibeta]) + " deg", color=colors[ibeta])
    if EigenAbscissa >= 0.: # Otherwise the natural frequency is not defined (negative or zero stiffness coefficient).
        plt.axvline(x = EigenAbscissa, linewidth = 2, linestyle = "--", color = 'k')
    plt.ylabel(ylabel1, fontsize=18)
    if (save == False): # The title is not necessary for the generation of automatic report.
        plt.title(title, fontsize=20)
    plt.grid()
    plt.legend()

    # Phase.
    plt.subplot(2, 1, 2)
    for ibeta in range(0, beta.shape[0]):
        plt.plot(Abscissa, np.angle(data[iw_min:, ibeta], deg=True), linestyle="-", linewidth=2, color = colors[ibeta])
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    # Show and save.
    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    return EigenAbscissa


def plotKochinElem(data, Angle, DifforRad, w, ibody, iforce, beta, **kwargs):
    """Plots the elementary Kochin functions.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: elementary Kochin function.
    Angle : Array of floats.
        Angular discretization.
    DifforRad : Bool
        Boolean to know if the Kochin function comes from a diffraction (T) or a radiation (F) problem.
    w : float.
        Wave frequency.
    ibody : int.
        The index of the body.
    iforce : int.
        The index of the body's force mode.
    beta : float.
        Wave direction in radians.
    kwargs: optional.
        Arguments that are to be used by pyplot
    """

    # Labels and title.
    xlabel = r'$\theta$' + ' $(rad)$'
    if(DifforRad == 0): # Diffraction problem.
        ylabel1 = r'$|K(\theta)|$'
        ylabel2 = r'$Arg(K(\theta))$'
        title = r'Diffraction Kochin function for a wave frequency of %.1f rad/s and a wave direction of %.1f deg' % \
                (w, np.degrees(beta))
    else: # Radiation problem.
        ylabel1 = r'$|K(\theta)|$'
        ylabel2 = r'$Arg(K(\theta))$'
        title = r'Radiation Kochin function due to the %s motion of body %u for a wave frequency of %.1f rad/s' % \
                (Dof_name[iforce], ibody+1, w)

    # Plots.
    plt.subplot(2, 1, 1)
    plt.plot(Angle, np.absolute(data), linestyle="-", linewidth=2)
    plt.ylabel(ylabel1, fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(Angle, np.angle(data, deg=True), linestyle="-", linewidth=2)
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    plt.show()


def plotKochin(data, Angle, w, beta, DeriveOrNot, **kwargs):
    """Plots the total Kochin functions.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: elementary Kochin function.
    Angle : Array of floats.
        Angular discretization.
    w : float.
        Wave frequency.
    beta : float.
        Wave direction in radians.
    kwargs: optional.
        Arguments that are to be used by pyplot
    """

    # Labels and title.
    xlabel = r'$\theta$' + ' $(rad)$'
    if(DeriveOrNot == 0):
        ylabel1 = r'$|K(\theta)|$'
        ylabel2 = r'$Arg(K(\theta))$'
        title = r'Total Kochin function for a wave frequency of %.1f rad/s and a wave direction of %.1f deg' % \
                (w, np.degrees(beta))
    else:
        ylabel1 = r'$\left|\dfrac{dK}{d\theta}(\theta)\right|$'
        ylabel2 = r'$Arg\left(\dfrac{dK}{d\theta}(\theta)\right)$'
        title = r'Total angular derivative Kochin function for a wave frequency of %.1f rad/s and a wave direction of %.1f deg' % \
                (w, np.degrees(beta))

    # Plots.
    plt.subplot(2, 1, 1)
    plt.plot(Angle, np.absolute(data), linestyle="-", linewidth=2)
    plt.ylabel(ylabel1, fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(Angle, np.angle(data, deg=True), linestyle="-", linewidth=2)
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    plt.show()


def plotDrift(data, w, beta, iforce, show=True, save=False, is_period=False, filename="Drift.png"):
    """Plots the drift loads.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: elementary Kochin function.
    w : Array of floats.
        Wave frequencies.
    beta : float.
        Wave direction in radians.
    iforce : int.
        Degree of freedom (surge, sway or yaw).
    """

    # Labels and title.
    if is_period:
        xlabel = r'$T$' + '$ (s)$'
        xvar = 'T'
    else:
        xlabel = r'$\omega$' + ' $(rad/s)$'
        xvar = '\omega'

    if (iforce <= 2):
        force_str = 'force'
    else:
        force_str = 'moment'

    ylabel1 = r'$F_D({},\beta)$'.format(xvar)

    title = r'Mean drift %s in %s for a wave direction of %.1f deg' % \
            (force_str, Dof_name[iforce], np.degrees(beta))

    # Plots.
    if (save == False):  # The size is smaller for the generation of automatic report because the title is not including.
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))

    if is_period:
        plt.plot(2.*np.pi / w, data, linestyle="-", linewidth=2)
    else:
        plt.plot(w, data, linestyle="-", linewidth=2)

    plt.ylabel(ylabel1, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)

    if (save == False):  # The title is not necessary for the generation of automatic report.
        plt.title(title, fontsize=20)

    plt.grid()

    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def plotDrift_all_wave_dir(data, w, beta, iforce, show=True, save=False, is_period=False, filename = "Drift.png"):
    """Plots the drift loads for all wave directions.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: elementary Kochin function.
    w : Array of floats.
        Wave frequencies.
    beta : float.
        Wave direction in degrees.
    iforce : int.
        Degree of freedom (surge, sway or yaw).
    """

    # Labels and title.
    if is_period:
        xlabel = r'$T$' + '$ (s)$'
        xvar = 'T'
    else:
        xlabel = r'$\omega$' + ' $(rad/s)$'
        xvar = '\omega'

    if (iforce <= 2):
        force_str = 'force'
    else:
        force_str = 'moment'

    title = r'Mean drift %s in %s' % \
            (force_str, Dof_name[iforce])

    ylabel1 = r'$C_{Drift}^{%s}({%s},\beta)$' % (Dof_notation[iforce], xvar)

    # Colors.
    colors = cm.jet(np.linspace(1, 0, beta.shape[0]))

    # Plots.
    if (save == False):  # The size is smaller for the generation of automatic report because the title is not including.
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))
    for ibeta in range(0, beta.shape[0]):
        if is_period:
            plt.plot(2.*np.pi/w, data[iforce, :, ibeta], linestyle="-", linewidth=2, label = str(beta[ibeta])+" deg", color = colors[ibeta])
        else:
            plt.plot(w, data[iforce, :, ibeta], linestyle="-", linewidth=2, label = str(beta[ibeta])+" deg", color = colors[ibeta])
    plt.ylabel(ylabel1, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    if (save == False):  # The title is not necessary for the generation of automatic report.
        plt.title(title, fontsize=20)
    plt.grid()
    plt.legend()

    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def plot_VF(data, w, ibody_force, iforce, ibody_motion, idof, Speed, show=True, save=False, filename="VF.png"):
    """Plots the the vector fitting approximation and the frequency-domain IRF of a given modes set.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: Frequency-domain IRF and vector-fitting approximation.
    w : Array of floats.
        Wave frequencies.
    ibody_force : int
        Index of the body where the radiation force is applied.
    iforce : int
        Index of the local body's force mode.
    ibody_motion : int
        Index of the body having a motion.
    idof : int
        Index of the local body's radiation mode (motion).
    """

    # Label.
    xlabel = r'$\omega$' + ' $(rad/s)$'
    ylabel1 = r'$\Re(H_{%s}^0)(\omega)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
    ylabel2 = r'$\Im(H_{%s}^0)(\omega)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
    if(Speed):
        ylabel1 = r'$\Re(H_{%s}^U)(\omega)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
        ylabel2 = r'$\Im(H_{%s}^U)(\omega)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))

    ylabel_tmp = ""
    if (iforce <= 2):
        force_str = 'force'
        if (idof <= 2):  # Translation.
            ylabel_tmp += r' $(kg/s)$'
            motion_str = 'translation'
        else:  # Rotation.
            ylabel_tmp += r' $(kg\,m/s)$'
            motion_str = 'rotation'
    else:
        force_str = 'moment'
        if (idof <= 2):  # Translation.
            ylabel_tmp += r' $(kg\,m/s)$'
            motion_str = 'translation'
        else:  # Rotation.
            ylabel_tmp += r' $(kg\,m^2/s)$'
            motion_str = 'rotation'
    ylabel1 += ylabel_tmp
    ylabel2 += ylabel_tmp

    title = r'Wave radiation frequency reponse at zero forward speed and its vector fitting approximation '+"\n"+f"of the {force_str} in {Dof_name[iforce]} " \
                                                                                                                 f"of body {ibody_force + 1} for a {motion_str} in {Dof_name[idof]} of body {ibody_motion + 1}" \
            +"\n"+r"$H^0(j\omega) = B(\omega) + j\omega[A(\omega) - A^{\infty}]$"
    if(Speed):
        title = r'Wave radiation frequency reponse proportional to the forward speed and its vector fitting approximation ' \
                + "\n" + f"of the {force_str} in {Dof_name[iforce]} of body {ibody_force + 1} for a {motion_str} in {Dof_name[idof]} of body {ibody_motion + 1}" \
                + "\n" + r"$H^U(j\omega) = (j / \omega)B(\omega) - [A(\omega) - A^{\infty}]$"

    plt.close()
    if (save == False):
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))

    # Real part.
    plt.subplot(2, 1, 1)
    plt.plot(w, data[:, 0].real, linestyle="None", marker="+", color = "b", label = "Wave radiation frequency reponse")
    plt.plot(w, data[:, 1].real, linestyle="-", color = "r", linewidth=2, label="Vector fitting approximation")
    plt.ylabel(ylabel1, fontsize=18)
    if (save == False):
        plt.title(title, fontsize=20)
    plt.grid()
    plt.legend()

    # Damping.
    plt.subplot(2, 1, 2)
    plt.plot(w, data[:, 0].imag, linestyle="None", marker="+", color = "b", linewidth=2)
    plt.plot(w, data[:, 1].imag, linestyle="-", color = "r", linewidth=2)
    plt.ylabel(ylabel2, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.grid()

    # Show and save.
    if (show == True):
        plt.show()
    if (save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_VF_array(data, w, ibody_force, ibody_motion, Speed):
    """Plots the the vector fitting approximation and the frequency-domain IRF per body.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: Combinaison of both added mass and damping coefficients.
    w : Array of floats.
        Wave frequencies.
    ibody_force : int
        Index of the body where the radiation force is applied.
    ibody_motion : int
        Index of the body having a motion.
    """

    # Title.
    title = r'Wave radiation frequency reponse ($FR$) at zero forward speed and its vector fitting approximation ($VF$) '+f"of body {ibody_force + 1} due to body {ibody_motion + 1}" \
            +"\n"+r"$H_{%s}^0(j\omega) = B_{%s}(\omega) + j\omega[A_{%s}(\omega) - A_{%s}^{\infty}]$" \
            % (str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1),
               str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1))
    if(Speed):
        title = r'Wave radiation frequency reponse ($FR$) propotional to the forward speed and its vector fitting approximation ($VF$) ' + f"of body {ibody_force + 1} due to body {ibody_motion + 1}" \
                + "\n" + r"$H_{%s}^U(j\omega) = (j / \omega)B_{%s}(\omega) - [A_{%s}(\omega) - A_{%s}^{\infty}]$" \
                % (str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1),
                   str(ibody_force + 1) + str(ibody_motion + 1), str(ibody_force + 1) + str(ibody_motion + 1))

    # Definition of the figure.
    plt.close()
    fig, axes = plt.subplots(6, 6, figsize=(16, 8.5))

    # Plot.
    for iforce in range(0, 6):
        for idof in range(0, 6):
            labelA = r'$H_{%s}^0 (\omega)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            if (Speed):
                labelA = r'$H_{%s}^U (\omega)$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
            plot_1 = axes[iforce, idof].plot(w, data[iforce, idof, :, 0].real, linestyle="None", marker="+", color="b")
            plot_2 = axes[iforce, idof].plot(w, data[iforce, idof, :, 1].real, linestyle="-", color="r", linewidth=2)
            plot_3 = axes[iforce, idof].plot(w, data[iforce, idof, :, 0].imag, linestyle="None", marker="+", color="g")
            plot_4 = axes[iforce, idof].plot(w, data[iforce, idof, :, 1].imag, linestyle="-", color="darkorange", linewidth=2)
            axes[iforce, idof].grid()
            axes[iforce, idof].set_title(labelA)

    line_labels = ["$\Re(H_{FR}^0)$", "$\Re(H_{VF}^0)$", "$\Im(H_{FR}^0)$", "$\Im(H_{VF}^0)$"]
    if (Speed):
        line_labels = ["$\Re(H_{FR}^U)$", "$\Re(H_{VF}^U)$", "$\Im(H_{FR}^U)$", "$\Im(H_{VF}^U)$"]
    fig.legend([plot_1, plot_2, plot_3, plot_4], # The line objects
               labels=line_labels, # The labels for each line
               loc="lower center", # Position of legend
               ncol = 4)

    # Title.
    plt.suptitle(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(top=0.88)

    # Show the plot.
    plt.show()


def plot_VF_multiple_coef(data, w, ibody_force, iforce, ibody_motion, show = True, save = False, filename = "AB.png"):
    """Plots the vector fitting of a given modes set.

    Parameters
    ----------
    data : Array of floats.
        Data to plot: vector fitting data.
    w : Array of floats.
        Wave frequencies.
    ibody_force : int
        Index of the body where the radiation force is applied.
    iforce : int
        Index of the local body's force mode.
    ibody_motion : int
        Index of the body having a motion.
    """

    # Labels.
    xlabel = r'$\omega$' + ' $(rad/s)$'
    ylabel1 = r'$\Re(H)(\omega)$'
    ylabel2 = r'$\Im(H)(\omega)$'

    # Colors.
    colors = cm.jet(np.linspace(0.9, 0, 6))

    # Definition of the figure.
    plt.close()
    if (save == False):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8.5))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))

    # Real part.
    legend_plot = [[] for i in range(6)]
    ax1bis = ax1.twinx()
    for idof in range(0, 6):
        labelA = r'$\Re(H_{%s})$' % (Dof_notation[iforce] + "_" + str(ibody_force + 1) + Dof_notation[idof] + "_" + str(ibody_motion + 1))
        if (iforce <= 2):
            if(idof <= 2): # Translation
                labelA = labelA + r' $(kg/s)$'
            else: # Rotation.
                labelA = labelA + r' $(kg\,m/s)$'
        else:
            if(idof <= 2): # Translation.
                labelA = labelA + r' $(kg\,m/s)$'
            else: # Rotation.
                labelA = labelA + r' $(kg\,m^2/s)$'
        if(idof <= 2): # Translation.
            legend_plot[idof] = ax1.plot(w, data[:, idof, 0].real, linestyle="-", linewidth=2, label=labelA, color=colors[idof])
            ax1.plot(w, data[:, idof, 1].real, linestyle="None", marker="+", markersize=10, mew=3, color=colors[idof])
        else: # Rotation.
            legend_plot[idof] = ax1bis.plot(w, data[:, idof, 0].real, linestyle="-", linewidth=2, label=labelA, color=colors[idof])
            ax1bis.plot(w, data[:, idof, 1].real, linestyle="None", marker="+", markersize=10, mew=3, color=colors[idof])

    # Units.
    if (iforce <= 2):
        ylabel1_ax1 = ylabel1 + r' $(kg/s)$' # Translation.
        ylabel1_ax1bis = ylabel1 + r' $(kg\,m/s)$' # Rotation.
    else:
        ylabel1_ax1 = ylabel1 + r' $(kg\,m/s)$' # Translation.
        ylabel1_ax1bis = ylabel1 + r' $(kg\,m^2/s)$' # Rotation.
    ax1.set_ylabel(ylabel1_ax1, fontsize=18)
    ax1bis.set_ylabel(ylabel1_ax1bis, fontsize=18)

    # Legend.
    legend = []
    for idof in range(0, 6):
        legend += legend_plot[idof]
    labs = [l.get_label() for l in legend]
    ax1.legend(legend, labs, fontsize=12, ncol=2)
    ax1.grid()

    # Imaginary part.
    legend_plot = [[] for i in range(6)]
    ax2bis = ax2.twinx()
    for idof in range(0, 6):
        labelB = r'$\Im(H_{%s})$' % (Dof_notation[iforce]+"_"+str(ibody_force+1) + Dof_notation[idof]+"_"+str(ibody_motion+1))
        if (iforce <= 2):
            if(idof <= 2): # Translation.
                labelB = labelB + r' $(kg/s)$'
            else: # Rotation.
                labelB = labelB + r' $(kg\,m/s)$'
        else:
            if(idof <= 2): # Translation.
                labelB = labelB + r' $(kg\,m/s)$'
            else: # Rotation.
                labelB = labelB + r' $(kg\,m^2/s)$'
        if (idof <= 2):  # Translation.
            legend_plot[idof] = ax2.plot(w, data[:, idof, 0].imag, linestyle="-", linewidth=2, label=labelB, color=colors[idof])
            ax2.plot(w, data[:, idof, 1].imag, linestyle="None", marker="+", markersize=10, mew=3, color=colors[idof])
        else:  # Rotation.
            legend_plot[idof] = ax2bis.plot(w, data[:, idof, 0].imag, linestyle = "-", linewidth = 2, label = labelB, color = colors[idof])
            ax2bis.plot(w, data[:, idof, 1].imag, linestyle="None", marker="+", markersize=10, mew=3, color=colors[idof])

    # Units.
    if (iforce <= 2):
        ylabel2_ax2 = ylabel2 + r' $(kg/s)$' # Translation.
        ylabel2_ax2bis = ylabel2 + r' $(kg\,m/s)$' # Rotation.
    else:
        ylabel2_ax2 = ylabel2 + r' $(kg\,m/s)$' # Translation.
        ylabel2_ax2bis = ylabel2 + r' $(kg\,m^2/s)$' # Rotation.
    ax2.set_ylabel(ylabel2_ax2, fontsize=18)
    ax2bis.set_ylabel(ylabel2_ax2bis, fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.grid()

    # Legend.
    legend = []
    for idof in range(0, 6):
        legend += legend_plot[idof]
    labs = [l.get_label() for l in legend]
    ax2.legend(legend, labs, fontsize=12, ncol=2)

    if (show == True):
        plt.show()
    if(save == True):
        plt.tight_layout()
        plt.savefig(filename)
    plt.close()


def plot_wave_spectrum(Hs, Tp, gamma, rads_or_period, show = True, save = False, filename = "Wave_spectrum.png"):

    # This method plots the wave spectrums.

    # Labels and title.
    if rads_or_period is True: # Wave frequency.
        xlabel = r'$\omega (rad/s)$'
        ylabel = r'$S(\omega)$ $(m^2.s)$'
    else: # Wave period.
        xlabel = r'$T (s)$'
        ylabel = r'$S(T)$ $(m^2/s)$'
    title = r'JONSWAP wave spectrums'

    # Colors.
    colors = cm.jet(np.linspace(1, 0, Hs.shape[0]))

    # Plots.
    if save == False: # The size is smaller for the generation of automatic report because the title is not including.
        plt.figure(num=None, figsize=(16, 8.5))
    else:
        plt.figure(num=None, figsize=(10, 6))
    nPts = 1000
    if rads_or_period is True:
        w_max = 2.5 * 2 * np.pi / min(Tp) # w_max = 2.5 * max(w_p).
        abscissa = np.linspace(0, w_max, nPts) # Wave frequency.
    else:
        abscissa = np.linspace(0.01, 20, nPts) # Wave period.
    for iHs in range(0, Hs.shape[0]):

        # Label.
        label = r"Hs = %.1f m - Tp = %.1f s - $\gamma$ = %.1f" % (Hs[iHs], Tp[iHs], gamma[iHs])

        # Wave spectrum.
        wave_spectrum = JonswapWaveSpectrum(Hs[iHs], Tp[iHs], gamma[iHs])
        if rads_or_period is True: # Wave frequency.
            data = wave_spectrum.eval(abscissa)
        else: # Wave period.
            data = (2. * np.pi / pow(abscissa, 2)) * wave_spectrum.eval(2 * np.pi / abscissa)

        # Plot.
        plt.plot(abscissa, data, linestyle="-", linewidth=2, label=label, color=colors[iHs])

    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    if save == False: # The title is not necessary for the generation of automatic report.
        plt.title(title, fontsize=20)
    plt.grid()
    plt.legend()

    if show == True:
        plt.show()
    if save == True:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def Meshmagick_viewer(mesh):
    """This function plots a mesh."""

    mesh_vizu = copy.deepcopy(mesh)
    vtk_polydata = mesh_vizu._vtk_polydata()
    mesh_vizu.viewer = meshmagick.MMviewer.MMViewer()
    mesh_vizu.viewer.add_polydata(vtk_polydata)
    mesh_vizu.viewer.renderer.ResetCamera()
    mesh_vizu.viewer.render_window.Render()
    mesh_vizu.viewer.render_window_interactor.Start()
    mesh_vizu.viewer.finalize()

