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
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_function

class WaveSpectrum():
    """Class for dealing with wave spectrums."""

    def __init__(self, Hs, Tp):
        """Constructor of the class."""

        # Significant wave height (m).
        self._Hs = Hs

        # Peak period (s).
        self._Tp = Tp

    def get_wave_component_amplitudes(self, nw, wmin, wmax):
        """This method computes the wave amplitudes corresponding to a discretization of the wave spectrum."""

        w = np.linspace(wmin, wmax, nw)
        dw = w[1] - w[0]
        s_w = self.eval(w, unit=unit)

        return np.sqrt(2. * s_w * dw)

    def eval(self, wave_frequency_rads):
        raise NotImplementedError('You must use a specialized version of WaveSpectrum, not directly WaveSpectrum')

    def plot(self):
        """This method plots the wave spectrum with respect to the wave frequency."""

        # Wave frequency.
        wave_frequency_rads = np.linspace(0., 10., 2000)

        # Computation of the wave spectrum.
        spectrum = self.eval(wave_frequency_rads)

        # Plot.
        xlabel = r'$\omega\;(rad/s)$'
        ylabel = r'$S(\omega)\;(m^2/Hz)$'
        plt.plot(wave_frequency_rads, spectrum)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.grid(axis='both')
        plt.show()

class JonswapWaveSpectrum(WaveSpectrum):
    """Class for representing the modified two-parameter Jonswap wave spectrum."""

    # Constant indicating average width to the left of the wave spectrum.
    _inverse_sigma2_left = 1. / pow(0.07, 2.)

    # Constant indicating average width to the right of the wave spectrum.
    _inverse_sigma2_right = 1. / pow(0.09, 2.)

    def __init__(self, Hs, Tp, gamma = 3.3):

        # Constructor of the mother class.
        super(JonswapWaveSpectrum, self).__init__(Hs, Tp)

        # Checking the value of gamma.
        self._check_gamma(gamma)
        self._gamma = gamma

    @property
    def gamma(self):
        """Getter of the gamma peakedness factor"""

        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """Set the gamma peakedness factor"""
        self._check_gamma(gamma)
        self._gamma = gamma

    @staticmethod
    def _check_gamma(gamma):
        """This method checks the consistency of the gamma factor."""

        if not 1. <= gamma <= 10.:
            warn('JONSWAP wave spectrum: gamma parameter for usually lies between 1 and 10. Given: %.2f') % gamma

    def eval(self, wave_frequency_rads):
        """This method computes the Jonswap wave spectrum for the given wave frequencies."""

        if np.isscalar(wave_frequency_rads):
            wave_frequency_rads = np.array([wave_frequency_rads], dtype=np.float64)

        # Wave frequency as array is in rad/s
        w = np.asarray(wave_frequency_rads, dtype=np.float64)

        # Wave spectrum parameters.
        gamma = self._gamma
        hs = self._Hs
        wp = 2. * np.pi / self._Tp

        # Temporary data.
        wp2 = wp * wp
        wp4 = wp2 * wp2
        left = w <= wp # Left part of the spectrum.
        right = w > wp # Right part of the spectrum.

        # Computation.
        with np.errstate(divide='ignore', invalid='ignore'):

            # TODO: w != 0.
            inverse_w4 = np.power(w, -4)
            inverse_w5 = inverse_w4 / w
            spectrum = 0.3125 * hs * hs * wp4 * inverse_w5 * np.exp(- 1.25 * wp4 * inverse_w4) * (1. - 0.287 * np.log(gamma)) # 0.3125 = 5/16.

            # Exponent of gamma.
            gamma_exponent = np.exp(-np.power(w - wp, 2) / (2. * wp2))
            gamma_exponent[left] **= self._inverse_sigma2_left
            gamma_exponent[right] **= self._inverse_sigma2_right

            # Application of the peak enhancement factor.
            spectrum[left] *= np.power(gamma, gamma_exponent[left])
            spectrum[right] *= np.power(gamma, gamma_exponent[right])

            # Spectrum is zero outside the given wave frequencies.
            spectrum[np.isnan(spectrum)] = 0.

        return spectrum

class ITTCWaveSpectrum(WaveSpectrum):

    """Class for representing the ITTC 1978 wave spectrum."""

    def __init__(self, Hs, Tp, gamma = 3.3):

        # Constructor of the mother class.
        super(ITTCWaveSpectrum, self).__init__(Hs, Tp)

    def eval(self, wave_frequency_rads):
        """This method computes the ITTC 1978 wave spectrum for the given wave frequencies."""

        if np.isscalar(wave_frequency_rads):
            wave_frequency_rads = np.array([wave_frequency_rads], dtype=np.float64)

        # Wave frequency as array is in rad/s
        w = np.asarray(wave_frequency_rads, dtype=np.float64)

        # Wave spectrum parameters.
        hs = self._Hs
        Tp = self._Tp

        # Temporary data.
        Tp4 = pow(Tp, 4.)
        A = 487. * hs * hs / Tp4
        B = 1945. / Tp4

        # Computation.
        with np.errstate(divide='ignore', invalid='ignore'):

            # TODO: w != 0.
            inverse_w4 = np.power(w, -4)
            inverse_w5 = inverse_w4 / w
            spectrum = A * inverse_w5 * np.exp(- B * inverse_w4)

            # Spectrum is zero outside the given wave frequencies.
            spectrum[np.isnan(spectrum)] = 0.

        return spectrum


class DirectionalWaveSpectrum(WaveSpectrum):
    """Class for dealing with directional wave spectrums of type cos2s."""

    def __init__(self, wave_spectrum, theta_mean_deg, spreading_factor = 10):

        # Wave spectrum.
        self._wave_spectrum = wave_spectrum

        # Mean wave direction.
        self._theta_mean_rad = np.radians(theta_mean_deg)

        # Spreading factor.
        self._check_spreading_factor(spreading_factor)
        self._spreading_factor = spreading_factor

    @property
    def spreading_factor(self):
        return self._spreading_factor

    @spreading_factor.setter
    def spreading_factor(self, spreading_factor):
        self._check_spreading_factor(spreading_factor)
        self._spreading_factor = spreading_factor

    @staticmethod
    def _check_spreading_factor(spreading_factor):
        if not 1. <= spreading_factor <= 100.:
            raise print('DirectionalWaveSpectrum: spreading factor must lie between 1 and 100.')

    def get_spreading_function(self, theta_rad):
        """This method evaluates the spreading function."""

        s = self.spreading_factor
        tmp = gamma_function(s + 1)
        nondimensional_factor = (2 ** (2. * s - 1.) / np.pi) * (tmp * tmp / gamma_function(2 * s + 1))
        spreading_function = nondimensional_factor * np.power(np.cos((theta_rad - self._theta_mean_rad) * 0.5), 2. * s)

        return spreading_function

    def plot_spreading_function(self):
        """This method plots the spreading function."""

        # Direction.
        theta_rad = np.linspace(-np.pi + self._theta_mean_rad, np.pi + self._theta_mean_rad, 200)

        # Spreading function.
        spreading_function = self.get_spreading_function(theta_rad)

        # Plot.
        xlabel = r'$\theta - \theta_m\;(°)$'
        ylabel = r'$D(\theta)\;(-)$'
        plt.plot(np.degrees(theta_rad - self._theta_mean_rad), spreading_function)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.grid(axis='both')
        plt.title('Angular wave distribution (model $cos^{2s}$ with $s=%u$)' % self.spreading_factor)
        plt.show()

    @property
    def hs(self):
        return self._wave_spectrum.hs

    @property
    def tp(self):
        return self._wave_spectrum.tp

    @property
    def wp(self):
        return self._wave_spectrum.wp

    def eval(self, wave_frequency_rads, theta_rad):
        """This method computes the direction wave spectrum."""

        wave_spectrum = self._wave_spectrum.eval(wave_frequency_rads)
        spreading_function = self.get_spreading_function(theta_rad)

        return np.einsum('i, j -> ij', spreading_function, wave_spectrum)