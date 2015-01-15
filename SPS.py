from __future__ import division
import numpy as np
from scipy.constants import c, e, m_p

from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.simple_long_tracking import LinearMap, RFSystems


class SPS(object):

    def __init__(self, gamma, optics='Q20', n_segments=1, focusing='linear'):

        self.circumference = 1100*2*np.pi
        self.gamma = gamma

        self.optics = optics
        self.n_segments = n_segments
        self.focusing = focusing

        self.create_transverse_map()
        self.create_longitudinal_map()

        self.one_turn_map = []
        for m in self.transverse_map:
            self.one_turn_map.append(m)
        self.one_turn_map.append(self.longitudinal_map)

    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma**-2)
        self._betagamma = np.sqrt(self.gamma**2 - 1)
        self._p0 = self.betagamma * m_p * c

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1 - value ** 2)

    @property
    def betagamma(self):
        return self._betagamma
    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value**2 + 1)

    @property
    def p0(self):
        return self._p0
    @p0.setter
    def p0(self, value):
        self.gamma = value / (m_p * self.beta * c)

    def create_transverse_map(self):

        s = np.arange(0, self.n_segments + 1) * self.circumference / self.n_segments

        if self.optics=='Q20':
            alpha_x        = 0 * np.ones(self.n_segments)
            beta_x         = 54.6 * np.ones(self.n_segments)
            D_x            = 0 * np.ones(self.n_segments)
            alpha_y        = 0 * np.ones(self.n_segments)
            beta_y         = 54.6 * np.ones(self.n_segments)
            D_y            = 0 * np.ones(self.n_segments)

            Q_x            = 20.13
            Q_y            = 20.18

            Qp_x           = 0
            Qp_y           = 0

            app_x          = 0.0000e-9
            app_y          = 0.0000e-9
            app_xy         = 0

            self.alpha     = 0.00192
        elif self.optics=='Q26':
            alpha_x        = [0] * np.ones(self.n_segments)
            beta_x         = [42] * np.ones(self.n_segments)
            D_x            = 0 * np.ones(self.n_segments)
            alpha_y        = [0] * np.ones(self.n_segments)
            beta_y         = [42] * np.ones(self.n_segments)
            D_y            = 0 * np.ones(self.n_segments)

            Q_x            = 26.13
            Q_y            = 26.18

            Qp_x           = 0
            Qp_y           = 0

            app_x          = 0.0000e-9
            app_y          = 0.0000e-9
            app_xy         = 0

            self.alpha     = 0.00308
        else:
            raise ValueError('ERROR: unknown optics', self.optics)

        self.transverse_map = TransverseMap(
            self.circumference, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
            Chromaticity(Qp_x, Qp_y),
            AmplitudeDetuning(app_x, app_y, app_xy))

    def create_longitudinal_map(self):

        self.h1, self.h2          = 4620, 4620*4
        self.dphi1, self.dphi2    = 0, np.pi
        # p_increment     = 0 * e/c * self.circumference/(beta*c)

        if self.optics=='Q20':
            self.V1, self.V2      = 5.75e6, 0#*self.V1/10
        elif self.optics=='Q26':
            self.V1, self.V2      = 2e6, 0#*self.V1/10
        else:
            raise ValueError('ERROR: unknown optics', self.optics)

        if self.focusing=='linear':
            eta = self.alpha - self.gamma**-2
            beta = np.sqrt(1 - self.gamma**-2)
            p0 = np.sqrt(self.gamma**2 - 1) * m_p * c
            Q_s = np.sqrt( e*np.abs(eta)*(self.h1*self.V1 + self.h2*self.V2)
                         /(2*np.pi*p0*beta*c) )

            self.longitudinal_map = LinearMap([self.alpha], self.circumference, self.Q_s())
        elif self.focusing=='non-linear':
            self.longitudinal_map = RFSystems(self.circumference, [h1, h2], [V1, V2], [dphi1, dphi2],
                                        [self.alpha], gamma)
        else:
            raise ValueError('ERROR: unknown focusing', self.focusing)

    def Q_s(self):
        eta = self.alpha - self.gamma**-2
        return np.sqrt( e*np.abs(eta)*(self.h1*self.V1 + self.h2*self.V2)
                        /(2*np.pi*self.p0*self.beta*c) )

    def track(self, bunch):
        for m in self.one_turn_map:
            m.track(bunch)
