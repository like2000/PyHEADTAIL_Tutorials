from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p
from scipy.optimize import curve_fit
from scipy.integrate import quad, dblquad

from PyHEADTAIL.trackers.rf_bucket import RFBucket
from PyHEADTAIL.particles.generators import ParticleGenerator
from PyHEADTAIL.particles.generators import gaussian2D, RF_bucket_distribution

import seaborn as sns
plt.switch_backend('TkAgg')
sns.set_context('notebook', font_scale=1.5,
                rc={'lines.markeredgewidth': 1})
sns.set_style('darkgrid', {
        'axes.linewidth': 2,
        'legend.fancybox': True})


def quad2d(f, ylimits, xmin, xmax):
    Q, error = dblquad(lambda y, x: f(x, y), xmin, xmax,
                       lambda x: -ylimits(x), lambda x: ylimits(x))

    return Q


# PARAMETERS
# ==========
p0 = 7000e9 * e/c
E0 = p0*c
gamma = np.sqrt((p0/(m_p*c))**2 + 1)
beta = np.sqrt(1 - gamma**-2)

C = 26658.883
R = C/(2*np.pi)
T0 = C/(beta*c)
omega0 = 2*np.pi/T0
alpha = 53.86**-2
# alpha = 3.225e-4
eta = alpha - gamma**-2

# 200 MHz
V_RF = [6e6, 0*-3e6]
h_RF = [17820, 35640]
dphi_RF = [0, 0*np.pi]
epsn_z = 3.8
zcut = 0.1500 * 2

# 400 MHz
V_RF = [16e6, 0*8e6]
h_RF = [35640, 71280]
dphi_RF = [0, 0*np.pi]
epsn_z = 2.5
zcut = 0.0810 * 2
# zcut = 0.0755 * 2


V, h, phi = V_RF[0], h_RF[0], dphi_RF[0]
p_increment = 0 * e/c * C/(beta*c)
phi_c = h*zcut/R
phi_s = np.pi


# THEO BUCKET
# ===========
def T(DE):
    return 1/2.*eta*h*omega0**2/(beta**2*E0) * (DE/omega0)**2


def U(phi):
    return (e/(2*np.pi)*V *
            (np.cos(phi) - np.cos(phi_s) + (phi - phi_s)*np.sin(phi_s)))


def hamiltonian(phi, DE):
    return -(T(DE) - U(phi))


def eqH(phi_c):
    def f(phi):
        Hc = hamiltonian(phi_c, 0)
        return (np.sqrt(2*beta**2*E0/(eta*h*omega0**2) *
                        (U(phi) - Hc)))
    return f


# PyHEADTAIL BUCKET
# =================
rfbucket = RFBucket(
    circumference=C, charge=e, mass=938e6*e*c**-2, gamma=gamma,
    alpha_array=[alpha], p_increment=0.,
    harmonic_list=h_RF, voltage_list=V_RF, phi_offset_list=dphi_RF)
Qs = rfbucket.Qs

# A = dblquad(lambda y, x: 1, -phi_c, +phi_c, lambda x: 0, eqH(phi_c))
# A = quad2d(lambda y, x: 1, eqH(phi_c), -phi_c, +phi_c)
A = quad(eqH(phi_c), -phi_c, +phi_c)
print A

# COMPARE BUCKETS
print "\nEmittance 1: {:g}".format(rfbucket.emittance_sp(z=zcut))
print "Emittance 2: {:g}".format(2*A[0]/e/h)
print "Emittance Gauss: {:g}".format(4*np.pi * (zcut/2.)**2 * Qs/eta/R * p0/e)
# print "Bucket area 1: {:g}".format(rfbucket.emittance_sp())


print "\nCanonically conjugate pair 1: z={:1.2e}, dp/p={:1.2e}".format(
    zcut/2., rfbucket.dp_max(zcut)/2.)
print "Canonically conjugate pair 2: z={:1.2e}, dp/p={:1.2e}".format(
    zcut/2., eqH(phi_c)(0)*omega0/E0/beta**2/2.)


wurstel


# PLOT BUCKETS
# ============
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 10))

zz = np.linspace(1.1*rfbucket.zleft, 1.1*rfbucket.zright, 200)
pp = np.linspace(-1.5*rfbucket.dp_max(rfbucket.zleft),
                 +1.5*rfbucket.dp_max(rfbucket.zleft), 200)
phi = h*zz/R
hh = rfbucket.hamiltonian

ZZ, PP = np.meshgrid(zz, pp)
PH, EE = h*ZZ/R, PP*E0*beta**2
print "\nHmax 1: {:e}".format(np.max(hh(ZZ, PP)))
print "\nHmax 2: {:e}".format(np.max(hamiltonian(PH, EE)*omega0))

ax1.contourf(ZZ, PP, hh(ZZ, PP), 10, cmap=plt.cm.viridis)
ax1.contour(ZZ, PP, hh(ZZ, PP), levels=[hh(rfbucket.zright, 0), hh(zcut, 0)],
            colors='orange')
# ax1.plot(zcut, 0, '*', ms=20)
# ax1.contour(ZZ, PP, hh(ZZ, PP), 10, cmap=plt.cm.rainbow)
# ax1.contour(ZZ, PP, hamiltonian(PH, EE), 10, cmap=plt.cm.rainbow)
ax1.plot(zz, +eqH(phi_c)(phi)*omega0/E0/beta**2, c='red', lw=2, ls='--')
ax1.plot(zz, -eqH(phi_c)(phi)*omega0/E0/beta**2, c='red', lw=2, ls='--')

ax2.contourf(ZZ, PP, hamiltonian(PH, EE), 10, cmap=plt.cm.viridis)
ax2.contour(ZZ, PP, hamiltonian(PH, EE),
            levels=sorted([hamiltonian(h*zcut/R, 0),
                           hamiltonian(h*rfbucket.zright/R, 0)]))
# ax2.plot(zcut, 0, '*', ms=20)
# ax2.plot(zz, +eqH(phi_c)(h*zz/R)*omega0/E0, c='red', lw=2)
# ax2.plot(zz, -eqH(phi_c)(h*zz/R)*omega0/E0, c='red', lw=2)


plt.draw()
plt.pause(1)
plt.close('all')


# PARTICLE DISTRIBUTION
# =====================
bunch2 = ParticleGenerator(
    macroparticlenumber=1e6, intensity=1e11,
    charge=e, mass=m_p, gamma=gamma,
    circumference=C,
    distribution_x=gaussian2D(2e-6),
    distribution_y=gaussian2D(2e-6),
    distribution_z=RF_bucket_distribution(rfbucket,
                                          sigma_z=zcut/2.)).generate()

print "\n\nBunch length: {:1.2e}, momentum spread: {:1.2e}, emittance: {:1.2e}".format(
    bunch2.sigma_z(), bunch2.sigma_dp(), bunch2.epsn_z())


wurstel




eps_geo_z = epsn_z * e/(4*np.pi*p0)
bunch1 = ParticleGenerator(
    macroparticlenumber=2e6, intensity=1e11,
    charge=e, mass=m_p, gamma=gamma,
    circumference=C,
    distribution_x=gaussian2D(2e-6),
    distribution_y=gaussian2D(2e-6),
    distribution_z=gaussian2D(eps_geo_z), Qs=Qs, eta=eta).generate()

bunch2 = ParticleGenerator(
    macroparticlenumber=2e6, intensity=1e11,
    charge=e, mass=m_p, gamma=gamma,
    circumference=C,
    distribution_x=gaussian2D(2e-6),
    distribution_y=gaussian2D(2e-6),
    distribution_z=RF_bucket_distribution(rfbucket, epsn_z=epsn_z)).generate()

print "\n\nBunch1 length: {:g}, momentum spread: {:g}".format(bunch1.sigma_z(),
                                                          bunch1.sigma_dp())
print "Bunch2 length: {:g}, momentum spread: {:g}".format(bunch2.sigma_z(),
                                                          bunch2.sigma_dp())
print "Synchrotron tune {:g}".format(Qs)
print "\nBunch length ratios {:g}".format(bunch1.sigma_z()/bunch2.sigma_z())




def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


fig, (ax1, ax2) = plt.subplots(2, sharex=True)

hist1, bins1, p = ax1.hist(bunch1.z, 100, normed=True)
hist2, bins2, p = ax2.hist(bunch2.z, 100, normed=True)
bins1 = (bins1[:-1] + bins1[1:])/2.
bins2 = (bins2[:-1] + bins2[1:])/2.

p0 = [1., 0., 1.]
coeff1, var_matrix = curve_fit(gauss, bins1, hist1, p0=p0)
coeff2, var_matrix = curve_fit(gauss, bins2, hist2, p0=p0)

ax1.plot(bins1, gauss(bins1, *coeff1), '-')
ax2.plot(bins2, gauss(bins2, *coeff2), '-')

ax1.text(0.1, 0.9, "$\sigma$ fit: {:g}".format(coeff1[-1]),
         transform=ax1.transAxes, fontsize=20)
ax2.text(0.1, 0.9, "$\sigma$ fit: {:g}".format(coeff2[-1]),
         transform=ax2.transAxes, fontsize=20)

plt.show()
