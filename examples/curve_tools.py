
# coding: utf-8

# In[111]:

import numpy as np
import pylab as plt

from scipy.optimize import brentq


# In[112]:

def zero_crossings(f, x):
    y = f(x)
    zix = np.where(np.abs(np.diff(np.sign(y))) == 2)[0]

    x0 = np.array([brentq(f, x[i], x[i+1]) for i in zix])
    y0 = np.array([f(i) for i in x0])

    return x0, y0


# In[104]:

def extrema(x, y=None):
    zix = np.where(np.abs(np.diff(np.sign(np.diff(x)))) == 2)[0]
    if not y:
        print zix
        return zix


# In[1]:

# %matplotlib inline

# z = np.linspace(-2*np.pi, 2*np.pi, 100)
# y = lambda x: np.sin(x)
# plt.plot(z, y(z), '-x')

# z0, y0 = zero_crossings(y, z)
# plt.plot(z0, y0, 'ro')
# print z0

# plt.show()

