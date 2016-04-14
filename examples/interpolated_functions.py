import numpy as np


#~ import int_field_for as iff
#~
#~
#~ def interpolated_function(fn, xmin, xmax, steps):
    #~ x_map = np.linspace(xmin, xmax, steps)
    #~ fn_map = fn(x_map)
    #~ # print 'I finished calculating the map of your function'
    #~ bias_x = np.min(x_map)
    #~ dx = x_map[1]- x_map[0]
      #~
    #~ def function(x):
         #~ return iff.int_field(x, bias_x, dx, fn_map)
         # return np.interp(x, x_map, fn_map)
    #~ return function
    #~
#~ sin_interpolated = interpolated_function(np.sin, -2.1 * np.pi, 2.1 * np.pi, 2000000)
#~ cos_interpolated = interpolated_function(np.cos, -2.1 * np.pi, 2.1 * np.pi, 2000000)
#~



import interpolate as ip

def interpolated_mod2pi(fn, xmin, xmax, steps):
    x_map = np.linspace(xmin, xmax, steps)
    fn_map = fn(x_map)
    # print 'I finished calculating the map of your function'
    bias_x = np.min(x_map)
    dx = x_map[1]- x_map[0]
    def function(x):
        return np.reshape(ip.interpolate(np.mod(x, 2 * np.pi), bias_x, dx, fn_map), x.shape)
    return function


# sin_interpolated = interpolated_mod2pi(np.sin, -2.1 * np.pi, 2.1 * np.pi, 2000000)
# cos_interpolated = interpolated_mod2pi(np.cos, -2.1 * np.pi, 2.1 * np.pi, 2000000)
