import atmosphere as atm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import constants as const

# import global constants
g = const.g # m/s2
re = const.re # m radius of earth
gamma = const.gamma # specific heat ratio
R = const.R # specific gas constant - air
H = const.H # scale height m

def ballistic_dynamics(t, z, vehicle):
    """ Dynamics for initial dive and pull up maneuver.

    Args:
        t: time use by ivp integration [s]
        z: state vector [fpa, x, y, v] [sr, m, m, m/s]
        vehicle: vehicle object

    Returns:
        dz_dt: state vector time derivative [dfpa_dt, dx_dt, dy_dt, dv_dt] [sr/s, m/s, m/s, m/s2]

    """
    fpa, x, y, v = z
    l_d = vehicle.l_d
    b = vehicle.b 

    dfpa_dt = 1/v * (-.5 * atm.get_density(y) * (v ** 2) * l_d /b + g * np.cos(fpa) - v ** 2 / (re + y) * np.cos(fpa))
    dx_dt = v * np.cos(fpa)
    dy_dt = -v * np.sin(fpa)
    dv_dt = -.5 * atm.get_density(y) * (v ** 2) / b + g * np.sin(fpa)
    
    return [dfpa_dt, dx_dt, dy_dt, dv_dt]

def dive_pull_dynamics(t, z, vehicle):
    """ Dynamics for initial dive and pull up maneuver.

    Args:
        t: time use by ivp integration [s]
        z: state vector [fpa, x, y, v] [sr, m, m, m/s]
        vehicle: vehicle object

    Returns:
        dz_dt: state vector time derivative [dfpa_dt, dx_dt, dy_dt, dv_dt] [sr/s, m/s, m/s, m/s2]

    """
    fpa, x, y, v = z
    l_d = vehicle.l_d
    b = vehicle.b 

    dfpa_dt = 1/v * (-.5 * atm.get_density(y) * (v ** 2) * l_d /b + g * np.cos(fpa) - v ** 2 / (re + y) * np.cos(fpa))
    dx_dt = v * np.cos(fpa)
    dy_dt = -v * np.sin(fpa)
    dv_dt = -.5 * atm.get_density(y) * (v ** 2) / b + g * np.sin(fpa)
    
    return [dfpa_dt, dx_dt, dy_dt, dv_dt]

def glide_dynamics(t, z, v0, y0, vehicle):
    """ Dynamics for constant dynamic pressure glide.

    Args:
        t: time use by ivp integration [s]
        z: state vector [x, y] [m, m]
        v0: initial velocity (end velocity from pull-up) [m/s]
        y0: initial altitude (end altitude from pull-up) [m]
        vehicle: vehicle object

    Returns:
        dz_dt: state vector time derivative [dx_dt, dy_dt] [m/s, m/s]

    """
    x, y = z
    l_d = vehicle.l_d
    b = vehicle.b 

    v = np.sqrt(atm.get_density(y0)/atm.get_density(y)) * v0
    fpa =  1 / l_d / (1 + v ** 2 / 2 / g / H)

    dx_dt = v * np.cos(fpa)
    dy_dt = -v * np.sin(fpa)

    return [dx_dt, dy_dt]
