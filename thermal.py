import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad, dblquad
from scipy.optimize import fsolve
import constants as const
import atmosphere as atm

c = const.c # speed of light [m/s]
h = const.h # Planck's constant [m2 kg / s]
kb = const.kb # Boltzmann's constant [m2 kg s-2 K-1]

# aerothermal heating
# correlations due to Tauber, taken from Tracy & Wright

def stagnation_heating(Tw, y, v, rn):

	hw = v ** 2 + 2.3e5
	h0 = 1e3 * Tw
	
	q = np.abs(1.83e-4 / np.sqrt(rn) * (1 - (hw / h0)) * atm.get_density(y) * v ** 3)

	return q

def convective_heating(Tw, y, v, pos, theta):

	hw = v ** 2 + 2.3e5
	h0 = 1e3 * Tw
	
	if v > 4e3:
		q = np.abs(2.2e-5*(np.cos(np.radians(theta)) ** 2.08 * np.sin(np.radians(theta)) ** 1.6) / (pos ** 0.2) * (1 - 1.11 * hw / h0) * atm.get_density(y) ** 0.8 * v ** 3.7)
	else:
		q = np.abs(3.89e-4*(np.cos(np.radians(theta)) ** 1.78 * np.sin(np.radians(theta)) ** 1.6) / (pos ** 0.2) * (1 - 1.11 * hw / h0) * (556/Tw) ** 0.25 * atm.get_density(y) ** 0.8 * v ** 3.37)
	
	return q

def radiatve_heating(Tw):

	sigma = 5.67e-8
	e = 1

	return e * sigma * pow(Tw,4)

def equilibrium_wall_temperature_convective(y, v, pos, theta):

	a = 1
	b = 8e4

	Nmax = 50
	N = 0
	tol = 1e-6

	while N < Nmax:

		Tw = (a + b) / 2

		q_conv_a = convective_heating(a, y, v, pos, theta)
		q_conv_b = convective_heating(b, y, v, pos, theta)
		q_conv_Tw = convective_heating(Tw, y, v, pos, theta)

		q_rad_a = radiatve_heating(a)
		q_rad_b = radiatve_heating(b)
		q_rad_Tw = radiatve_heating(Tw)

		if q_conv_Tw - q_rad_Tw == 0 or (b - a) / 2 < tol:
			break

		if (q_conv_Tw - q_rad_Tw > 0 and q_conv_Tw - q_rad_a > 0) or (q_conv_a - q_rad_Tw < 0 and q_conv_a - q_rad_a < 0):
			a = Tw  
		else:
			b = Tw
	
	return Tw

def equilibrium_wall_temperature_stagnation(y, v, rn):

	a = 1
	b = 8e4

	Nmax = 50
	N = 0
	tol = 1e-6

	while N < Nmax:

		Tw = (a + b) / 2

		q_stag_a = stagnation_heating(a, y, v, rn)
		q_stag_b = stagnation_heating(b, y, v, rn)
		q_stag_Tw = stagnation_heating(Tw, y, v, rn)

		q_rad_a = radiatve_heating(a)
		q_rad_b = radiatve_heating(b)
		q_rad_Tw = radiatve_heating(Tw)

		if q_stag_Tw - q_rad_Tw == 0 or (b - a) / 2 < tol:
			break

		if (q_stag_Tw - q_rad_Tw > 0 and q_stag_Tw - q_rad_a > 0) or (q_stag_a - q_rad_Tw < 0 and q_stag_a - q_rad_a < 0):
			a = Tw  
		else:
			b = Tw
	
	return Tw

# radiation emission due to surface heating

def spectral_radiance_wavelength(wavelength, Tw):
	""" Computes spectral radiance of emitted radiation as a function of wavelength and wall temperature.

	Args:
		wavelength: [m]
		wall temperature: [K]
	
	Returns:
		spectral radiance: [W m-3 sr-1]

	"""

	L = ((2 * h * c ** 2) / (wavelength ** 5)) * ( 1 / (np.exp(h * c / kb / Tw / wavelength) - 1))

	return L

def spectral_radiance_frequency(frequency, Tw):
	""" Computes spectral radiance of emitted radiation as a function of frequency and wall temperature.

	Args:
		frequency: [Hz]
		wall temperature: [K]
	
	Returns:
		spectral radiance: [W m-2 sr-1 Hz-1]

	"""

	L = ((2 * h * frequency ** 2) / (c ** 2)) * ( 1 / (np.exp(h * frequency / kb / Tw) - 1))

	return L

def temperature_distribution(x, v, theta, y):
	""" TEST FUNCTION: Generates mock temperature distribution along 1D profile to test integration.

	Args:
		x: position relative to orign (positive) [m]
	
	Returns:
		T: temperature [K]

	"""

	rn = 0.034

	if x < 1e-3:
		T = equilibrium_wall_temperature_stagnation(y, v, rn)
	else:
		T = equilibrium_wall_temperature_convective(y, v, x, theta)

	return T

def double_integral_wavelength(wav_min, wav_max, x_min, x_max, v, theta):
	""" Integrates over wavelength range and 1D region to find radiant intensity (radiant flux per solid angle element).

	Args:
		wav_min: wavelength lower limit on wavelength [m].
		wav_max: wavelength upper limit on wavelength [m].
	    x_min: lower limit on x [m].
		x_max: upper limit on x [m].

	Returns:
		radiant intensity [W sr-1]

	"""

	return dblquad(lambda x, wavelength: spectral_radiance_wavelength(wavelength, temperature_distribution(x, v, theta)), wav_min, wav_max, lambda x: x_min, lambda x: x_max,epsabs=1e-2, epsrel=1e-2)

def double_integral_frequency(freq_min, freq_max, x_min, x_max):
	""" Integrates over frequency range and 1D region to find radiant intensity (radiant flux per solid angle element).

	Args:
		freq_min: wavelength lower limit on frequency [m].
		wav_max: wavelength upper limit on frequency [m].
	    x_min: lower limit on x [m].
		x_max: upper limit on x [m].

	Returns:
		radiant intensity [W sr-1]

	"""

	return dblquad(lambda x, frequency: spectral_radiance_frequency(frequency, temperature_distribution(x)), freq_min, freq_max, lambda x: x_min, lambda x: x_max)


def single_integral_wavelength(x_min, x_max, v, theta, y):
	""" Integrates over wavelength range and 1D region to find radiant intensity (radiant flux per solid angle element).

	Args:
		wav_min: wavelength lower limit on wavelength [m].
		wav_max: wavelength upper limit on wavelength [m].
	    x_min: lower limit on x [m].
		x_max: upper limit on x [m].

	Returns:
		radiant intensity [W sr-1]

	"""

	return quad(lambda x: spectral_radiance_wavelength(1e-4, temperature_distribution(x, v, theta, y)), x_min, x_max)



