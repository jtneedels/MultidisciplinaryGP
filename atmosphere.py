import skaero.atmosphere.coesa as sk
import numpy as np
import constants as const

def get_density(alt):

	if alt >= 80e3 and alt < 200e3:
		return MSISE_density(alt)
	elif alt < 80e3 and alt >= 0:
		return sk.density(alt)
	else:
		print("Error: Input outside 0-200 km range.")
		quit()

def get_temperature(alt):

	if alt >= 80e3 and alt < 200e3:
		return MSISE_temperature(alt)
	elif alt < 80e3 and alt >= 0:
		return sk.temperature(alt)
	else:
		print("Error: Input outside 0-200 km range.")
		quit()

def get_pressure(alt):

	if alt >= 80e3 and alt < 200e3:
		return MSISE_pressure(alt)
	elif alt < 80e3 and alt >= 0:
		return sk.pressure(alt)
	else:
		print("Error: Input outside 0-200 km range.")
		quit()

def get_R(alt):

	if alt >= 80e3 and alt < 200e3:
		return MSISE_R(alt)
	elif alt < 80e3 and alt >= 0:
		return const.Ru/28.95
	else:
		print("Error: Input outside 0-200 km range.")
		quit()


# MSISE-90 upper atmosphere model
# data: ref. http://www.braeunig.us/space/atmos.htm
# Mean solar activity

MSISE_alt = [80e3, 100e3, 120e3, 140e3, 160e3, 180e3, 200e3]
MSISE_T = [196.3636, 184.0160, 374.9715, 635.5703, 787.5532, 877.6729, 931.2806]
MSISE_rho = [1.68E-05, 5.08E-07, 1.80E-08, 3.26E-09, 1.18E-09, 5.51E-10, 2.91E-10]
MSISE_P = [9.45E-01, 2.81E-02, 2.17E-03, 7.03E-04, 3.31E-04, 1.80E-04, 1.05E-04]
MSISE_Mw = [29.0175, 27.7137, 25.8745, 24.5349, 23.4225, 22.4106, 21.4734]

def MSISE_density(alt):
	return np.interp(alt, MSISE_alt, MSISE_rho)

def MSISE_temperature(alt):
	return np.interp(alt, MSISE_alt, MSISE_T)

def MSISE_pressure(alt):
	return np.interp(alt, MSISE_alt, MSISE_P)

def MSISE_R(alt):
	Mw = np.interp(alt, MSISE_alt, MSISE_Mw)
	return const.Ru/Mw

