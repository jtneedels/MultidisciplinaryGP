import atmosphere as atm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import constants as const
from dynamics import dive_pull_dynamics, glide_dynamics, ballistic_dynamics
from vehicle_parameters import VehicleParameters
from data_processing import process_solutions, extract_trajectory_point_data, process_solutions_ballistic, extract_trajectory_point_data_ballistic

## CONFIG OPTIONS##

# define vehicle
missile = vehicle_parameters()
missile.b = 3500
missile.l_d = 1

# entry parameters
h_o = 100e3 # m
v_o_vec = [12e3] # m/s
fpa_o_vec = [np.radians(15), np.radians(20), np.radians(25), np.radians(30)] # radians
q_lim_vec = [] # Pa
alt_lim = 100 # m

# define altitude limit
def altitude_limit(t, z, vehicle):

	fpa, x, y, v = z

	return y - 500

altitude_limit.terminal = True 

######################################################################
######################################################################
######################################################################

# run trajectory sweep 
print(" ")
print("BALLISTIC-ENTRY TRAJECTORY SIMULATION")
print(" ")
print("Vehicle Parameters: ")
print("		Ballistic Coefficient: ", missile.b)
print("		Max L/D: ", missile.l_d)
print(" ")
print("Trajectory Parameters: ")
print("		Entry Interface Altitude [km]: ", np.divide(h_o, 1e3))
print("		Entry Velocity [km/s]: ", np.divide(v_o_vec, 1e3))
print("		Entry Flight Path Angle [deg]: ", np.degrees(fpa_o_vec))
print("		Glide Dynamic Pressure [kPa]: ", np.divide(q_lim_vec, 1e3))
print(" ")
print("Running trajectories...")

# initialize counter
counter = -1
diveList = []

for fpa_o in fpa_o_vec:
	for v_o in v_o_vec:
		counter += 1
		print('Run: '+str(counter)+', V0: '+str(int(v_o/1e3))+'kps, EFPA: '+str(int(np.ceil(np.degrees(fpa_o))))+'deg')

		dive_sol = solve_ivp(ballistic_dynamics, [0, 5e3], [fpa_o, 0, h_o, v_o], method='RK45', events=altitude_limit, args=[missile], max_step=1)
		diveList.append(dive_sol)

		plt.plot(dive_sol.y[1]/1e3, dive_sol.y[2]/1e3, label='V0: '+str(int(v_o/1e3))+'kps, EFPA: '+str(int(np.ceil(np.degrees(fpa_o))))+'deg')

print("Done.")
print("Generating plots...")
plt.xlabel('Distance [km]')
plt.ylabel('Altitude [km]',color='black')
plt.title('Ballistic Entry Trajectory')
plt.legend()
plt.grid('minor')
plt.show()

runsToProcess = [3]

for i in runsToProcess:
	fpa, x, y, v = process_solutions_ballistic(diveList[i], missile)
	points = extract_trajectory_point_data_ballistic(diveList[i], missile, [10e3, 20e3, 40e3, 50e3, 75e3, 100e3, 125e3, 150e3, 175e3, 200e3, 225e3, 250e3, 300e3, 325e3, 350e3])
	
	for point in points:
		plt.scatter(point.Ma, point.y/1e3,color='red')

plt.ylabel('Alt [km]')
plt.xlabel('Ma ')
plt.title('Alt. vs Mach 30 deg EFPA')
plt.grid('minor')
plt.show()