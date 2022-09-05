import atmosphere as atm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import constants as const
from dynamics import dive_pull_dynamics, glide_dynamics
from vehicle_parameters import VehicleParameters
from data_processing import process_solutions, extract_trajectory_point_data

## CONFIG OPTIONS##

# define vehicle
missile = VehicleParameters()

# entry parameters
h_o = 90e3 # m
v_o_vec = [5e3] # m/s
fpa_o_vec = [np.radians(5), np.radians(10), np.radians(20)] # radians
q_lim_vec = [20e3, 30e3, 40e3] # Pa
alt_lim = 100 # m

# define dive-pull limit
def dynamic_pressure_limit(t, z, vehicle):

	fpa, x, y, v = z

	return .5 * atm.get_density(y) * v ** 2 - q_lim

dynamic_pressure_limit.terminal = True 
dynamic_pressure_limit.direction = -1 

# define terminal dive limit
def altitude_limit(t, z, v0, y0, vehicle):

	x, y = z

	return y - 100

altitude_limit.terminal = True 

######################################################################
######################################################################
######################################################################

# run trajectory sweep 
print(" ")
print("BOOST-GLIDE TRAJECTORY SIMULATION")
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

# intialize lists for storing solutions
diveList = []
glideList = []
y0List = []
v0List = []

for fpa_o in fpa_o_vec:
	for q_lim in q_lim_vec:
		for v_o in v_o_vec:

			counter += 1
			print('Run: '+str(counter)+', Vo: '+str(int(v_o/1e3))+'kps, EFPA: '+str(int(np.degrees(fpa_o)))+'deg, q: '+str(int(q_lim/1e3))+'kPa')

			dive_sol = solve_ivp(dive_pull_dynamics, [0, 5e3], [fpa_o, 0, h_o, v_o], method='RK45', events=dynamic_pressure_limit, args=[missile], max_step=1)
			diveList.append(dive_sol)

			y0 = dive_sol.y_events[0][0][2]
			v0 = dive_sol.y_events[0][0][3]

			y0List.append(y0)
			v0List.append(v0)

			glide_sol = solve_ivp(glide_dynamics, [0, 5e3], [dive_sol.y_events[0][0][1], dive_sol.y_events[0][0][2]], method='RK45', events=altitude_limit, args=(v0, y0, missile), max_step=1)
			print(dive_sol.t[len(dive_sol.t)-1], glide_sol.t[len(glide_sol.t)-1])
			plt.plot(np.concatenate([dive_sol.y[1]/1e3, glide_sol.y[0]/1e3]), np.concatenate([dive_sol.y[2]/1e3, glide_sol.y[1]/1e3]), label='V0: '+str(int(v_o/1e3))+'kps, EFPA: '+str(int(np.degrees(fpa_o)))+'deg, q: '+str(int(q_lim/1e3))+'kPa')
			glideList.append(glide_sol)

print("Done.")
print("Generating plots...")
plt.xlabel('Distance [km]')
plt.ylabel('Altitude [km]',color='black')
plt.title('Trajectory')
plt.legend()
plt.grid('minor')
#plt.show()

# process data

runsToProcess = np.arange(len(y0List))

for i in runsToProcess:
	fpa, x, y, v = process_solutions(diveList[i], glideList[i], missile, y0List[i], v0List[i])
	points = extract_trajectory_point_data(diveList[i], glideList[i], missile, y0List[i], v0List[i], [100e3, 300e3, 500e3, 1000e3, 1500e3, 2000e3, 2500e3])
	
	#for point in points:
		#print('Range [m]: ', point.x, 'Alt [m]: ', point.y, 'Pressure [Pa]: ', point.P, 'Temperature [K]: ', point.T, 'Ma: ', point.Ma, 'v [m/s]: ', point.v, 'rho [kg/m^3]: ', point.rho)
		#print(point.x, point.y, point.P, point.T, point.Ma, point.v, point.rho)
		#plt.scatter(point.x/1e3, point.y/1e3, color='black')

#plt.ylabel('Tw [K]')
#plt.xlabel('Ma')
#plt.title('Leading Edge Equilibrium Wall Temperature, rn = 34 mm')
#plt.grid('minor')
plt.savefig('../../Downloads/singleTraj.png',dpi=300)
plt.show()
