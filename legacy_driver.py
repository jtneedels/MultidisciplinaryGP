import atmosphere as atm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import constants as const
from dynamics import dive_pull_dynamics, glide_dynamics
from vehicle_parameters import VehicleParameters
from data_processing import process_solutions, extract_trajectory_point_data

# define vehicle
missile = VehicleParameters()

# entry parameters
v_o = 5e3 # m/s
h_o = 80e3 # m
fpa_o_vec = [np.radians(5)]#, np.radians(10), np.radians(20)] # radians
q_lim_vec = [10e3] #[10e3, 20e3, 40e3] # Pa
alt_lim = 100 # m

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for fpa_o in fpa_o_vec:
	for q_lim in q_lim_vec:

		print(fpa_o, q_lim)

		def dynamic_pressure_limit(t, z, vehicle):

			fpa, x, y, v = z

			return .5 * atm.get_density(y) * v ** 2 - q_lim

		def altitude_limit(t, z, v0, y0, vehicle):

			x, y = z

			return y - 100

		dynamic_pressure_limit.terminal = True 
		dynamic_pressure_limit.direction = -1 

		altitude_limit.terminal = True 

		dive_sol = solve_ivp(dive_pull_dynamics, [0, 5e3], [fpa_o, 0, h_o, v_o], method='RK45', events=dynamic_pressure_limit, args=[missile], max_step=1)

		y0 = dive_sol.y_events[0][0][2]
		v0 = dive_sol.y_events[0][0][3]

		glide_sol = solve_ivp(glide_dynamics, [0, 5e3], [dive_sol.y_events[0][0][1], dive_sol.y_events[0][0][2]], method='RK45', events=altitude_limit, args=(v0, y0, missile), max_step=1)

		ax1.plot(np.concatenate([dive_sol.y[1]/1e3, glide_sol.y[0]/1e3]), np.concatenate([dive_sol.y[2]/1e3, glide_sol.y[1]/1e3]), label='V0: '+str(int(v_o/1e3))+'kps, EFPA: '+str(int(np.degrees(fpa_o)))+'deg, q: '+str(int(q_lim/1e3))+'kPa')

		fpa, x, y, v = process_solutions(dive_sol, glide_sol, missile, y0, v0)
		points = extract_trajectory_point_data(dive_sol, glide_sol, missile, y0, v0, [100e3, 200e3, 300e3, 400e3, 500e3, 1000e3, 1500e3, 2000e3, 2500e3])
		#points = extract_trajectory_point_data(dive_sol, glide_sol, missile, y0, v0, x)

		#ma = []
		#for point in points:
		#	print('Alt: ', point.y, 'Press: ', point.P, 'Temp: ', point.T, 'Ma: ', point.Ma, 'v: ', point.v)
		#	ax1.scatter(point.x/1e3, point.y/1e3)
	        #	ma.append(point.Ma)

		#ax2.plot(np.concatenate([dive_sol.y[1]/1e3, glide_sol.y[0]/1e3]), ma, linestyle='dashed')

#fpa, x, y, v = process_solutions(dive_sol, glide_sol, missile, y0, v0)
#points = extract_trajectory_point_data(dive_sol, glide_sol, missile, y0, v0, [100e3, 200e3, 300e3, 400e3, 500e3, 1000e3, 1500e3, 2000e3, 2500e3])
#points = extract_trajectory_point_data(dive_sol, glide_sol, missile, y0, v0, x)

#ma = []
#for point in points:
#	print('Alt: ', point.y, 'Press: ', point.P, 'Temp: ', point.T, 'Ma: ', point.Ma, 'v: ', point.v)
#	ax1.scatter(point.x/1e3, point.y/1e3)
#	ma.append(point.Ma)

#ax2.plot(np.concatenate([dive_sol.y[1]/1e3, glide_sol.y[0]/1e3]), ma, color='red')

points = extract_trajectory_point_data(dive_sol, glide_sol, missile, y0, v0, [150e3, 400e3, 1000e3, 2000e3, 2700e3])

for point in points:
	print('Alt: ', point.y, 'Press: ', point.P, 'Temp: ', point.T, 'Ma: ', point.Ma, 'v: ', point.v)
	ax1.scatter(point.x/1e3, point.y/1e3, color='black')

ax1.set_xlabel('Distance [km]')
ax1.set_ylabel('Altitude [km]',color='black')
ax2.set_ylabel('Ma')
plt.title('Trajectory')
ax1.legend()
ax1.grid('minor')

plt.show()


#for point in points:
	#plt.scatter(point.Ma, point.Tw)

#plt.ylabel('Tw [K]')
#plt.xlabel('Ma ')
#plt.title('Tw History, 5 deg inclination, 1 m axial position')
#plt.grid('minor')
#plt.show()