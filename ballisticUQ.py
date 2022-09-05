import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import atmosphere as atm
import constants as const
from vehicle_parameters import VehicleParameters
from dynamics import dive_pull_dynamics, glide_dynamics, ballistic_dynamics
import time

# define vehicle
#missile = VehicleParameters()
#missile.b = 3500
#missile.l_d = 0.1

# entry parameters
#h_o = 100e3 # m
#v_o_vec = [12e3] # m/s
#fpa_o_vec = [np.radians(25)] # radians
#alt_lim = 0 # m

# define altitude limit
#def altitude_limit(t, z, vehicle):

    #fpa, x, y, v = z

    #return y - 500

#altitude_limit.terminal = True 

# initialize counter
#counter = -1
#diveList = []

#for fpa_o in fpa_o_vec:
    #for v_o in v_o_vec:
        #counter += 1
        #print('Run: '+str(counter)+', V0: '+str(int(v_o/1e3))+'kps, EFPA: '+str(int(np.ceil(np.degrees(fpa_o))))+'deg')

        #dive_sol = solve_ivp(ballistic_dynamics, [0, 5e3], [fpa_o, 0, h_o, v_o], method='RK45', events=altitude_limit, args=[missile], max_step=1)
        #diveList.append(dive_sol)

        #plt.plot(dive_sol.y[1]/1e3, dive_sol.y[2]/1e3, label='V0: '+str(int(v_o/1e3))+'kps, EFPA: '+str(int(np.ceil(np.degrees(fpa_o))))+'deg')

#print("Done.")
#print("Generating plots...")
#plt.xlabel('Distance [km]')
#plt.ylabel('Altitude [km]',color='black')
#plt.title('Ballistic Entry Trajectory')
#plt.legend()
#plt.grid('minor')
#plt.show()

start = time.time()

bc = 2700
l_d = 0
g = 9.81
re = 6.3781e6
dt = 0.5
Rn = 1.0

class State:
    def __init__(self, x, y, fpa, u, q, rho, Ma, t):
        self.x = x
        self.y = y
        self.fpa = fpa
        self.u = u
        self.q = q
        self.rho = rho
        self.Ma = Ma
        self.t = t
        
def compute_uncertainty(state):
    
    if state.Ma > 6:
        dq = np.random.normal(0,1e6)
    else:
        dq = np.random.normal(0,1e5)
    
    if state.Ma > 5:
        db = np.random.normal(0,500)
    elif state.Ma > 3:
        db = np.random.normal(0,250)
    elif state.Ma > 1.2:
        db = np.random.normal(0,200)    
    elif state.Ma > 0.8:
        db = np.random.normal(0,500)
    else:
        db = np.random.normal(0,50)  
        
        
    db = 0
    dq = 0
    return dq, db
        
def update_state(state):
    
    rho = atm.get_density(state.y) 
    
    dq, db = compute_uncertainty(state)
    
    b = bc + db
    
    state.x += state.u * np.cos(state.fpa) * dt
    state.y -= state.u * np.sin(state.fpa) * dt
    state.fpa += (1/state.u * (-.5 * rho * (state.u ** 2) * l_d /b + g * np.cos(state.fpa) - state.u ** 2 / (re + state.y) * np.cos(state.fpa))) * dt
    state.u += (-.5 * rho * (state.u ** 2) / b + g * np.sin(state.fpa)) * dt
    state.q = 1.7415e-4 * (rho / Rn) ** 0.5 * (state.u ** 3) + dq
    state.rho = rho
    state.Ma = state.u / np.sqrt(1.4 * atm.get_R(state.y) * atm.get_temperature(state.y))
    state.t += dt
    return state        

def run_sim():        
    
    state = State(0, 100e3, np.radians(20), 12e3, 0, atm.get_density(100e3), 0, 0)    
    
    trajectory = []  
    while state.y > 100: 
        state = update_state(state)
        trajectory.append([state.x, state.y, state.fpa, state.u, state.q, state.rho, state.Ma, state.t])
        
    return trajectory
    
def run_mc():   
    
    trajectories = []
    for i in range(1):
        print("run ", i)
        trajectory = run_sim()
        trajectories.append(trajectory)
        
    return trajectories

trajectories = run_mc()

end = time. time()
print(end - start)

temp_max_q = []
temp_max_x = []
for i in range(len(trajectories)):
    temp_y = []
    temp_q = []
    temp_x = []
    for j in range(len(trajectories[i])):
        temp_y.append(trajectories[i][j][1])
        temp_q.append(trajectories[i][j][4])
        temp_x.append(trajectories[i][j][0])
    temp_max_q.append(max(temp_q))
    temp_max_x.append(temp_x[len(temp_x) - 1])
    #plt.plot(np.divide(temp_x,1e3), np.divide(temp_y,1e3))
    plt.plot(temp_y, temp_q)
    
plt.ylabel('Q [$W/m^{2}$]')
#plt.ylabel('Altitude [km]')
plt.xlabel('Altitude [m]')
#plt.xlim([min(temp_max_x)/1e3 - 1, max(temp_max_x)/1e3 + 1])
#plt.ylim([0, 5])
plt.grid()    
plt.show()

plt.hist(temp_max_q,50)
plt.xlabel('$Q_{peak}$ [$W/m^{2}$]')
#plt.xlabel('$x(t_{f})$ [m]')
plt.ylabel('Frequency')
plt.show()
    
