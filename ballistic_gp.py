import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import atmosphere as atm
import constants as const
from vehicle_parameters import VehicleParameters
from dynamics import dive_pull_dynamics, glide_dynamics, ballistic_dynamics
import time

b = 2700
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
        
def get_gp_q(state):
    
    rho = np.linspace(0, 1.2, 15)
    v = np.linspace(0, 15, 15)   
    
    mat = np.array([-6.77452070e-02,  4.05073477e-02,  4.00635988e-02, -1.27078346e-02,
 -5.38936671e-02, -5.84382810e-02, -2.68580751e-02,  2.48744303e-02,
  5.71371523e-02,  5.35203934e-02,  2.39456998e-02, -2.75720872e-02,
 -8.79923814e-02, -7.75413330e-02,  9.51480911e-02, -1.05955883e+00,
  4.86075705e+00,  4.96472017e+01,  1.71365274e+02,  4.08288841e+02,
  7.98519286e+02,  1.38059865e+03,  2.19286011e+03,  3.27379593e+03,
  4.66184162e+03,  6.39546319e+03,  8.51302046e+03,  1.10527685e+04,
  1.40529296e+04,  1.75515355e+04,  6.77706455e-01,  8.92580380e+00,
  7.07841311e+01,  2.38989840e+02,  5.66211539e+02,  1.10538645e+03,
  1.90948931e+03,  3.03184694e+03,  4.52528846e+03,  6.44314542e+03,
  8.83838860e+03,  1.17640474e+04,  1.52731164e+04,  1.94183123e+04,
  2.42521784e+04,  4.42551111e-01,  1.05703214e+01,  8.65832505e+01,
  2.93181053e+02,  6.95203361e+02,  1.35769930e+03,  2.34563566e+03,
  3.72436336e+03,  5.55912409e+03,  7.91514159e+03,  1.08576780e+04,
  1.44518361e+04,  1.87626792e+04,  2.38549740e+04,  2.97935321e+04,
 -2.01316300e-02,  1.17179545e+01,  9.97127495e+01,  3.38547970e+02,
  8.03254268e+02,  1.56894315e+03,  2.71098424e+03,  4.30480273e+03,
  6.42584103e+03,  9.14938986e+03,  1.25511309e+04,  1.67060585e+04,
  2.16895147e+04,  2.75762265e+04,  3.44410886e+04,  1.22514964e+00,
  1.40801144e+01,  1.12225076e+02,  3.78917900e+02,  8.97953405e+02,
  1.75322390e+03,  3.02883786e+03,  4.80888764e+03,  7.17767503e+03,
  1.02197135e+04,  1.40188398e+04,  1.86594327e+04,  2.42252483e+04,
  3.08000843e+04,  3.84673537e+04,  9.53968281e-01,  1.52493451e+01,
  1.22888877e+02,  4.15227629e+02,  9.83832920e+02,  1.92081015e+03,
  3.31839947e+03,  5.26870460e+03,  7.86411314e+03,  1.11968384e+04,
  1.53593537e+04,  2.04437030e+04,  2.65418157e+04,  3.37452664e+04,
  4.21457185e+04, -5.58078980e-01,  1.49274575e+01,  1.30963460e+02,
  4.46998095e+02,  1.06137047e+03,  2.07351477e+03,  3.58338586e+03,
  5.69055831e+03,  8.49435403e+03,  1.20951552e+04,  1.65920282e+04,
  2.20849356e+04,  2.86729937e+04,  3.64553170e+04,  4.55308491e+04,
  1.09916781e+00,  1.76759358e+01,  1.41651999e+02,  4.79215126e+02,
  1.13592160e+03,  2.21798920e+03,  3.83185292e+03,  6.08430289e+03,
  9.08147080e+03,  1.29302085e+04,  1.77370060e+04,  2.36083990e+04,
  3.06504301e+04,  3.89692740e+04,  4.86701979e+04,  1.19966756e+00,
  1.88148500e+01,  1.50549439e+02,  5.08386171e+02,  1.20509650e+03,
  2.35269361e+03,  4.06470728e+03,  6.45356891e+03,  9.63262846e+03,
  1.37152372e+04,  1.88138560e+04,  2.50418781e+04,  3.25115371e+04,
  4.13355141e+04,  5.16255989e+04,  3.11374553e-01,  1.87690023e+01,
  1.57539063e+02,  5.34800048e+02,  1.26890075e+03,  2.47851100e+03,
  4.28268186e+03,  6.80031883e+03,  1.01507976e+04,  1.44531503e+04,
  1.98265461e+04,  2.63899352e+04,  3.42620940e+04, 4.35611714e+04,
  5.44054332e+04,  7.80387628e-01,  2.01574494e+01,  1.65690324e+02,
  5.61458281e+02,  1.33159705e+03,  2.60064393e+03 , 4.49326613e+03,
  7.13463291e+03,  1.06493336e+04,  1.51628909e+04 , 2.07998137e+04,
  2.76852243e+04,  3.59432253e+04,  4.56985908e+04,  5.70749863e+04,
  5.84706688e-01,  2.09533958e+01,  1.73218815e+02,  5.86663703e+02,
  1.39144298e+03,  2.71726999e+03,  4.69487730e+03,  7.45470413e+03,
  1.11274627e+04,  1.58438010e+04,  2.17339893e+04,  2.89288514e+04,
  3.75582711e+04,  4.77521358e+04,  5.96399306e+04,  2.34698342e+00,
  2.32360160e+01,  1.81313126e+02,  6.10723976e+02,  1.44677046e+03,
  2.82382208e+03,  4.87805059e+03,  7.74420464e+03,  1.15586343e+04,
  1.64569028e+04,  2.25743855e+04,  3.00465193e+04,  3.90086417e+04,
  4.95955126e+04,  6.19413936e+04,  1.15677085e+00,  2.25961851e+01,
  1.87021471e+02,  6.33584039e+02,  1.50236271e+03,  2.93395852e+03,
  5.06879100e+03,  8.04806456e+03,  1.20129767e+04,  1.71044278e+04,
  2.34634260e+04,  3.12306759e+04,  4.05466856e+04,  5.15515317e+04,
  6.43842011e+04])
    #mat = mat.reshape((10,10))
    
    rhoidx = (np.abs(rho - state.rho)).argmin()
    vidx = (np.abs(v - state.u/1000)).argmin()
    q = float(mat[vidx + 15*rhoidx])
    print(state.y, state.u, state.rho, q)
    return q
        
def update_state(state):
    
    rho = atm.get_density(state.y) 
    
    
    state.x += state.u * np.cos(state.fpa) * dt
    state.y -= state.u * np.sin(state.fpa) * dt
    state.fpa += (1/state.u * (-.5 * rho * (state.u ** 2) * l_d /b + g * np.cos(state.fpa) - state.u ** 2 / (re + state.y) * np.cos(state.fpa))) * dt
    state.u += (-.5 * rho * (state.u ** 2) / b + g * np.sin(state.fpa)) * dt
    state.rho = rho
    state.Ma = state.u / np.sqrt(1.4 * atm.get_R(state.y) * atm.get_temperature(state.y))
    state.q = get_gp_q(state)
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

#plt.hist(temp_max_q,50)
#plt.xlabel('$Q_{peak}$ [$W/m^{2}$]')
#plt.xlabel('$x(t_{f})$ [m]')
#plt.ylabel('Frequency')
#plt.show()
