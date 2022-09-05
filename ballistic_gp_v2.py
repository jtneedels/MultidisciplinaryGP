import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import atmosphere as atm
import constants as const
from vehicle_parameters import VehicleParameters
from dynamics import dive_pull_dynamics, glide_dynamics, ballistic_dynamics
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import ConstantKernel
from mpl_toolkits import mplot3d

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
        
def plot_gpr_samples(gpr_model, rho_train, u_train, y_train, n_samples):
    nVals = 50
    x = np.zeros((nVals**2,2))
    rho = np.linspace(np.log(1e-6), np.log(1.2), nVals)
    v = np.linspace(0, 7, nVals)
    k = 0
    for i in range(nVals):
        for j in range(nVals):
            x[k,:] = np.array([rho[i], v[j]])
            k += 1

    y_mean, y_std = gpr_model.predict(x, return_std=True)
    y_samples = gpr_model.sample_y(x, n_samples, random_state=8)
    
    x, y = np.meshgrid(np.exp(rho), v)
    z = y_samples.reshape((nVals,nVals))
 
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    plt.xlabel('Density $[kg/m^{3}]$')
    plt.ylabel('Velocity $[km/s]$')
    ax.set_zlabel('Q $[W/cm^{2}]$')

    surf1 = ax.plot_surface(x, y, y_mean.reshape((nVals,nVals)), color  = 'black', label='mean')
    surf2 = ax.plot_surface(x, y, y_mean.reshape((nVals,nVals)) + 1.96 * y_std.reshape((nVals, nVals)), alpha = 0.5, color='grey', label='+/-1 std. dev.')
    ax.plot_surface(x, y, y_mean.reshape((nVals,nVals)) - 1.96 * y_std.reshape((nVals, nVals)), alpha = 0.5, color = 'grey')   
    
    surf1._edgecolors2d = surf1._edgecolor3d
    surf1._facecolors2d = surf1._facecolor3d  
    
    surf2._edgecolors2d = surf2._edgecolor3d
    surf2._facecolors2d = surf2._facecolor3d      
    
    ax.scatter(np.exp(rho_train), u_train, y_train, color='red', label='sampled points')
    ax.legend()
    plt.show()         
    
    return y_samples, rho, v

def training_function(x):
    
    return 1.7415e-8 * (np.exp(x[:,0]) / Rn) ** 0.5 * ((x[:,1]*1000) ** 3) # W/cm^2
        
def get_gp_q(state, samp, rho, v): 
    
    rhoidx = (np.abs(rho - np.log(state.rho))).argmin()
    vidx = (np.abs(v - state.u/1000)).argmin()
    q = float(samp[vidx + 50*rhoidx])
    return q
        
def update_state(state, samp, rho_new, v):
    
    rho = atm.get_density(state.y) 
    
    
    state.x += state.u * np.cos(state.fpa) * dt
    state.y -= state.u * np.sin(state.fpa) * dt
    state.fpa += (1/state.u * (-.5 * rho * (state.u ** 2) * l_d /b + g * np.cos(state.fpa) - state.u ** 2 / (re + state.y) * np.cos(state.fpa))) * dt
    state.u += (-.5 * rho * (state.u ** 2) / b + g * np.sin(state.fpa)) * dt
    state.rho = rho
    state.Ma = state.u / np.sqrt(1.4 * atm.get_R(state.y) * atm.get_temperature(state.y))
    state.q = get_gp_q(state, samp, rho_new, v)
    state.t += dt
    return state        

def run_sim(samp, rho, v):        
    
    state = State(0, 100e3, np.radians(20), 7e3, 0, atm.get_density(100e3), 0, 0)    
    
    trajectory = []  
    while state.y > 100: 
        state = update_state(state, samp, rho, v)
        trajectory.append([state.x, state.y, state.fpa, state.u, state.q, state.rho, state.Ma, state.t])
        
    return trajectory
    
def run_mc():   
    
    rng = np.random.RandomState(8)
    n_vals = 20
    rho_train = np.random.uniform(np.log(1e-6), np.log(1.2), n_vals**2)
    u_train = np.random.uniform(0, 7, n_vals**2)
    #rho_train, u_train = np.meshgrid(rho_train, u_train)
    X_train = np.zeros((n_vals**2,2))

    X_train[:,0] = rho_train.reshape((n_vals**2,))
    X_train[:,1] = u_train.reshape((n_vals**2,))
    y_train = training_function(X_train)
    n_samples = 1
    
    alphaVec = 1e-8*np.ones(n_vals**2)
    #alphaVec[699:899] = 1000
    
    #kernel = 1.0 * RBF(length_scale=[10,1], length_scale_bounds=(0.1, 100))
    kernel = ConstantKernel() * Matern(length_scale=[1.0,1.0], nu=1.5) +  1.0 * Matern(length_scale=[1.0,1.0], nu=1.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alphaVec, random_state=8)
    gpr.fit(X_train, y_train)  
    print(kernel.theta)
    
    # add gp for drag coefficient
    #X_cd_train = np.array([2.11, 2.79, 4.04, 5.83, 7.78, 9.54])
    #y_cd_train = np.array([0.985, 0.965, 0.910, 0.917, 0.910, 0.925])
    
    #kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10))
    #gpr_cd = GaussianProcessRegressor(kernel=kernel, random_state=8)
    #gpr_cd.fit(X_cd_train, y_cd_train)  
    
    y_samples, rho, v = plot_gpr_samples(gpr, rho_train, u_train, y_train, n_samples=n_samples)  
    
    trajectories = []
    count = 0
    for samp in y_samples.T:
        trajectory = run_sim(samp, rho, v)
        trajectories.append(trajectory)
        count += 1
        print(count)
        
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
    
plt.ylabel('Q [$W/cm^{2}$]')
#plt.ylabel('Altitude [km]')
plt.xlabel('Altitude [m]')
#plt.xlim([min(temp_max_x)/1e3 - 1, max(temp_max_x)/1e3 + 1])
#plt.ylim([0, 5])
plt.grid()    
plt.show()

plt.hist(temp_max_q,50)
plt.xlabel('$Q_{peak}$ [$W/cm^{2}$]')
#plt.xlabel('$x(t_{f})$ [m]')
plt.ylabel('Frequency')
plt.show()
