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

"""Simulation Inputs""" 
# Parameters and constants
#b = 2700
l_d = 0
g = 9.81
re = 6.3781e6
dt = 0.5
Rn = 1.0

# Initial conditions
x0 = 0
y0 = 100e3
fpa0 = np.radians(20)
u0 = 7e3
q0 = 0
rho0 = atm.get_density(y0)
Ma0 = 0
t0 = 0

"""Trajectory Solver"""
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

def update_state(state, samp_q, samp_b, X_eval_q, X_eval_b):
    
    # Get atmospheric properties
    rho, R, T = atm.get_density(state.y), atm.get_R(state.y), atm.get_temperature(state.y)
    
    b = get_b(state, samp_b, X_eval_b)
    
    # Update dynamics - forward difference approximation
    state.x += state.u * np.cos(state.fpa) * dt
    state.y -= state.u * np.sin(state.fpa) * dt
    state.fpa += (1/state.u * (-.5 * rho * (state.u ** 2) * l_d /b + g * np.cos(state.fpa) - state.u ** 2 / (re + state.y) * np.cos(state.fpa))) * dt
    state.u += (-.5 * rho * (state.u ** 2) / b + g * np.sin(state.fpa)) * dt
    state.rho = rho
    state.Ma = state.u / np.sqrt(1.4 * R * T)
    state.q = get_q(state, samp_q, X_eval_q)
    #state.q = 1.7415e-8 * (rho / Rn) ** 0.5 * ((state.u) ** 3)
    state.t += dt
    
    return state        

def run_sim(samp_q, samp_b, X_eval_q, X_eval_b):        
    
    # Initialize state with sim initial conditions
    state = State(x0, y0, fpa0, u0, q0, rho0, Ma0, t0)    
    
    trajectory = []  
    while state.y > 100: 
        state = update_state(state, samp_q, samp_b, X_eval_q, X_eval_b)
        trajectory.append([state.x, state.y, state.fpa, state.u, state.q, state.rho, state.Ma, state.t])
        
    return trajectory

def sample_gp_q(gpr_model, X_train, y_train, n_samples):
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
    
    x = np.zeros((nVals,2))
    x[:,0] = rho
    x[:,1] = v
    
    return y_samples, x 

def sample_gp_b(gpr_model, X_train, y_train, n_samples):
    nVals = 50
    x = np.linspace(0, 7, nVals)

    y_mean, y_std = gpr_model.predict(x.reshape(-1,1), return_std=True)
    y_samples = gpr_model.sample_y(x.reshape(-1,1), n_samples, random_state=8)    
    
    return y_samples, x

def plot_gpr_samples_b(gpr_model, n_samples):
    x = np.linspace(0, 7, 100)
    X = x.reshape(-1, 1)
    
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(10, 8))

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.plot(x, 2450 + 2000/(x + 1), color='blue', label="True Function")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    
    n_vals = 6
    X_train_b = np.linspace(0, 7, n_vals)
    y_train_b = training_function_b(X_train_b)    
    
    ax.scatter(X_train_b, y_train_b, color="red", zorder=10, label="Observations")
    ax.legend()
    ax.set_title("Ballistic Coefficient Posterior")    
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([min(y_train_b) - 100, max(y_train_b) + 100])
    plt.show()
    
def plot_gp(gpr_model, X_train, y_train):
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

def training_function_q(x):
    
    # Sutton-Graves heating correlation
    return 1.7415e-8 * (np.exp(x[:,0]) / Rn) ** 0.5 * ((x[:,1]*1000) ** 3) # W/cm^2

def training_function_b(x):
    
    # asymptote to fixed value of ballistic coefficient
    return 2450 + 2000/(x + 1)
        
def get_q(state, samp_q, X_eval_q): 
    
    rho = X_eval_q[:,0]
    v = X_eval_q[:,1]
    
    rhoidx = (np.abs(rho - np.log(state.rho))).argmin()
    vidx = (np.abs(v - state.u/1000)).argmin()
    q = float(samp_q[vidx + 50*rhoidx])
    
    return q
    
def get_b(state, samp_b, X_eval_b):
    
    v = X_eval_b
    
    vidx = (np.abs(v - state.u/1000)).argmin()
    b = float(samp_b[vidx])    
    
    return b

def run_mc():   
    
    rng = np.random.RandomState(8)
    
    n_samples = 1
    
    # gp for q
    n_vals = 30
    rho_train = np.random.uniform(np.log(1e-6), np.log(1.2), n_vals**2)
    u_train = np.random.uniform(0, 7, n_vals**2)
    #rho_train, u_train = np.meshgrid(rho_train, u_train)
    X_train_q = np.zeros((n_vals**2,2))
    X_train_q[:,0] = rho_train.reshape((n_vals**2,))
    X_train_q[:,1] = u_train.reshape((n_vals**2,))
    y_train_q = training_function_q(X_train_q)
    alphaVec = 1e-8*np.ones(n_vals**2)
    
    alphaVec[799:899] = 1000
    
    #kernel = 1.0 * RBF(length_scale=[1.0,1.0], length_scale_bounds=(1, 10))
    kernel = ConstantKernel() * Matern(length_scale=[1.0,1.0], nu=1.5) +  1.0 * Matern(length_scale=[1.0,1.0], nu=1.5)
    gpr_q = GaussianProcessRegressor(kernel=kernel, alpha=alphaVec, random_state=8)
    gpr_q.fit(X_train_q, y_train_q)  
    
    samp_q, X_eval_q = sample_gp_q(gpr_q, X_train_q, y_train_q, n_samples=n_samples) 
    
    # gp for b
    n_vals = 6
    X_train_b = np.linspace(0, 7, n_vals)
    y_train_b = training_function_b(X_train_b)
    alphaVec = 1e-8*np.ones(n_vals)
    alphaVec[3:5] = y_train_b[3:5]
    
    kernel = ConstantKernel() * Matern(length_scale=1.0, nu=1.5)
    gpr_b = GaussianProcessRegressor(kernel=kernel, alpha=alphaVec, random_state=8)
    gpr_b.fit(X_train_b.reshape(-1,1), y_train_b)      
    
    samp_b, X_eval_b = sample_gp_b(gpr_b, X_train_b, y_train_b, n_samples=n_samples) 
    
    plot_gpr_samples_b(gpr_b, n_samples)
    
    trajectories = []
    count = 0
    for i in range(n_samples):
        trajectory = run_sim(samp_q[:,i], samp_b[:,i], X_eval_q, X_eval_b)
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
    temp_Ma = []
    for j in range(len(trajectories[i])):
        temp_y.append(trajectories[i][j][1])
        temp_q.append(trajectories[i][j][4])
        temp_x.append(trajectories[i][j][0])
        temp_Ma.append(trajectories[i][j][6])
    temp_max_q.append(max(temp_q))
    temp_max_x.append(temp_x[len(temp_x) - 1]/1e3)
    #plt.plot(np.divide(temp_x,1e3), np.divide(temp_y,1e3))
    plt.plot(temp_y, temp_q)
    
plt.ylabel('Q [$W/cm^{2}$]')
#plt.ylabel('Altitude [km]')
plt.xlabel('Altitude [m]')
#plt.xlim([min(temp_max_x)/1e3 - 1, max(temp_max_x)/1e3 + 1])
#plt.ylim([0, 5])
plt.grid()    
plt.show()

plt.hist(temp_max_x, 50)
#plt.xlabel('$Q_{peak}$ [$W/cm^{2}$]')
plt.xlabel('$x(t_{f})$ [km]')
plt.ylabel('Frequency')
plt.show()
