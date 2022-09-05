from scipy.optimize import minimize, least_squares
import numpy as np
import itertools
from math import comb
import time
import matplotlib.pyplot as plt

def get_polynomial_indices(numvars, power): 

	vars = np.arange(numvars)
	 
	vars = np.insert(vars, 0, -1) # add dummy variable 
	 
	# compute all combinations of variables 
	terms = [] 
	for x in itertools.combinations_with_replacement(vars, power):
		terms.append(x) 
 
	return terms 

def get_coeffs(Data, degree=2):

	numsamples = Data.shape[0]
	numvars = Data.shape[1] - 1
	numterms = comb(2 + numvars, 2)

	X = np.zeros((numsamples, numterms))
	Y = np.zeros((numsamples, 1))

	terms = get_polynomial_indices(numvars, degree)

	for i in range(numsamples):
		Y[i][0] = Data[i][numvars]
		j = 0
		for term in terms:
			X[i][j] = 1
			for val in term:
				if val >= 0:
					X[i][j] *= Data[i][val]
			j += 1

	beta = np.dot(np.linalg.pinv(X) , Y)

	return beta

def evaluate_surrogate(Input, beta, degree=2):

	numvars = Input.size
	numterms = comb(2 + numvars, 2)

	vec = np.zeros((1, numterms))

	terms = get_polynomial_indices(numvars, degree)

	j = 0
	for term in terms:
		vec[0][j] = 1
		for val in term:
			if val >= 0:
				vec[0][j] *= Input[val]
		j += 1

	return np.dot(vec, beta)

def lsq_objective_function(Input, target, beta):

	return (target - float(evaluate_surrogate(Input, beta)))**2



## Test
Data = np.array([[0.5, 2500, 48967.51312928783], [1, 2500, 95349.12253494107], [1.5, 2500, 140792.84623373678],
				[0.5, 3000, 59839.13494037382], [1, 3000, 116528.4055065199], [1.5, 3000, 172075.88922502898],
				[0.5, 3500, 70775.98315803635], [1, 3500, 137835.08209934094], [1.5, 3500, 203547.35378356333]])
x0 = np.array([1, 3000])
target = 175e3
beta = get_coeffs(Data)

Data = np.array([[35000.0, 2500, 1, 95488.68631206776], [35000.0, 2500, 3, 135280.0812899231], [35000.0, 2500, 5, 158313.4319921878], [40000.0, 2500, 1, 84407.72540049365], [40000.0, 2500, 3, 120089.98055796238], [40000.0, 2500, 5, 140792.84623373678], [45000.0, 2500, 1, 75000.46311491764], [45000.0, 2500, 3, 107159.21907127144], [45000.0, 2500, 5, 125854.37806599155], [35000.0, 3000, 1, 117017.0530517992], [35000.0, 3000, 3, 165364.44364178463], [35000.0, 3000, 5, 193430.22118121624], [40000.0, 3000, 1, 103580.2157510591], [40000.0, 3000, 3, 146888.71709849892], [40000.0, 3000, 5, 172075.88922502898], [45000.0, 3000, 1, 92180.56509835615], [45000.0, 3000, 3, 131180.2688338305], [45000.0, 3000, 5, 153896.93514596543], [35000.0, 3500, 1, 138680.24288688076], [35000.0, 3500, 3, 195630.51502269512], [35000.0, 3500, 5, 228756.85461093634], [40000.0, 3500, 1, 122875.21433689045], [40000.0, 3500, 3, 173850.9272371449], [40000.0, 3500, 5, 203547.35378356333], [45000.0, 3500, 1, 109473.1192445359], [45000.0, 3500, 3, 155349.57645835794], [45000.0, 3500, 5, 182109.86707501902]])
x0 = np.array([42e3, 3.2e3, 2])
target = 175e3
beta = get_coeffs(Data)

start = time.time()
res_1 = minimize(lsq_objective_function, x0, args=(target, beta), bounds = ((35e3, 45e3), (2500, 3500), (1,5)), method='L-BFGS-B')
print(res_1.x)
print(res_1.fun)
print(res_1.nfev, res_1.njev)
end = time.time()
print(end - start)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# for v in np.linspace(2500, 3500, 10):
# 	for y in np.linspace(35000, 45000, 10):
# 		for theta in np.linspace(1,5, 10):
# 			val = evaluate_surrogate(np.array([v, y, theta]), beta,)
# 			ax.scatter(v, theta, val)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

from scipy.stats import multivariate_normal
rv = multivariate_normal([3000, 40e3], [[1, 0], [0, 1]])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for v in np.linspace(2500, 3500, 5):
	for y in np.linspace(35000, 45000, 5):
		for theta in np.linspace(1,5, 5):
			x0 = np.array([y, v, theta])
			res_1 = minimize(lsq_objective_function, x0, args=(target, beta), bounds = ((35e3, 45e3), (2500, 3500), (1,5)), method='L-BFGS-B')
			ax.scatter(res_1.x[0]/1e3, res_1.x[1]/1e3, res_1.x[2], color='blue')
			print(res_1.x[0], res_1.x[1], rv.cdf([res_1.x[1]*1.01, res_1.x[0]*1.01]) - rv.cdf([res_1.x[1]*.99, res_1.x[0]*.99]))

ax.set_xlabel('Altitude [km]')
ax.set_ylabel('Velocity [km/s]')
ax.set_zlabel('Inclination [deg]')
ax.set_title('Optimal Points')

plt.show()




