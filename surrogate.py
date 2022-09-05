import numpy as np
import itertools
from math import comb 
from past.builtins import xrange

# test

Data = np.array([[0.5, 2500, 48967.51312928783], [1, 2500, 95349.12253494107], [1.5, 2500, 140792.84623373678],
				[0.5, 3000, 59839.13494037382], [1, 3000, 116528.4055065199], [1.5, 3000, 172075.88922502898],
				[0.5, 3500, 70775.98315803635], [1, 3500, 137835.08209934094], [1.5, 3500, 203547.35378356333]])

Input = np.array([1.35, 3340])
 

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


beta = get_coeffs(Data, degree=2)

print(beta)

print(evaluate_surrogate(Input, beta, degree=2))

