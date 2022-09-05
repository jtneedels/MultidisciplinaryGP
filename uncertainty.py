import numpy as numpy



def compute_joint_distribution():

	redEdges = 3
	greedEdges = 3
	nbins = 3

	red = np.array([1, 2, 3, 4, 5, 6])
	green = np.array([10, 30, 50, 40, 60, 20])

	(H,redEdges,greedEdges) = np.histogram2d(
	    red.ravel(),green.ravel(),
	    bins=nbins
	)

	#divide by the total to get the probability of 
	#each cell -> the joint distribution
	Prg = H/H.sum()

	print(Prg)


