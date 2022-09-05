import thermal as th
import matplotlib.pyplot as plt


#Tw = []

#Tw.append(th.equilibrium_wall_temperature_stagnation(40335, 3303, 0.034))

#pos = [0.1, 0.25, 0.5, 1, 1.5, 3, 5]

#for x in pos:
#	Tw.append(th.equilibrium_wall_temperature_convective(40335, 3303, x, 5))

#vec = [0, 0.1, 0.25, 0.5, 1, 1.5, 3, 5]
#print(Tw)

#plt.scatter(vec, Tw)

#plt.ylabel('Tw [K]')
#plt.xlabel('Axial Position [m]')
#plt.title('Radiative Equilibrium Tw, rn = 34 mm, 5 deg inclination')
#plt.grid('minor')
#plt.show()

import time

output = []

x = 1.5

print(th.single_integral_wavelength(0, x, 3.17901703e+03, 3.71032705e+0, 3.82098298e+04))

for v in [2500, 3000, 3500]:
	for y in [35e3, 40e3, 45e3]:
		for theta in [1, 3, 5]:
			start = time.time()
			#print(x, v, theta, th.double_integral_wavelength(7.8e-7, 1e-3, 0, x, v, theta))
			output.append([y, v, theta, th.single_integral_wavelength(0, x, v, theta, y)[0]])
			end = time.time()
			print(end - start)

print(output)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
