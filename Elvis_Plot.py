#Caleb Williams
#Memetic Algorithm applied to elvis needs boats problem

import math
import numpy as np
import random
import copy
#	Ploting library 
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
'''
#	Wireframe
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

min_range = -8
max_range = 8


step_size = 0.25

def checkDiversity(x,pop_size):

        same = 1

        for i in range(1,pop_size):

                   if(x[i] != x[0]):
                          same = 0
                          return same
        
        return same

#evaluation function
def fitness(x,N):
	sum = 0;
	sin_sum = 0;
	for i in range(0,N):
		sum += (x[i] + ((-1)**(i+1))*(i+1 % 4))**2
		sin_sum += x[i]**(i+1)

	elv = -math.sqrt(sum) + math.sin(sin_sum)
	return elv

def wiggle(x):

        dist_to_boundary = max_range - abs(x)

        # if there is a potential we go out of bounds, limit the wiggle room so we don't        
        if(dist_to_boundary < step_size):
                if(x < 0):
                        x+=random.uniform(-(dist_to_boundary),step_size)
                else:
                        x+=random.uniform(-step_size, dist_to_boundary) 
        
        else:
                x+=random.uniform(-step_size,step_size)

        # just in case 
        x = np.clip(x,min_range,max_range)

        return x

def SA(x, n, pop_size,ax):



	T = 1

	# separate cooling for GA or else we will get overflows cuz T gets too small too fast
	comp_coop_T = 1	

	# interval determines how many iterations of SA to do before we do Reproduction and Crossover
	interval = 10**3
	interval_count = 0

	# phase will either be comp or coop, 0 for comp and 1 for coop
	phase = 0


	# total number of iterations for SA
	num_iter = 10**5 * pop_size
	iter = 0	

	best = fitness(x[0],n)

	while(num_iter > iter):

		T = T * .9999
		
		# current individual
		curr = iter % pop_size
	
		newx = list(map(wiggle, x[curr]))

		newfit = fitness(newx, n)
		oldfit = fitness(x[curr], n)

		# want to check to see if solutions after a comp/coop phase are better than best
		if(oldfit > best):
			best = oldfit
	
		# this is delta E
		fitdif = oldfit - newfit

		if(fitdif <= 0):
			x[curr] = newx
		else:
			prob = math.exp(-(fitdif / T))

			if(random.uniform(0,1) <= prob):
				x[curr] = newx
	
		if(newfit > best):
			best = newfit

		# competitive phase
		if(interval_count >= interval and phase == 0):
			x = competitive(x, n, comp_coop_T, pop_size)
			interval_count = 0
			phase = 1

		# cooperative phase
		elif(interval_count >= interval and phase == 1):

			# if T is too small, we will get overflows
			if(comp_coop_T < 0.01):
				comp_coop_T = 0.01

			x = cooperative(x, n, comp_coop_T, pop_size)
			phase = 0
			interval_count = 0
			interval *= .9
			comp_coop_T *= .99

			interval = math.ceil(interval)

		same = checkDiversity(x, pop_size)

		# if there is no diversity then we exit
		if(same):
			break

		iter += 1

		# only one full pass for all individuals of SA counts as an interval count
		if(curr == pop_size - 1):
			interval_count += 1
			#	Scatter  plot
			scatterPlot(x, best, ax, pop_size)

	return best,x

def competitive(x,n,T,pop_size):

	fit = []

	newx = copy.deepcopy(x)
	
	# just store fitness values for all individuals
	for i in range(pop_size):
		fit.append(fitness(x[i], n))

	# always issue challenges with solution to the right
	for i in range(pop_size):

		# if we are last indiviudal in array, then we challenge the first array guy
		fitdif = fit[(i+1) % pop_size] - fit[i]

		prob = 1 / (1 + math.exp((fitdif / T)) )
		if(random.uniform(0,1) <= prob):
			newx[(i+1) % pop_size] = x[i]

	return newx
	

def cooperative(x, n, T,pop_size):

	# only cooperate with individuals on other half of array
	neighbor = int(pop_size/2)

	fit = []


	for i in range(pop_size):
		fit.append(fitness(x[i], n))

	# always propose to solutions at least neighbor away
	# go through each individual

	for i in range(pop_size):
		start = (i+neighbor) % pop_size
		end = (start + neighbor) % pop_size

		# check each neighbor (e.g. if array is size 8 then a[0] will be neighbors with a[4],a[5],[6],a[7])
		for j in range(start,end):

			# want both indiviuals to be A1 sauce

			prob1 = 1 / (1 + math.exp((-fit[i] / T )) )
			prob2 = 1 / (1 + math.exp((-fit[j] / T )) )

			prob = prob1 * prob2
	
			# perform the crossover
			if(random.uniform(0,1) <= prob):
				half = int(n/2)
				temp = x[j][:half]
				x[j] = x[i][:half] + x[j][half:]
				x[i] = temp + x[i][half:]
	
	return x

def elvis(X,Y):
	return -np.sqrt((X-1)**2+(Y+2)**2)+np.sin(X+Y**2) 

def scatterPlot(x, fit, ax, pop_size):
#	Scatter plot working 
#	All thats needed are other less concentraited plot
	xs = []
	ys = []
	zs = []
	print(x)
	for i in range(pop_size):
		xs.append(x[i][0])
		ys.append(x[i][1])
		zs.append(fitness(x[i], len(x[i])))

	#print("Xs", xs)
	#print("Ys",ys)
	#print("Zs",zs)
	
	plt.ion()
	sc = ax.scatter(xs, ys, zs, c = 'r')
	sc._offsets3d = (xs, ys, zs)
	plt.draw()
	plt.pause(.0005)
	sc.remove()


		
def main():
	elv = 0
	#	Draw Elvis function 
	X = np.arange(-8, 8, .05)
	Y = np.arange(-8, 8, .05)
	X, Y = np.meshgrid(X, Y)
	Z = elvis(X,Y)

	#	Plotting Elvis
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.contour3D(X, Y, Z, 50, cmap='binary')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z');
	#plt.show()


	pop_size = 8

	# dimensions
	#N = [1,2,3,5,8,13]
	N = [2]
	
	
	for n in N:
	
		x = []
		fit = []
		
		for i in range(pop_size):
			x.append([random.uniform(min_range,max_range) for t in range(n)])
		
		# Do the memes
		best,x = SA(x, n, pop_size,ax)
		
		#	Scatter plot points
        for i in range(pop_size):
                fit.append(fitness(x[i], n))
		
		print("Best for " + str(n) + " is " + str(best))		
	print("")
	# Ask for input to prevent closure		
	
if __name__ == "__main__":
	main()
