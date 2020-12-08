#!/usr/bin/python3
import self

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import traceback
from copy import copy, deepcopy
import random


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		try:
			# initialization
			# this takes O(n) space to store all the cities
			# it takes O(1) time
			bssf = None
			cities = self._scenario.getCities()
			self.num_cities = len(cities)
			solution_dict = {}
			start_time = time.time()
			# remain below time limit
			while time.time() - start_time < time_allowance:
				# for each city as the starting node
				for index_loop in range(len(cities)):
					city = cities[index_loop]
					city_path = []
					city_path.append(city)
					to_visit = deepcopy(cities)
					current_city = city
					del to_visit[index_loop]
					while len(to_visit):
						city_costs = self.get_closest_cities(current_city, to_visit)
						closest_city_tuple = city_costs[0]
						closest_city_index = to_visit.index(closest_city_tuple[0])
						closest_city = to_visit[closest_city_index]
						if not self._scenario._edge_exists[current_city._index][closest_city._index]:
							# will only be the case if the cost for all cities left are infinite
							break
						del to_visit[closest_city_index]
						city_path.append(closest_city)
						current_city = closest_city
					if len(to_visit):
						continue
					else:
						bssf = TSPSolution(city_path)
						end_time = time.time()
						results = {}
						results['cost'] = bssf.cost
						results['time'] = end_time - start_time
						results['count'] = None
						results['soln'] = bssf
						results['max'] = None
						results['total'] = None
						results['pruned'] = None
						solution_dict[index_loop] = results
						continue

				self.lowest_cost = float("inf")
				for key, solution in solution_dict.items():
					if solution["cost"] < self.lowest_cost:
						self.lowest_cost = solution["cost"]
						lowest = solution

				return lowest

		except Exception as e:
			traceback.print_exc()
			raise (e)

	'''
			used in the greedy function
			will find the closest city
			Will store the list and find the min 
			will take O(n) time and space
		'''

	def get_closest_cities(self, city, city_list):
		cost = {}
		for city_to_visit in city_list:
			cost[city_to_visit] = city.costTo(city_to_visit)

		sorted_x = sorted(cost.items(), key=lambda kv: kv[1])
		return sorted_x
	'''
		crossover -
			make 3 functions where we can check partial paths, 
			decide where we cross over in the array, 
			and then another to check if that generated paths that work
	'''

	def crossover(self, cities_length, city1, city2, cities):
		city_cost1 = city1[0]
		city_cost2 = city2[0]

		city1 = city1[1:]
		city2 = city2[1:]
		cut_length = cities_length/2
		new_city1 = []
		new_city2 = []
		for i in cities_length:
			if i < cut_length:
				new_city1.append(city2[i])
				new_city2.append(city1[i])
			else:
				new_city1.append(city1[i])
				new_city2.append(city2[i])
		route1 = []
		route2 = []
		for i in cities_length:
			route1.append(cities[new_city1[i]])
			route2.append(cities[new_city2[i]])


		valid1 = TSPSolution(route1)
		valid2 = TSPSolution(route2)

		if valid1.cost() < math.inf and valid1.cost() < city_cost1:
			result_city1 = new_city1
			city1.insert(0, valid1.cost())
		else:
			# might need deep copy
			result_city1 = city1
			city1.insert(0, city_cost1)

		if valid2.cost() < math.inf and valid2.cost() < city_cost2:
			result_city2 = new_city2
			city1.insert(0, valid2.cost())
		else:
			# might need deep copy
			result_city2 = city2
			city2.insert(0, city_cost2)
		return [result_city1, result_city2]


	def fitness(self, city1, city2):
		if city1[0] < city2[0]:
			return city1
		else:
			return city2

	def select(self,initial_pop):
		random_num1 = random.randint(0, len(initial_pop)-1)
		random_num2 = random.randint(0,len(initial_pop)-1)
		random_num3 = random.randint(0,len(initial_pop)-1)
		random_num4 = random.randint(0,len(initial_pop)-1)

		while random_num1 == random_num2 :
			random_num2 = random.randint(0, len(initial_pop) - 1)

		while random_num3 == random_num2 or random_num3 == random_num1:
			random_num3 = random.randint(0, len(initial_pop) - 1)

		while random_num4 == random_num2 or random_num4 == random_num1 or random_num4 == random_num3:
			random_num4 = random.randint(0, len(initial_pop) - 1)


		#take the best cost of all paths
		random_paths=[random_num1, random_num2, random_num3, random_num4]
		random_paths.sort()

		return random_paths[-1], random_paths[-2]


	def mutate(self, cities_length, city_path, cities):
		i = 1
		constant_mutation = 0.05
		path_copy = copy.deepcopy(city_path)

		while i < cities_length-1:
			city_index1 = city_path[i]
			city_index2 = city_path[i+1]

			random_mut1 = random.uniform(0.0,1.0)
			if random_mut1 < constant_mutation:
				path_copy[i] = city_index2
				path_copy[i+1] = city_index1
			i += 1
		i =1
		route = []
		while i < cities_length:
			route.append(cities[i])
		valid1 = TSPSolution(route)
		if valid1.cost() < math.inf and valid1.cost() < city_path[0]:
			return path_copy
		else:
			return city_path






	def fancy( self,time_allowance=60.0 ):
		populationSet = self.initializePopulation()

	def initializePopulation(self):
		numPathsToGenerate = (len(self._scenario.getCities()))**2
		populationSet = [] # initialize an empty array
		populationSet = self.greedyPathGenerator(populationSet)
		return self.randomPathGenerator(populationSet, numPathsToGenerate)

	def greedyPathGenerator(self, populationSet):
		cities = self._scenario.getCities()
		ncities = len(cities)
		# This loop adds n to the time complexity of the interior
		# complexity, where n is ncities
		for city in cities:
			route = [city]
			visited = [city._index]

			# Time complexity: O(mn^2) where m is the number of total
			# cities and n is the number of visited cities
			# Space complexity: O(n)
			while len(route) < ncities:

				# Get the next minimum cost
				# Time complexity: O(mn) where m is the number of total
				# cities and n is the number of visited cities
				# Space complexity: O(1)
				minIndex = self.greedyFindMin(route[-1], cities, visited)
				if minIndex is None:
					break

				# As per the Python language description,
				# This has time and space complexity of O(1)
				route.append(cities[minIndex])
				visited.append(minIndex)

			if len(route) < ncities:
				continue

			# Use the TSPSolution functionality to add up
			# the costs. This is necessary to add the cost
			# of the last city back to the first.
			# Time Complexity: O(n) where n is the number of cities
			# Space Complexity: O(1)
			completePath = TSPSolution(route)

			if completePath.cost < np.inf:
					# Found a valid route
				populationElement = self.tspSolutionToArray(completePath)
				populationSet.append(populationElement)

		return populationSet

	def randomPathGenerator(self, populationSet, desiredNumPaths):
		cities = self._scenario.getCities()
		ncities = len(cities)

		while len(populationSet) < desiredNumPaths:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			completePath = TSPSolution(route)
			if completePath.cost < np.inf:\
				# Found a valid route
				populationElement = self.tspSolutionToArray(completePath)
				populationSet.append(populationElement)

		return populationSet

	# See interior for broken-down description:
	# Time complexity: O(mn) where m is the number of total
	# cities and n is the number of visited cities
	# Space complexity: O(1)
	def greedyFindMin(self, city, cities, visited):
		minCost = np.inf
		minIndex = None

		# Loop through the cities and find the min value
		# Time complexity: O(mn) where m is the number of total
		# cities and n is the number of visited cities
		# Space complexity: O(1) Because we're just updating the
		# variables declared above
		for i in range(len(cities)):
			if i in visited:
				continue
			else:
				# Time/Space: O(1) from the City class
				if city.costTo(cities[i]) < minCost:
					minCost = city.costTo(cities[i])
					minIndex = i

		return minIndex

	# Time/space complexity: O(n) where n is the number of cities
	def tspSolutionToArray(self, tspSolution):
		results = []

		# The first element of the array is the cost of the solution
		results.append(tspSolution.cost)

		# The remaining elements are the city ids in order
		for city in tspSolution.route:
			results.append(city._index)

		return results
		



