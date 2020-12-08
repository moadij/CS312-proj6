#!/usr/bin/python3

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
from TSPBranchAndBound import *



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
    
    # Time Complexity: O(bn^2logN) where b is the number of attempted
    # solutions, n is the number of cities, and logN represents the average
    # number of cities visited during route determination (see 'm' in inner while loop)
    # Space complexity: O(n) where n is the number of cities
    def greedy( self,time_allowance=60.0 ):
        
        # Setting up current variables
        # Time Complexity: O(1)
        # Space Complexity: O(1)
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        
        start_time = time.time()
        
        # The meat of the algorithm. This outer while loop allows for
        # truncation of the algorithm at the time limit if no solution
        # has yet been found, and it also allows for multiple attempts
        # of the problem if the current greedy try results in an incomplete
        # path.
        # Time Complexity: O(bn^2logN) where b is the number of attempted
        # solutions, n is the number of cities, and logN represents the average
        # number of cities visited during route determination (see 'm' in inner while loop)
        # Space complexity: O(n) where n is the number of cities
        while not foundTour and time.time()-start_time < time_allowance and count < ncities:
            
            # Setting up current variables
            # Time Complexity: O(1)
            # Space Complexity: O(1) - each while loop overwrites them
            startingCity = cities[count]
            route = [startingCity]
            visited = [startingCity._index]
            
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
            bssf = TSPSolution(route)
            count += 1
                
            if bssf.cost < np.inf:
				# Found a valid route
                foundTour = True
                
        # Just setting up results variables, so constant space and time
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
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''
		
    # See the contents for details:
    # Time/Space Complexity: O(bmnqz) where b is the number of states that
    # were generated, m is the size of the biggest queue, n is num cities,
    # q is the number of queues (never bigger than n), and z is the number of
    # times the while loop executes because the queues are not empty
    def branchAndBound( self, time_allowance=60.0 ):
        results = {}
        
        # Greedy algorithm for initial bssf.
        # Time Complexity: O(bn^2logN) where b is the number of attempted
        # solutions, n is the number of cities, and logN represents the average
        # number of cities visited during route determination
        # Space complexity: O(n) where n is the number of cities
        #
        # If the Greedy algorithm couldn't find a path, the default
        # random tour is used
        # Time Complexity: O(bn) where b is the number of attempted solutions
        # and n is the number of cities
        # Space complexity: O(n) where n is the number of cities
        bssf = (self.greedy(time_allowance))['soln']
        if bssf is None:
            bssf = (self.defaultRandomTour())['soln']
        cities = self._scenario.getCities()
        
        # Just initialize the class. Because I generate the route state in the
        # __init__ function, this actually takes:
        # Time Complexity: O(n^2) where n is ncities
        # Space complexity: O(n^2) where n is ncities
        branchAndBound = BranchAndBoundSolver(time_allowance, bssf, cities, self._scenario)
        
        # Here is where I actually do the branch and bound searching.
        # See the function definition for details.
        # The following time/space is worst case scenario
        # Time/Space Complexity: O(bmnqz) where b is the number of states that
        # were generated, m is the size of the biggest queue, n is num cities,
        # q is the number of queues (never bigger than n), and z is the number of
        # times the while loop executes because the queues are not empty
        branchAndBound.branchAndBoundAlgorithm(time.time())
        
        # Just setting up results variables, so constant space and time
        end_time = time.time()
        results['cost'] = branchAndBound.bssf.cost
        results['time'] = end_time - branchAndBound.start_time
        results['count'] = branchAndBound.count
        results['soln'] = branchAndBound.bssf
        results['max'] = branchAndBound.maxQueueSize
        results['total'] = branchAndBound.statesCreated
        results['pruned'] = branchAndBound.prunedStates
        return results



    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
    def fancy( self,time_allowance=60.0 ):
        pass


    ''' <summary>
		This is a section of helper functions to assist with the algorithms above.
		</summary>
	'''
    
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
    
    