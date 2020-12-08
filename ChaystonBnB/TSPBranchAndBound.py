# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:03:45 2020

@author: Chayston Wood
"""

import time
import numpy as np
from TSPClasses import *
from queue import PriorityQueue
import copy

class BranchAndBoundSolver:
    
    # Totals:
    # Time Complexity: O(n^2) where n is ncities
    # Space complexity: O(n^2) where n is ncities
    def __init__(self, time_allowance, bssf, cities, scenario):
        
        # Constant time and space just setting up class variables.
        self.time_allowance = time_allowance    
        self.bssf = bssf
        self.cities = cities
        self.ncities = len(cities)
        self.scenario = scenario
        
        self.queues = []
        startingCity = self.cities[0]
        self.statesCreated = 0
        
        # Set up the root state.
        # See the function definition to get more details.
        # Time Complexity: O(n^2) where n is ncities
        # Space complexity: O(n^2) where n is ncities
        rootState = self.getRootState(startingCity._index)
        queue = PriorityQueue()
        
        # As per the python priority queue documentation,
        # queue insertion is of Time Complexity O(logN)
        # and Space Complexity is O(1)
        queue.put((rootState._bound, rootState))
        self.queues.append(queue)
        
        self.count = 0
        self.currQueueSize = 1
        self.maxQueueSize = 1
        self.prunedStates = 0
        
        self.start_time = time.time()
        
    ''' <summary>
		This is a section of initialization functions to create the root state
        and original cost matrix.
		</summary>
	'''
       
    # Gets the root state using the first city as the starting index.
    # Time Complexity: O(n^2) where n is ncities
    # Space Complexity: O(n^2) where n is ncities
    def getRootState(self, startingIndex):
        # Initialize the four variables to be used in creating the root state
        bound = 0
        
        # See the function definition to get more details
        # Time Complexity: O(n^2) where n is ncities
        # Space complexity: O(n^2) where n is ncities
        costMatrix = self.buildCostMatrix()
        partialPath = [startingIndex]
        depth = 0
        
        # Calculate the original reduced matrix and update the bound cost
        # accordingly, then return the new root state
        # 
        # See the function definition to get more details
        # Time Complexity: O(n^2) where n is ncities
        # Space complexity: O(n^2) where n is ncities
        bound, costMatrix = self.getReducedMatrixAndNewBound(bound, costMatrix)
        self.statesCreated += 1
        return State(self.statesCreated, bound, costMatrix, partialPath, depth)
    
    # This builds the initial cost matrix using the class's variables of ncities and cities
    # Time Complexity: O(n^2) where n is ncities
    # Space Complexity: O(n^2) where n is ncities
    def buildCostMatrix(self):
        costMatrix = [[np.inf for j in range(self.ncities)] for i in range(self.ncities)]
        for i in range(self.ncities):
            for j in range(self.ncities):
                if i == j:
                    continue
                else:
                    costMatrix[i][j] = self.cities[i].costTo(self.cities[j])
        
        return costMatrix
    
    # This ensures all the rows and columns have a 0 in them by finding the min
    # and subtracting it from each cell. It also adds each the min to the
    # bound.
    # Time Complexity: O(n^2) where n is ncities
    # Space complexity: O(n^2) where n is ncities
    def getReducedMatrixAndNewBound(self, bound, costMatrix):
        
        # This ensures all the rows have a 0 in them by finding the min
        # and subtracting it from each cell. It also adds the min to the
        # bound.
        # Time Complexity: O(n^2) where n is ncities
        # Space complexity: O(n) where n is ncities
        for i in range(self.ncities):
            
            # Time Complexity: O(n) where n is the length of the cost matrix's
            # row passed in (always equal to n cities)
            # Space Complexity: O(1)
            minCost = self.branchFindMin(costMatrix[i])
            if minCost is np.inf:
                continue
            bound += minCost
            
            # Actually edit each cell in the current row
            # Time Complexity: O(n) where n is ncities
            # Space Complexity: O(1) - just editing existing values
            for j in range(self.ncities):
                if costMatrix[i][j] is np.inf:
                    continue
                else:
                    costMatrix[i][j] -= minCost
             
        # This ensures all the columns have a 0 in them by finding the min
        # and subtracting it from each cell. It also adds the min to the
        # bound.
        # Time Complexity: O(n^2) where n is ncities
        # Space complexity: O(n^2) where n is ncities
        for i in range(self.ncities):
            
            # Time Complexity: O(n) where n is the length of the cost matrix's
            # column passed in (always equal to n cities)
            # Space Complexity: O(n) where n is ncities because we make a new
            # column variable
            minCost = self.branchFindMin(self.getColumn(i, costMatrix))
            if minCost is np.inf:
                continue
            bound += minCost
            
            # Actually edit each cell in the current column
            # Time Complexity: O(n) where n is ncities
            # Space Complexity: O(1) - just editing existing values
            for j in range(self.ncities):
                if costMatrix[j][i] is np.inf:
                    continue
                else:
                    costMatrix[j][i] -= minCost
        
        return bound, costMatrix
    
    ''' <summary>
		This is the actual branch and bound algorithm that repeats recursively,
        expanding the states and updating the BSSF.
		</summary>
	'''
    
    # This is the branch and bound algorithm itself.
    # The following time/space is worst case scenario
    # Time/Space Complexity: O(bmnqz) where b is the number of states that
    # were generated, m is the size of the biggest queue, n is num cities,
    # q is the number of queues (never bigger than n), and z is the number of
    # times the while loop executes because the queues are not empty
    def branchAndBoundAlgorithm(self, currTime):
        
        # This while loop keeps trying while the queues are not empty
        #
        # The following time/space is worst case scenario
        # Time/Space Complexity: O(bmnqz) where b is the number of states that
        # were generated, m is the size of the biggest queue, n is num cities,
        # q is the number of queues (never bigger than n), and z is the number of
        # times the while loop executes because the queues are not empty
        while not self.queuesEmpty() and time.time()-self.start_time < self.time_allowance:
            bssfUpdated = False
            queueAdded = False
            
            # This loop goes through each queue, and each queue represents a
            # depth level. In order to get better performance on average, the
            # deepest queue is expanded first in order to promote deep drilling,
            # and the iteration is short circuited every time a new depth level
            # is reached or the BSSF was updated.
            #
            # The following time/space is worst case scenario
            # Time/Space Complexity: O(bmnq) where b is the number of states that
            # were generated, m is the size of the biggest queue, n is num cities,
            # and q is the number of queues (never bigger than n)
            for q in range(len(self.queues)):
                
                # Check the short circuit variables
                if bssfUpdated or queueAdded:
                    break
                
                # Calculate the index for reverse iteration
                index = len(self.queues) - q - 1
                
                # If the selected queue is empty, pick another queue
                if self.queues[index].empty():
                    continue
                
                # As per the python documentation, Priority Queue
                # deletion (retreival) is O(logN) time and O(1) space
                state = (self.queues[index].get())[1]
                self.currQueueSize -= 1
                
                if state._bound < self.bssf.cost:
                    
                    # Generate child states, see the function definition for
                    # details.
                    #
                    # Time Complexity: O(n^3) where n is ncities
                    # Space Complexity: O(n^3) where n is ncities
                    states = self.generateStates(state)
                    
                    # The following are for the worst case (i.e., the BSSF is updated
                    # with every state that was generated)
                    # Time/Space Complexity: O(bmn) where b is the number of states that
                    # were generated, m is the size of the biggest queue, and n is num cities
                    for i in range(len(states)):
                        
                        # Test the state, basically checking to
                        # see if the partial path is a complete circuit.
                        # See the function definition for more details.
                        # Time Complexity: O(n) where n is the number of cities. Aborts if
                        # the partial path is incomplete (doesn't include all cities)
                        # Space Complexity: O(1)
                        testedState = self.testState(states[i])
                        if testedState is not None:
                            if testedState.cost < self.bssf.cost:
                                
                                # Here, we update the BSSF, and prune all states
                                # that no longer fit our new BSSF.
                                self.bssf = testedState
                                self.count += 1
                                
                                # Time Complexity: O(mn) where n is the size of the biggest queue
                                # and m is ncities
                                # Space complexity: O(mn) where n is the size of the biggest queue
                                # and m is ncities - the space after this function is called
                                # should be less than the beginning space
                                self.pruneStates()
                                bssfUpdated = True
                        elif states[i]._bound < self.bssf.cost:
                            if index is (len(self.queues) - 1):
                                
                                # Add another queue, meaning we've reached a new
                                # depth level
                                # Time/Space: O(1)
                                self.queues.append(PriorityQueue())
                                queueAdded = True
                                
                            # As per the python Priority Queue documentation,
                            # Insertion is O(logN) time and O(1) space
                            self.queues[index + 1].put((states[i]._bound, states[i]))
                            self.currQueueSize += 1
                            
                            # Adjust the max queue size if the current
                            # queue is greater than the max queue size
                            # O(1) time and space
                            if self.currQueueSize > self.maxQueueSize:
                                self.maxQueueSize = self.currQueueSize
                        else:
                            self.prunedStates += 1
             
        # Add all the states that are still on
        # the queue to the number of pruned states
        # and return the best solution so far.
        self.prunedStates += self.currQueueSize
        return self.bssf
    
    ''' <summary>
		This is a section of functions to create child states when expanding 
        a particular state.
		</summary>
	'''
    
    # Time Complexity: O(n^3) where n is ncities
    # Space Complexity: O(n^3) where n is ncities
    def generateStates(self, parentState):
        states = []
        
        # Run through all the cities and if one is not in the partial path
        # of the parent state, try to generate a state for it.
        #
        # Time Complexity: O(n^3) where n is ncities
        # Space Complexity: O(n^3) where n is ncities
        for i in self.cities:
            
            # Time Complexity: O(n) where n is the number of cities in the partial path
            # Space Complexity: O(1)
            if i._index in parentState._partialPath:
                continue
            else:
                if self.scenario._edge_exists[parentState._partialPath[-1]][i._index]:
                    
                    # Generate the state as long as we know there's a path between the
                    # last visited city and the next city
                    #
                    # See the function definition to get more details
                    # Time Complexity: O(n^2) where n is ncities
                    # Space complexity: O(n^2) where n is ncities
                    state = self.generateState(parentState, i)
                    if state is not None:
                        states.append(state)
                
        return states
    
    # This function creates a new individual state based on the given parent
    # state and selected next city.
    # Time Complexity: O(n^2) where n is ncities
    # Space complexity: O(n^2) where n is ncities
    def generateState(self, parentState, nextCity):
        # Initialize the four variables to be used in creating a new state
        # As per the python documentation, the time complexity of deep copies
        # is O(1) amortized time.
        # Space complexity: O(1)
        bound = copy.deepcopy(parentState._bound)
        costMatrix = copy.deepcopy(parentState._matrix)
        partialPath = copy.deepcopy(parentState._partialPath)
        depth = copy.deepcopy(parentState._depth) + 1
        
        # Get the indices of the cities for cost matrix referencing
        indexCurrCity = partialPath[-1]
        indexNextCity = nextCity._index
        
        # Add the new city to the partial path
        partialPath.append(nextCity._index)
        
        # Add the cost of going to the next city to the bound
        bound += costMatrix[indexCurrCity][indexNextCity]
        
        # Make the row and column corresponding to the current and next
        # city infinity in the cost matrix
        # Time complexity: O(n) where n is ncities
        # Space complexity: O(1)
        for i in range(self.ncities):
            costMatrix[indexCurrCity][i] = np.inf
            costMatrix[i][indexNextCity] = np.inf
        
        # Make the return also infinity
        costMatrix[indexNextCity][indexCurrCity] = np.inf
        
        # Recalculate the reduced matrix and add any additional costs
        # to the bound, then return the new state
        # 
        # See the function definition to get more details
        # Time Complexity: O(n^2) where n is ncities
        # Space complexity: O(n^2) where n is ncities
        bound, costMatrix = self.getReducedMatrixAndNewBound(bound, costMatrix)
        self.statesCreated += 1
        
        if bound > self.bssf.cost:
            self.prunedStates += 1
            return None
        
        # The creation of a new State object happens in constant time and space
        return State(self.statesCreated, bound, costMatrix, partialPath, depth)
    
    ''' <summary>
		This is a section of helper functions to assist with the algorithms above.
		</summary>
	'''
    
    # Time Complexity: O(mn) where n is the size of the biggest queue
    # and m is ncities
    # Space complexity: O(mn) where n is the size of the biggest queue
    # and m is ncities - the space after this function is called
    # should be less than the beginning space
    def pruneStates(self):
        self.currQueueSize = 0
        
        # Go through the queues in our queue array
        # Time Complexity: O(mn) where n is the size of the biggest queue
        # and m is ncities
        # Space complexity: O(mn) where n is the size of the biggest queue
        # and m is ncities
        for q in range(len(self.queues)):
            prunedQueue = PriorityQueue()
            
            # While the current queue is not empty
            # Time Complexity: O(n) where n is the size of the queue
            # Space Complexity: O(n) where n is the number of states
            # that still have lower bounds than the BSSF cost
            while not self.queues[q].empty():
                
                # As per the python priority queue documentation,
                # queue deletion (retrieval) is of Time Complexity O(logN)
                # and Space Complexity is O(1)
                stateTuple = (self.queues[q].get())
                
                # As per the python priority queue documentation,
                # queue insertion is of Time Complexity O(logN)
                # and Space Complexity is O(1)
                if stateTuple[0] < self.bssf.cost:
                    prunedQueue.put(stateTuple)
                    self.currQueueSize += 1
                else:
                    self.prunedStates += 1
            self.queues[q] = prunedQueue
    
    # Time Complexity: O(n) where n is the number of cities. Aborts if
    # the partial path is incomplete (doesn't include all cities)
    # Space Complexity: O(1)
    def testState(self, state):
        if len(state._partialPath) is self.ncities:
            route = []
            
            # Time Complexity: O(n) where n is the number of cities.
            # Space Complexity: O(1)
            for i in range(self.ncities):
                route.append(self.cities[state._partialPath[i]])
                
            # Time Complexity: O(n) where n is the number of cities.
            # Space Complexity: O(1)
            return TSPSolution(route)
        else:
            return None

    # Time Complexity: O(n) where n is the length of the cost matrix's
    # row or column passed in (always equal to n cities)
    # Space Complexity: O(1)
    def branchFindMin(self, costList):
        minCost = np.inf
        for i in range(len(costList)):
            if costList[i] < minCost:
                minCost = costList[i]
                    
        return minCost           
         
    # Time Complexity: O(n) where n is the number of cities
    # Space Complexity: O(n) where n is the number of cities
    def getColumn(self, colIndex, costMatrix):
        column = []
        for j in range(self.ncities):
            column.append(costMatrix[colIndex][j])
            
        return column
    
    # Time Complexity: O(n) where n is the current length of the queue array
    # Space Complexity: O(1)
    def queuesEmpty(self):
        for i in range(len(self.queues)):
            if not self.queues[i].empty():
                return False
            
        return True