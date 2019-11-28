import itertools
import timeit
import random
import copy
import numpy as np
import matplotlib.pyplot as pp

# Oystein Kapperud, 2017

'''
OBS: NOTE THAT THIS PROGRAM IS WRITTEN IN PYTHON 2.7

This problems solves the travellings salesman - i.e. the problem of finding the shortes distance between locations - 
using one, several, or all, of four different algorithms: Exhaustivve search, Hill-climbing, Genetic algorithm, and Hybrid algorithm. 

'''


class Travel:
	def __init__(self, filename, n = 'all', exhaustive = False, hill = False, ga = False, hybrid = False, runs = 1, population_size = 200, stop_condition_hill = 1000, stop_condition_ga = 1000, hybrid_type_lamarckian = True):
		"""
		A class for solving Traveling Salesman problems, i.e. the problem of finding the shortest tour between several cities given a distance matrix.
		
		Parameters:
		filename (str): filename (or -path) for file with distance matrix
		n (int or str): number of cities to be included in the analysis. If 'all' is passed, all cities in the distance matrix file will be included (default: 'all')
		exhaustive (boolean): the 'run' method runs an exhaustive search if True (default: False)
		hill (boolean):  the 'run' method runs a hill-climbing algorithm if True (default: False)
		ga (boolean):  the 'run' method runs a genetic algorithm if True (default: False)
		hybrid (boolean):  the 'run' method runs a hybrid algorithm (a genetic algortihm with lamarckian or baldwinian inheritance) if True (default: False)
		runs (int): the number of runs the program will execute (default: 1)
		population_size (int): the population size in the genetic algroithm and hybrid algorithm (default: 200)
		stop_condition_hill (int): the number of generation without improvement the hill-climber will run before it stops (default: 1000)
		stop_condition_ga (int): the number of generations the genetic algorithm and hybrid algorithm will run in total before it stops (default: 1000)
		hybrid_type_lamarckian (boolean): if True, the hybrid algorithm will run with lamarckian inheritance (if 'hybrid' is set to True). if False, the hybrid algorithm will run with baldwinian inheritance (default: True)
		
		"""
		self.n = n
		self.hill = hill
		self.exhaustive = exhaustive
		self.ga = ga
		self.hybrid = hybrid
		self.distances = []
		self.readfile(filename)
		self.runs = runs
		self.best_orders_hill = []
		self.shortest_distances_hill = []
		self.shortest_distances_ga = []
		self.shortest_distances_hybrid = []
		self.population_size = population_size
		self.stop_condition_hill = stop_condition_hill
		self.stop_condition_ga = stop_condition_ga
		self.hybrid_type_lamarckian = hybrid_type_lamarckian
		if n == 'all':
			print 'All cities included'
		else:
			print 'Only the first', n, 'cities included'

	
	def run(self):
		if self.exhaustive:
			t1 = timeit.Timer(self.exhaustive_search)
			ex = t1.repeat(self.runs, 1)
			print '\nTime: ', ex
			print ''
		if self.hill:
			t2 = timeit.Timer(self.hillclimbing)
			h = t2.repeat(self.runs, 1)
			if self.runs>1:
				print '\nShortest distances: ', self.shortest_distances_hill
			print '\nTime: ', h
			print ''
		
		if self.ga:
			t3 = timeit.Timer(self.genetic)
			ga = t3.repeat(self.runs, 1)
			if self.runs>1:
				print '\nShortest distances: ', self.shortest_distances_ga
			print '\nTime: ', ga
			print ''
		
		if self.hybrid:
			t4 = timeit.Timer(self.hybrid_method)
			hy = t4.repeat(self.runs,1)
			if self.runs>1:
				print '\nShortest distances: ', self.shortest_distances_hybrid
			print '\nTime: ', hy
			print ''
			

	def readfile(self, filename):
		infile = open(filename, 'r')
		firstline = True
		i = 0
		for line in infile:
			if firstline: 
				self.citynames = line.split(';') # read city names
				if self.n == 'all':
					self.n = len(self.citynames)
				self.index = range(self.n)
				firstline = False
			else:
				self.distances.append([float(d.strip()) for d in line.split(';')])


	
	def hillclimbing(self, start = None):
		"""
		This function implements a hill-climbing algorithm, as suggested in Marshland 2015
		"""
		if start is None:
			current_order = copy.copy(self.index)
			random.shuffle(current_order)
			max_number_of_iterations = float('inf')
			current_distance = self.find_distance(current_order)
			progress = [current_distance]
			
		else:
			current_order = start
			max_number_of_iterations = 10
			current_distance = self.find_distance(current_order)
		n_without_improvement = 0
		iteration_count = 0
		while n_without_improvement < self.stop_condition_hill and iteration_count < max_number_of_iterations: # stop the search when the algorithm has gone a given number (stop_condition_hill) of iterations without imporving the distance or when the maximum number of iterations is reached
			picklist = copy.copy(current_order)
			city1 = random.choice(picklist)
			picklist.remove(city1)
			city2 = random.choice(picklist)
			i = current_order.index(city1)
			j = current_order.index(city2)
			new_order = copy.copy(current_order)
			new_order[i], new_order[j] = new_order[j], new_order[i]
			new_distance = self.find_distance(new_order)
			if new_distance < current_distance:
				current_order = new_order
				current_distance = new_distance
				n_without_improvement = 0
			else:
				n_without_improvement += 1
			iteration_count += 1
			if start == None:
				progress.append(current_distance)
		if start == None:
			print "\n***"
			print "RESULTS HILLCLIMBING:"
			print "Best order: ", current_order
			cities = ''
			for city_index in current_order:
				cities = cities + self.citynames[city_index] + ', '
			cities = cities[:-2]
			print cities
			print ''
			print "Distance: ", current_distance
			print "Progress: "
			print progress
			self.shortest_distances_hill.append(current_distance)
		if start is not None:
			return [current_order, current_distance]

			
	def find_distance(self, order):
		"""
		This function returns the distance of a given order of cities
		"""
		distance = 0
		for i in range(self.n-1):
			distance += self.distances[order[i]][order[i+1]]
		distance += self.distances[order[self.n-1]][order[0]]
		return distance
	
	
	def exhaustive_search(self):
		"""
		This function iterates over all possible permutations of the cities, and finds the one with the shortest distance.
		"""
		shortest_distance = None
		best_order = None
		c = 0
		for order in itertools.permutations(self.index):
			distance = self.find_distance(order)
			if shortest_distance is None:
				shortest_distance = distance
				best_order = order
			else:
				if distance < shortest_distance:
					shortest_distance = distance
					best_order = order
			c += 1
		print "\n***"
		print 'RESULTS EXHAUSTIVE:'
		print "Best order: ", best_order
		cities = ''
		for city_index in best_order:
			cities = cities + self.citynames[city_index] + ', '
		cities = cities[:-2]
		print cities
		print ''
		print "Shortest distance: ", shortest_distance
	
	def genetic(self):
		population = Population(self)
		population.evolve()
	
	
	def hybrid_method(self):
		population = Population(self)
		population.evolve(hybrid = True)
		
		

class Population:
	def __init__(self, travel):
		self.travel = travel
		self.population_size = travel.population_size
		genotypes = []
		self.progress = []
		self.indices = np.array(range(self.population_size))
		distances = [] # the order of the 'genotypes' and 'distances' lists will be sorted so that the distance of the genotype at index i in 'genotypes' will be found at index i in 'distances' 
		self.generation = 0
		self.gens_without_improvement = 0 # the number of generations the algorithm has run without finding a shorter distance
		self.current_best_genotype = None
		self.current_shortest_distance = None
		self.mutrate = 0.8 # mutation rate
		self.recrate = 0.8 # recombination rate
		for i in range(self.population_size): # create a population of n randomly shuffled orders
			g = range(self.travel.n)
			random.shuffle(g)
			genotypes.append(g)
			d = self.travel.find_distance(g)
			distances.append(d)
		self.genotypes = np.array(genotypes) # convert to numpy arrays
		self.distances = np.array(distances)
		self.sort_genotypes_and_find_cumulative_fitness()
		
		
		
		
	def sort_genotypes_and_find_cumulative_fitness(self):
		"""
		This function sorts the genotypes and corresponding distances and fitnesses from highest to lowest fitness
		"""
		indices = self.distances.argsort() # sorts the indices
		self.fitness = 1/self.distances # set fitness to the inverse of the distance, so that genotypes with short distances have high fitness
		self.fitness = self.fitness/np.sum(self.fitness) # normalize fitness
		self.fitness = self.fitness[indices]
		self.genotypes = self.genotypes[indices]
		self.distances = self.distances[indices]
		
		if self.current_shortest_distance is not None:
			if self.distances[0] < self.current_shortest_distance:
				self.gens_without_improvement = 0
			else:
				self.gens_without_improvement += 1
		
		self.progress.append(self.distances[0])
		current_best_genotype = self.genotypes[0]
		self.current_shortest_distance = self.distances[0]
		self.cumulative_fitness = np.cumsum(self.fitness)
		
		self.generation += 1
	
	def evolve(self, hybrid = False):
		"""
		This function runs the algorithm until there has been a certain number (stop_condition) of generations without improvement
		"""
		generation = 0
		while generation < self.travel.stop_condition_ga:
			self.parent_selection()
			self.mate()
			self.mutate()
			self.survivor_selection()
			self.sort_genotypes_and_find_cumulative_fitness()
			if hybrid: # If the hybrid option is selected, the algorithm will run the hill-climbing algrotihm once on each member of the population
				for i in range(len(self.genotypes)):
					genotype = list(self.genotypes[i])
					hill_climbing_results = self.travel.hillclimbing(start = genotype)
					if self.travel.hybrid_type_lamarckian:
						self.genotypes[i] = hill_climbing_results[0]
					self.distances[i] = hill_climbing_results[1]
			generation += 1
		print '\n***'
		results_string = 'RESULTS FOR '
		if hybrid:
			results_string += 'HYBRID '
			if self.travel.hybrid_type_lamarckian:
				results_string += 'LAMARCKIAN '
			else:
				results_string += 'BALDWINIAN '
		else:
			results_string += 'GENETIC '
		results_string += 'ALGORITHM:'
		
		print results_string
		best_order = self.genotypes[0]
		print 'Best order: ', best_order
		cities = ''
		for city_index in best_order:
			cities = cities + self.travel.citynames[city_index] + ', '
		cities = cities[:-2]
		print cities
		print 'Distance: ', self.distances[0]
		print 'Progress: '
		print self.progress
		if hybrid:
			self.travel.shortest_distances_hybrid.append(self.distances[0])
		else:
			self.travel.shortest_distances_ga.append(self.distances[0])
						
			
			
	def parent_selection(self):
		"""
		This function selects the genotypes to be included in the mating pool by implementing the roulette wheel algorithm presented in Eiben and Smith 2015. The size of the mating pool is set to 1/10 of the total population.
		"""
		self.n_select = self.population_size/10
		self.mating_pool = []
		
		while len(self.mating_pool) < self.n_select:
			r = random.uniform(0, 1)
			i = 0
			while self.cumulative_fitness[i] < r:
				i = i+1
			self.mating_pool.append(copy.copy(self.genotypes[i]))
			
	
	def mate(self):
		"""
		This function loops over the mating pool list so as to repeatedly pair up two and two parents for recombination 
		(the actual recombination is done in the 'recombine' function). In the first iteration the parents are paired 
		in order of sorted fitness (so that the most fit and second most fit parent pairs, then the third and fourth most fit, etc), 
		and in subsequent iterations the the parents are paired randomly. The algorithm runs until populations_size - population_size/10
		offspring have been created. This is because the survivor selection function deterministically selects the 1/10 fittest 
		members of the previous generation along with all the offspring, so that the total population size stays the same.
		"""
		self.offspring = []
		self.offspring_fitness = []
		self.offspring_distances = []
		i = 0
		while(len(self.offspring) <= self.population_size- self.population_size/10 -2):
			self.recombine(self.mating_pool[i], self.mating_pool[i+1])
			self.recombine(self.mating_pool[i+1], self.mating_pool[i])
			if i >= len(self.mating_pool)-2:
				i = 0
				random.shuffle(self.mating_pool)
			else:
				i += 2
		
	def recombine(self, parent1, parent2):
		"""
		This function implements the Partially Mapped Crosover algorithm presented in Eiben and Smith 2015 so as to combine the
		the genotype of the two parents (parent1 and parent2) into a new offspring.
		"""
		if random.random() < self.recrate:
			new_offspring =[-1 for i in range(len(parent1))]
			parent1 = list(parent1)
			parent2 = list(parent2)
			c1 = random.randint(0, self.travel.n)
			c2 = random.randint(c1, self.travel.n)
			if c1 == c2:
				new_offspring = copy.copy(parent2)
			else:
				new_offspring[c1:c2] = parent1[c1:c2]
				for i in range(c1, c2):
					e2 = parent2[i]
					if e2 not in new_offspring:
						e1 = parent1[i]
						j = parent2.index(e1)
						while new_offspring[j] != -1:
							j = parent2.index(parent1[j])
						new_offspring[j] = e2
				for i in range(len(parent2)):
					if new_offspring[i] == -1:
						new_offspring[i] = parent2[i]
			new_offspring = np.array(new_offspring)
		else:
			new_offspring = np.array(parent1)
		
		distance = self.travel.find_distance(new_offspring)
		self.offspring_distances.append(distance)
		self.offspring_fitness.append(1/distance)
		self.offspring.append(new_offspring)
		

	def mutate(self):
		"""
		This function iterates over all the offspring and mutates each one with probability 'mutrate' by inverting a random segment of the genome.
		"""
		for individual in self.offspring:
			if random.random() < self.mutrate:
				c1 = random.randint(0, len(individual))
				c2 = random.randint(c1, len(individual))
				new_segment = individual[c1:c2][::-1]
				individual[c1:c2] = new_segment
	
	
	def survivor_selection(self):
		"""
		This function deterministically selects all the offspring and the 1/10 most fit parents to form the next generation.
		"""
		n_surv_from_parent_gen = self.population_size/10
		self.genotypes = list(self.genotypes)
		self.distances = list(self.distances)
		self.fitness = list(self.fitness)
		self.genotypes = self.genotypes[0:n_surv_from_parent_gen] + self.offspring
		self.fitness = self.fitness[0:n_surv_from_parent_gen] + self.offspring_fitness
		self.distances = self.distances[0:n_surv_from_parent_gen] + self.offspring_distances
		self.genotypes = np.array(self.genotypes)
		self.fitness = np.array(self.fitness)
		self.distances = np.array(self.distances)


if __name__ ==  '__main__':
	t = Travel('european_cities.csv', 5 , ga = True, exhaustive = True, hill = True, hybrid = True, hybrid_type_lamarckian = True, runs = 1, population_size = 200, stop_condition_ga = 1000, stop_condition_hill = 1000)
	t.run()
