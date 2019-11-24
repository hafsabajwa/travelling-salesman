# travelling-salesman
Python version: 2.7
Testrun: Macbook Pro OS 10.15.1

OBS: NOTE THAT THIS PROGRAM IS WRITTEN IN PYTHON 2.7

This problems solves the travellings salesman  - i.e. the problem of finding the shortes distance between locations - using one, several, or all, of four different algorithms: Exhaustivve search, Hill-climbing, Genetic algorithm, and Hybrid algorithm. These are described below. To run the program, just create an instance of the class Travel with the desired settings passed to the constructor (details in the documentation). Then execute the ‘run’ method. An example is given at the bottom of the script.

Overview:
The class Travel takes as arguments which methods (exhaustive/hill-climbing/genetic algorithm/hybrid) the user wants to run and the number of runs as well as some other arguments specific only to some of the methods (e.g. the population size for the genetic algorithm). Most of the work is done within the functions of this class, except for the genetic algorithm, which is mostly executed in functions of the class Population.

For all methods:
The timeit.repeat function times the execution of the method and prints the time afterwards. The function takes an argument repeat which allows the methods to be run and timed separately several times.

Exhaustive search:
The itertools.permutations functions creates a list of all permutations of the cities (i.e. order in which the cities are visited) that the program loops over in order to find the one permutation with the shortest distance. The best order and shortest distance is stored and printed to the screen at the end of the run.

Hill-climbing:
The hill-climbing algorithm is an implementation of the one suggested in Marshland 2015. It starts with a random city order, calculates the distance, swaps two of the cities, calculates the distance again, and keeps the new order if the new distance is shorter. It then repeats this procedure until it has run a given number of iterations (default 1000) without improvement.

Genetic algorithm:
As mentioned, this method has its own class: Population. It begins by creating a population of N randomly shuffled city orders, where N is the given population size. It then sorts the city orders in order of increasing distance. The parent_selection method selects which members of the parent generation to include in the mating pool through an implementation of the roulette wheel algorithm suggested by Eiben and Smith. For recombination and mutation I have implemented the Parially Mapped Crossover and inversion algorithms, respectively. The survivor selection is deterministic in that all the offspring and the 1/10 most fit individuals from the parent generation (including all members of this generation, not just the ones in the mating pool) are selected to make up the next generation. The algorithm stops when it has run for a given number of generations in total (default 1000)

Hybrid method:
The hybrid method works like the genetic algorithm, except that after each generation the hill- climbing algorithm is run once with each individual of the population as the starting state. If
the hybrid_type is set to “lamarckian” both the new (improved) city order and the new distance is stored, whereas if the hybrid type is set to “baldwinian” only the new distance (=improved fitness) is stored.

References:
Marshland, S. (2015). Machine Learning: An Algorithmic Approach. CRC Press.
