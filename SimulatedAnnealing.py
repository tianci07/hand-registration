"""@package SimulatedAnnealing
This package implements the simulated annealing (SA) optimisation method. SA is a metaheuristic to find the global optimum in an optimization problem.
For details, see https://en.wikipedia.org/wiki/Simulated_annealing and  Kirkpatrick, S.; Gelatt Jr, C. D.; Vecchi, M. P. (1983). "Optimization by Simulated Annealing". Science. 220 (4598): 671–680. doi:10.1126/science.220.4598.671.
@author Dr Franck P. Vidal, Bangor University
@date 15th May 2019
"""

#################################################
# import packages
###################################################
import math; # For exp
import copy; # For deepcopy
import random; # For uniform

from Optimiser import *

class Solution:
    def __init__(self, aParameterSet = None):

        self.parameter_set = [];

        if aParameterSet != None:
            self.parameter_set = copy.deepcopy(aParameterSet);

        self.energy = float('inf');

    def getParameter(self, i):
        if i >= len(self.parameter_set):
            raise IndexError;
        else:
            return self.parameter_set[i];

    def getObjective(self):
        return self.energy;

    def __repr__(self):
        value = "Parameters: ";
        value += ' '.join(str(e) for e in self.parameter_set)
        value += "\tEnergy: ";
        value += str(self.energy);
        return value;


## \class This class implements the simulated annealing optimisation method
class SimulatedAnnealing(Optimiser):

    ## \brief Constructor.
    # \param self
    # \param aCostFunction: The cost function to minimise
    # \param aTemperature: The initial temperature of the system (default value: 10,000)
    # \param aCoolingRate: The cooling rate (i.e. how fast the temperature will decrease) (default value: 0.003)
    def __init__(self, aCostFunction, aTemperature = 10000, aCoolingRate = 0.003):

        super().__init__(aCostFunction);

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

        # Create the current solution from random
        parameter_set = [];
        for i in range(self.objective_function.number_of_dimensions):
            parameter_set.append(self.system_random.uniform(self.objective_function.boundary_set[i][0], self.objective_function.boundary_set[i][1]));
        self.current_solution = Solution(parameter_set);

        # and copy input parameters
        self.initial_temperature = aTemperature;
        self.cooling_rate = aCoolingRate;

        # Initialise attributes
        self.initStates()

    ## \brief Initialise attributes.
    # \param self
    def initStates(self):
        self.min_energy =  float("inf");
        self.max_energy = -float("inf");

        self.temperature_set = [];

        self.best_solution_set = [];

        self.current_temperature = self.initial_temperature;

        # Compute its energy using the cost function
        self.current_solution.energy = self.computeEnergy(self.current_solution.parameter_set);

        # This is also the best solution so far
        self.best_solution = copy.deepcopy(self.current_solution);


    ## \brief Compute the energy corresponding to a given solution.
    # \param self
    # \param aSolution: The solution to assess
    # \return the corresponding energy
    def computeEnergy(self, aSolution):
        # Compute the cost function
        energy = self.objective_function.evaluate(aSolution, 1);

        # Keep track of the min/max cost values
        self.min_energy = min(self.min_energy, energy);
        self.max_energy = max(self.max_energy, energy);

        # Return the corresponding cost
        return energy;

    ## \brief Compute the acceptance probability corresponding to an energy.
    # \param self
    # \param aNewEnergy: The energy to assess
    # \return the corresponding acceptance probability
    def acceptanceProbability(self, aNewEnergy):

        # The new soluation is better (lower energy), keep it
        if aNewEnergy < self.current_solution.getObjective():
            return 1.0;
        # The new soluation is worse, calculate an acceptance probability
        else:
            return math.exp((self.current_solution.getObjective() - aNewEnergy) / self.current_temperature);

    ## \brief Get a neighbour in the vicinity of a given solution.
    # \param self
    # \param aSolution: The solution to assess
    # \return a neighbour
    def getRandomNeighbour(self, aSolution):
        # Start with an empty solution
        new_solution = [];

        # Process each dimension of the search space
        for i in range(self.objective_function.number_of_dimensions):
            min_val = self.objective_function.boundary_set[i][0];
            max_val = self.objective_function.boundary_set[i][1];
            range_val = max_val - min_val;
            new_solution.append(random.gauss(min_val + range_val / 2, range_val * 0.1));

        return (copy.deepcopy(new_solution));

    ## \brief Get a neighbour in the vicinity of a given solution.
    # \param self
    # \param aSolution: The solution to assess
    # \return a neighbour
    def getRandomNeighbor(self, aSolution):
        return self.getRandomNeighbour(aSolution);

    def evaluate(self, aParameterSet):
        return self.objective_function.evaluate(aParameterSet, 1);

    ## \brief Run one iteration of the SA algorithm.
    # \param self
    def runIteration(self):
        if self.current_temperature > 1.0:
            # Create a new solution depending on the current solution,
            # i.e. a neighbour
            neighbour = Solution(self.getRandomNeighbour(self.current_solution));

            # Get its energy (cost function)
            neighbour.energy = self.computeEnergy(neighbour.parameter_set);

            # Accept the neighbour or not depending on the acceptance probability
            if self.acceptanceProbability(neighbour.getObjective()) > self.system_random.uniform(0, 1):
                self.current_solution = copy.deepcopy(neighbour);

            # The neighbour is better thant the current element
            if self.best_solution.getObjective() > self.current_solution.getObjective():
                self.best_solution = copy.deepcopy(self.current_solution);

            # Log the current states
            self.logCurrentState();

            # Cool the system
            self.current_temperature *= 1.0 - self.cooling_rate;


    ## \brief Run the optimisation.
    # \param self
    # \param aRetartFlag: True if the algorithm has to run twice, False if it has to run only once (default value: False)
    # \param aVerboseFlag: True if intermediate results are printing in the terminal, False to print no intermediate results (default value: False)
    def run(self, aRetartFlag = False, aVerboseFlag = False):

        self.initStates();

        iteration = 0;
        if aVerboseFlag:
            header  = "iteration";
            header += " temperature";
            for i in range(self.objective_function.number_of_dimensions):
                header += " best_solution[" + str(i) + "]";
            header += " best_solution_energy";
            for i in range(self.objective_function.number_of_dimensions):
                header += " current_solution[" + str(i) + "]";
            header += " current_solution_energy";
            print(header);
            print(self.iterationDetails(iteration));

        # Log the current states
        self.logCurrentState();

        # Loop until system has cooled
        while self.current_temperature > 1.0:

            if aRetartFlag:
                if iteration != 0:
                    if (self.current_solution.getObjective() - self.min_energy) / (self.max_energy - self.min_energy) > 0.9:
                        #print("Restart")
                        self.current_solution = copy.deepcopy(self.best_solution);

            # Run one iteration of the loop
            self.runIteration();
            iteration = iteration + 1;

            if aVerboseFlag:
                print(self.iterationDetails(iteration));

        self.current_solution = copy.deepcopy(self.best_solution);

    ## \brief Log the current states to save the history (can be used to visualise how the algorithm behaved over time)
    # \param self
    def logCurrentState(self):
        self.temperature_set.append(self.current_temperature);
        self.current_solution_set.append(self.current_solution);
        self.best_solution_set.append(self.best_solution);

    ## \brief Print the current solution and the best solution so far.
    # \param self
    # \return a string that includes the current solution and the best solution so far (parameters and corresponding costs)
    def iterationDetails(self, iteration):
        return (str(iteration) + ', ' +
            str(self.current_temperature) + ', ' +
            self.best_solution.__repr__());

    ## \brief Print the best solution.
    # \param self
    # \return a string that includes the best solution parameters and its corresponding cost
    def __repr__(self):
        return self.best_solution.__repr__();
