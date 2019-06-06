
import random
import Individual as IND
import numpy as np
import math
import SelectionOperator

from Optimiser import *

class EvolutionaryAlgorithm(Optimiser):

    def __init__(self, aFitnessFunction, aNumberOfIndividuals, aGlobalFitnessFunction = 0, aUpdateIndividualContribution = 0):

        super().__init__(aFitnessFunction);

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

        # Selection operator
        self.selection_operator = SelectionOperator.SelectionOperator();

        # Genetic operators
        self.genetic_opterator_set = [];
        self.elitism_operator = None;

        # Store the population
        self.current_solution_set = [];

        # New individual callback
        self.individual_callback = 0;
        if aUpdateIndividualContribution:
            self.individual_callback = aUpdateIndividualContribution;

        # Keep track of the best individual
        self.current_solution_set.append(IND.Individual(self.objective_function.number_of_dimensions, self.objective_function.boundary_set, aFitnessFunction));

        # Create the population
        while (len(self.current_solution_set) < aNumberOfIndividuals):
            self.current_solution_set.append(IND.Individual(self.objective_function.number_of_dimensions, self.objective_function.boundary_set, aFitnessFunction))

        # Compute the global fitness
        self.global_fitness = 0;
        self.global_fitness_function = 0;

        if aGlobalFitnessFunction:

            set_of_individuals = [];
            for ind in self.current_solution_set:
                for gene in ind.genes:
                    set_of_individuals.append(gene);

            self.global_fitness_function = aGlobalFitnessFunction;
            self.global_fitness = self.global_fitness_function(set_of_individuals);

        # Compute the fitness value of all the individual
        # And keep track of who is the best individual
        best_individual_index = 0;
        for i in range(len(self.current_solution_set)):
            self.current_solution_set[i].computeObjectiveFunction();
            if (self.current_solution_set[best_individual_index].fitness < self.current_solution_set[i].fitness):
                best_individual_index = i;

        # Store the best individual
        self.best_solution = self.current_solution_set[best_individual_index].copy();

    def addGeneticOperator(self, aGeneticOperator):
        if aGeneticOperator.getName() == "Elitism operator":
            self.elitism_operator = aGeneticOperator
        else:
            self.genetic_opterator_set.append(aGeneticOperator);

    def clearGeneticOperatorSet(self):
        self.genetic_opterator_set = [];
        self.elitism_operator = None;

    def setSelectionOperator(self, aSelectionOperator):
        self.selection_operator = aSelectionOperator;

    def evaluate(self, aParameterSet):
        return self.objective_function.evaluate(aParameterSet, 2);

    def runIteration(self):

        if self.selection_operator == None:
            raise NotImplementedError("A selection operator has to be added")

        self.selection_operator.preProcess(self.current_solution_set);

        offspring_population = [];
        negative_fitness_parents = []

        best_individual_index = 0;

        # Sort index of individuals based on their fitness
        # (we use the negative of the fitness so that np.argsort returns
        # the array of indices in the right order)
        for i in range(len(self.current_solution_set)):
            negative_fitness_parents.append(-self.current_solution_set[i].fitness)
            #print("fitness  ",self.current_solution_set[i].fitness)

        # Sort the array of negative fitnesses
        index_sorted = np.argsort((negative_fitness_parents))

        # Retrieve the number of individuals to be created by elitism
        number_of_individuals_by_elitism = 0;


        if self.elitism_operator != None:
            math.floor(self.elitism_operator.getProbability() * len(self.current_solution_set))

        # Make sure we keep the best individual
        # EVEN if self.elitism_probability is null
        # (we don't want to lose the best one)
        if number_of_individuals_by_elitism == 0:
            number_of_individuals_by_elitism =  1

        #print(number_of_individuals_by_elitism)

        # Copy the best parents into the population of children
        for i in range(number_of_individuals_by_elitism):
            individual = self.current_solution_set[index_sorted[i]]
            offspring_population.append(individual.copy())
            if self.elitism_operator != None:
                self.elitism_operator.use_count += 1;

        probability_sum = 0.0;
        for genetic_opterator in self.genetic_opterator_set:
            probability_sum += genetic_opterator.getProbability();

        # Evolutionary loop
        while (len(offspring_population) < len(self.current_solution_set)):

            # Draw a random number between 0 and 1 minus the probability of elitism
            chosen_operator = self.system_random.uniform(0.0, probability_sum)

            accummulator = 0.0;
            current_number_of_children = len(offspring_population)

            for genetic_opterator in self.genetic_opterator_set:
                if genetic_opterator.getName() != "Elitism operator":
                    if current_number_of_children == len(offspring_population):

                        accummulator += genetic_opterator.getProbability();

                        if (chosen_operator <= accummulator):
                            offspring_population.append(genetic_opterator.apply(self));

        # Compute the global fitness
        if self.global_fitness:

            set_of_individuals = [];
            for ind in offspring_population:
                for gene in ind.genes:
                    set_of_individuals.append(gene);

            self.global_fitness = self.global_fitness_function(set_of_individuals);

        # Compute the fitness value of all the individual
        # And keep track of who is the best individual
        best_individual_index = 0;

        for child in offspring_population:
            child.computeObjectiveFunction();
            if (offspring_population[best_individual_index].fitness < child.fitness):
                best_individual_index = offspring_population.index(child);

        # Replace the parents by the offspring
        self.current_solution_set = offspring_population;

        # Store the best individual
        self.best_solution = self.current_solution_set[best_individual_index].copy();

        # Return the best individual
        return self.best_solution;
