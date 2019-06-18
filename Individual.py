import random
import copy


class Individual:

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    def __init__(self, aNumberOfGenes, aBoundarySet, aFitnessFunction, aGeneSet = None, aFitness = -float('inf')):

        # Set the genes
        self.genes = [];

        # Set fitness value
        self.fitness = aFitness;

        # Set the fitness function
        self.fitness_function = aFitnessFunction;

        # Set boundaries
        self.boundary_set = copy.deepcopy(aBoundarySet)

        if aGeneSet != None:
            for i in aGeneSet:
                self.genes.append(i)
        else:
            # Initialise the Individual's genes and fitness
            for i in range(aNumberOfGenes):
                # Get the boundaries
                min_i = self.boundary_set[i][0];
                max_i = self.boundary_set[i][1];

                # Get random value to each gene
                self.genes.append(Individual.system_random.uniform(min_i, max_i));


    def copy(self):

        return Individual(
                len(self.genes),
                self.boundary_set,
                self.fitness_function,
                self.genes,
                self.fitness
        );


    def computeObjectiveFunction(self):

        # Compute the fitness function
        self.fitness = self.fitness_function.evaluate(self.genes, 2);

        return self.fitness;

    def getParameter(self, i):
        if i >= len(self.genes):
            raise IndexError;
        else:
            return self.genes[i];

    def getObjective(self):
        return self.fitness;

    def __repr__(self):

        value = "Genes: ";
        value += ' '.join(str(e) for e in self.genes)
        value += "\tFitness: ";
        value += str(self.fitness)

        return value;
