import random
import copy
import GeneticOperator


class GaussianMutationOperator(GeneticOperator.GeneticOperator):

    # Contructor
    def __init__(self, aProbability, aMutationVariance):
        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Gaussian mutation operator";

        # Set the mutation variance
        self.mutation_variance = aMutationVariance;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    def getMutationVariance(self):
        return self.mutation_variance;

    def setMutationVariance(self, aMutationVariance):
        self.mutation_variance = aMutationVariance;

    def apply(self, anEA):

        self.use_count += 1;

        # Select the parents from the population
        parent_index = anEA.selection_operator.select(anEA.current_solution_set)

        # Copy the parent into a child
        child = copy.deepcopy(anEA.current_solution_set[parent_index]);

        # Mutate the child and return it
        return self.mutate(child);

    def mutate(self, anIndividual):

        for i in range(len(anIndividual.genes)):
            anIndividual.genes[i] = self.system_random.gauss(anIndividual.genes[i], self.mutation_variance);
            anIndividual.genes[i] = max(anIndividual.boundary_set[i][0], anIndividual.genes[i]);
            anIndividual.genes[i] = min(anIndividual.boundary_set[i][1], anIndividual.genes[i]);

        anIndividual.computeObjectiveFunction();

        return anIndividual;
