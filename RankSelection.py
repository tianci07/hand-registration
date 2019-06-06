import random
import numpy as np

from SelectionOperator import *

class RankSelection(SelectionOperator):

    def __init__(self):
        super().__init__("Rank selection");
        self.rank_set = [];
        self.sum_rank = 0;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    def preProcess(self, anIndividualSet):
        self.rank_set = [];

        # Sort index of individuals based on their fitness
        fitness_set = [];
        for individual in anIndividualSet:
            fitness_set.append(individual.fitness)

        # Sort the array
        self.rank_set = np.argsort((fitness_set))

        # Compute rank sumation
        self.sum_rank = 0;
        for rank in self.rank_set:
            self.sum_rank += rank;

    def __select__(self, anIndividualSet, aFlag): # aFlag == True for selecting good individuals,
                                                  # aFlag == False for selecting bad individuals,

        # Random number between(0 - self.sum_rank)
        random_number = self.system_random.uniform(0, self.sum_rank)

        # Select the individual depending on the probability
        accumulator = 0;
        for index, rank in np.ndenumerate(self.rank_set):
            accumulator += rank;
            if accumulator >= random_number:
                return index[0]
