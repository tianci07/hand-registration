import copy
import random
import math

class ObjectiveFunction:
    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    def __init__(self,
                 aNumberOfDimensions,
                 aBoundarySet,
                 anObjectiveFunction,
                 aFlag = 0):

        # aFlag is 1 for Minimisation
        # aFlag is 2 for Maximisation
        self.boundary_set = copy.deepcopy(aBoundarySet);
        self.number_of_dimensions = aNumberOfDimensions;
        self.objective_function = anObjectiveFunction;
        self.number_of_evaluation = 0;
        self.flag = aFlag;
        self.global_optimum = None;
        self.verbose = False;   # Use for debugging

    def initialGuess(self):
        if self.number_of_dimensions == 1:
            return ObjectiveFunction.system_random.uniform(self.boundary_set[0][0], self.boundary_set[0][1]);
        else:
            guess = [];
            for i in range(self.number_of_dimensions):
                guess.append(ObjectiveFunction.system_random.uniform(self.boundary_set[i][0], self.boundary_set[i][1]))
            return guess;

    def minimisationFunction(self, aParameterSet):
        return self.evaluate(aParameterSet, 1)

    def maximisationFunction(self, aParameterSet):
        return self.evaluate(aParameterSet, 2)

    def evaluate(self, aParameterSet, aFlag):
        self.number_of_evaluation += 1;

        objective_value = self.objective_function(aParameterSet);
        if aFlag != self.flag:
            objective_value *= -1;

        return objective_value;

    def getDistanceToGlobalOptimum(self, aParameterSet):
        if self.global_optimum != None:
            distance = 0.0;
            for t, r in zip(aParameterSet, self.global_optimum):
                distance += math.pow(t - r, 2);
            return math.sqrt(distance);
