import random
import copy
from RotateBones import bone_rotation

class PureRandomSearch:
    '''
    @parameters:
    aNumberOfDimensions: Number of variables.
    aBoundarySet: bounds for each parameters.
    anObjectiveFunction: Objective function used in optimisation
    aFlag: 0 for errors (minimising erros), 1 for SSIM and ZNCC (maximising index score).
    iterations: number of iterations for optimisation.
    '''
    def __init__(self,
                 aNumberOfDimensions,
                 aBoundarySet,
                 anObjectiveFunction,
                 aFlag,
                 iterations=500
                 ):

        self.number_of_dimensions = aNumberOfDimensions;
        self.boundaries = copy.deepcopy(aBoundarySet);
        self.objective_function = anObjectiveFunction;
        self.niter = iterations;
        self.flag = aFlag;
        self.number_of_distances = 2;
        self.number_of_angles = 22;

        if self.flag == 0:
            self.fvalue = 0;
            print("Minimising...");
        elif self.flag == 1:
            self.fvalue = 1;
            print("Maximising...");
        else:
            raise NotImplementedError("0 for minimisation and 1 for maximisation");

        # Initialise temporary and final solution sets
        self.best_solution = [];
        self.current_solution = [];
        self.current_objective_value = 0;
        self.best_objective_value = 1000;
        for i in range(self.number_of_dimensions):
            self.current_solution.append(0);
            self.best_solution.append(0);

        self.optimise();

    def optimise(self):

        for i in range(self.niter):

            self.current_solution[0] = random.uniform(self.boundaries[0][0], self.boundaries[0][1]);
            self.current_solution[1] = random.randint(self.boundaries[1][0], self.boundaries[1][1]);

            for j in range(self.number_of_angles):

                self.current_solution[j+self.number_of_distances] = random.randint(
                                                                        self.boundaries[j+self.number_of_distances][0],
                                                                        self.boundaries[j+self.number_of_distances][1]);

            self.current_objective_value = self.objective_function.objectiveFunction(self.current_solution);
            # print(self.current_solution);
            print(self.current_objective_value);
            print(self.best_objective_value);

            if self.fvalue == 1:

                if self.best_objective_value < self.current_objective_value:

                    self.best_objective_value = self.current_objective_value;

                    for m in range(self.number_of_dimensions):
                        self.best_solution[m] = self.current_solution[m];
                        print(self.best_solution);

            elif self.fvalue == 0:

                if self.current_objective_value < self.best_objective_value:

                    self.best_objective_value = self.current_objective_value;

                    for m in range(self.number_of_dimensions):
                        self.best_solution[m] = self.current_solution[m];
