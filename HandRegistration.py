import numpy as np
from Metrics import *
from RotateBones import bone_rotation
from DisplayMetrics import *
import PureRandomSearch as PRS
import gvxrPython3 as gvxr
import random
import matplotlib.pyplot as plt
from SimulatedAnnealing import *
from EvolutionaryAlgorithm import *
import time
import pandas as pd
from scipy import optimize

from HandFunction import *
# Selection operators
from RankSelection            import *

# Genetic operators
from ElitismOperator          import *
from BlendCrossoverOperator   import *
from GaussianMutationOperator import *
from NewBloodOperator         import *

# Optimise SDD and SOD
global target_image;

target_image, target = createTarget();
# plt.imsave("./posterior-anterior/MAE/target.png", target_image, cmap='Greys_r');
# gvxr.saveLastXRayImage("./posterior-anterior/MAE/target.mha");
# from PSO import PSO

# Optimising with Pure Random Search
number_of_iterations = 15;
# df = pd.DataFrame();
#
# for i in range(number_of_iterations):
#
#     start = time.time();
#     objective_function = HandFunction(target_image);
#     optimiser = PRS.PureRandomSearch(len(objective_function.boundaries),
#                                         objective_function.boundaries,
#                                         objective_function,
#                                         0,
#                                         1000);
#     end=time.time();
#     computing_time = end-start;
#     pred_image, df2 = display_metrics(optimiser.best_solution, target, computing_time);
#     df = df.append(df2, ignore_index=True);
#     print(df);
#     plt.imsave("./posterior-anterior/MAE/prediction-PRS-%d.png" % (i+1), pred_image, cmap='Greys_r');
#     gvxr.saveLastXRayImage("./posterior-anterior/MAE/prediction-PRS-%d.mha" % (i+1));
#
# print('Saving to csv file...');
# df.to_csv('./posterior-anterior/MAE/results_PRS.csv');
#
# # Optimising using Simulated Annealing
# df = pd.DataFrame();
#
# for i in range(number_of_iterations):
#
#     start = time.time();
#     objective_function = HandFunction(target_image);
#     optimiser = SimulatedAnnealing(objective_function, 20000, 0.01);
#     optimiser.run();
#
#     end=time.time();
#     computing_time = end-start;
#     pred_image, df2 = display_metrics(optimiser.best_solution.parameter_set, target, computing_time);
#     df = df.append(df2, ignore_index=True);
#     print(df);
#     plt.imsave("./posterior-anterior/MAE/prediction-SA-%d.png" % (i+1), pred_image, cmap='Greys_r');
#     gvxr.saveLastXRayImage("./posterior-anterior/MAE/prediction-SA-%d.mha" % (i+1));
#
# print('Saving to csv file...');
# df.to_csv('./posterior-anterior/MAE/results_SA.csv');

# Optimising with Evolutionary Algorithm
# g_number_of_individuals = 50;
# g_iterations            = 10;
#
# g_max_mutation_sigma = 0.1;
# g_min_mutation_sigma = 0.01;
#
# df = pd.DataFrame();
#
# for j in range(number_of_iterations):
#
#     start = time.time();
#     objective_function = HandFunction(target_image);
#     optimiser = EvolutionaryAlgorithm(objective_function, g_number_of_individuals);
#     optimiser.setSelectionOperator(RankSelection());
#
#     # Create the genetic operators
#     elitism = ElitismOperator(0.1);
#     new_blood = NewBloodOperator(0.1);
#     gaussian_mutation = GaussianMutationOperator(0.1, 0.2);
#     blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);
#
#     # Add the genetic operators to the EA
#     optimiser.addGeneticOperator(new_blood);
#     optimiser.addGeneticOperator(gaussian_mutation);
#     optimiser.addGeneticOperator(blend_cross_over);
#     optimiser.addGeneticOperator(elitism);
#
#     for i in range(g_iterations):
#         # Compute the value of the mutation variance
#         sigma = g_min_mutation_sigma + (g_iterations - 1 - i) / (g_iterations - 1) * (g_max_mutation_sigma - g_min_mutation_sigma);
#
#         # Set the mutation variance
#         gaussian_mutation.setMutationVariance(sigma);
#
#         # Run the optimisation loop
#         optimiser.runIteration();
#
#         # Print the current state in the console
#         optimiser.printCurrentStates(i + 1);
#
#     end=time.time();
#     computing_time = end-start;
#
#     pred_image, df2 = display_metrics(optimiser.best_solution.genes, target, computing_time);
#     df = df.append(df2, ignore_index=True);
#     plt.imsave("./posterior-anterior/MAE/prediction-EA-%d.png" % (j+1), pred_image, cmap='Greys_r');
#     gvxr.saveLastXRayImage("./posterior-anterior/MAE/prediction-EA-%d.mha" % (j+1));
#
# # Save the results to csv files
# print('Saving to csv file...');
# df.to_csv('./posterior-anterior/MAE/results_EA.csv');

# # Create the objective function
# objective_function = SOD_SDD_ObjectiveFunction(target_image);
#
# # Create a PSO
# g_number_of_particle   = 10;
# g_iterations           = 10;
# optimiser = PSO(objective_function, g_number_of_particle);
#
# # Print the current state in the console
# optimiser.printCurrentStates(0);
#
# x_ray_image = gvxr.computeXRayImage();
# pred_image = np.array(x_ray_image);
#
# plt.imsave("./posterior-anterior/MAE/prediction-PSO-%d.png" % 0, pred_image, cmap='Greys_r');
# gvxr.saveLastXRayImage("./posterior-anterior/MAE/prediction-PSO-%d.mha" % 0);
#
# # Optimisation
# for i in range(g_iterations):
#     # Run the optimisation loop
#     optimiser.runIteration();
#
#     # Print the current state in the console
#     optimiser.printCurrentStates(i + 1);
#
#     objective_function.objectiveFunctionSOD_SDD(optimiser.best_solution.position)
#
#     x_ray_image = gvxr.computeXRayImage();
#     pred_image = np.array(x_ray_image);
#
#     plt.imsave("./posterior-anterior/MAE/prediction-PSO-%d.png" % (i + 1), pred_image, cmap='Greys_r');
#
#     gvxr.saveLastXRayImage("./posterior-anterior/MAE/prediction-PSO-%d.mha" % (i + 1));
#
# print("Solution:\t", optimiser.best_solution);

# Bounds on variables for L-BFGS-B, TNC, SLSQP and trust-constr methods.
methods = [
        'SLSQP',
        'TNC',
        # 'Nelder-Mead',
        # 'Powell',
        # 'CG',
        # 'BFGS',
        'L-BFGS-B',
        # 'COBYLA'
    ];

df = pd.DataFrame();

for run in range(2):

    initial_guess = [];
    test_problem = HandFunction(target_image);

    for i in range(test_problem.number_of_dimensions):
        initial_guess.append(ObjectiveFunction.system_random.uniform(
                                                            test_problem.boundary_set[i][0],
                                                            test_problem.boundary_set[i][1]
                                                            )
                                                            );

    for method in methods:

        start = time.time();
        result = optimize.minimize(test_problem.minimisationFunction,
            initial_guess,
            method=method,
            bounds=test_problem.boundaries);

        end=time.time();

        computing_time = end-start;
        pred_image, df2 = display_metrics(result.x, target, computing_time);
        df = df.append(df2, ignore_index=True);
        print(result);

        plt.imsave("./posterior-anterior/MAE/%s_%d" % (method, (run+1)), pred_image, cmap='Greys_r');
        gvxr.saveLastXRayImage("./posterior-anterior/MAE/%s_%d.mha" % (method, (run+1)));

df.to_csv("./posterior-anterior/MAE/results_scipy_optimize.csv");
