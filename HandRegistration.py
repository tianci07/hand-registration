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
# number_of_iterations = 15;
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
#     pred_image, df2 = display_metrics(method, optimiser.best_solution, target, computing_time);
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
#     pred_image, df2 = display_metrics(method, optimiser.best_solution.parameter_set, target, computing_time);
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
#     pred_image, df2 = display_metrics(method, optimiser.best_solution.genes, target, computing_time);
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
        # 'SLSQP',
        # 'TNC',
        'Nelder-Mead',
        # 'Powell',
        'CG',
        # 'BFGS',
        'L-BFGS-B',
        # 'COBYLA'
    ];

df = pd.DataFrame();
df1 = pd.DataFrame();
df2 = pd.DataFrame();
df3 = pd.DataFrame();
df4 = pd.DataFrame();
df5 = pd.DataFrame();
df6 = pd.DataFrame();

for run in range(15):

    initial_guess = [];
    test_problem = HandFunction(target_image);

    for i in range(test_problem.number_of_dimensions):
        initial_guess.append(ObjectiveFunction.system_random.uniform(
                                                            test_problem.boundary_set[i][0],
                                                            test_problem.boundary_set[i][1]
                                                            )
                                                            );
    # Nelder-mean, CG, L-BFGS-B
    for method in methods:

        start = time.time();
        result = optimize.minimize(test_problem.minimisationFunction,
            initial_guess,
            method=method,
            bounds=test_problem.boundaries);

        end = time.time();
        computing_time = end-start;

        if method == 'Nelder-Mead':

            pred_image, df_1 = display_metrics(method, result.x, target, computing_time);
            df1 = df1.append(df_1, ignore_index=True);
            print(result);

            plt.imsave("./posterior-anterior/MAE/%s-%d" % (method, (run+1)), pred_image, cmap='Greys_r');
            gvxr.saveLastXRayImage("./posterior-anterior/MAE/%s_%d.mha" % (method, (run+1)));

        elif method == 'CG':

            pred_image, df_2 = display_metrics(method, result.x, target, computing_time);
            df2 = df2.append(df_2, ignore_index=True);
            print(result);

            plt.imsave("./posterior-anterior/MAE/%s-%d" % (method, (run+1)), pred_image, cmap='Greys_r');
            gvxr.saveLastXRayImage("./posterior-anterior/MAE/%s_%d.mha" % (method, (run+1)));

        elif method == 'L-BFGS-B':

            pred_image, df_3 = display_metrics(method, result.x, target, computing_time);
            df3 = df3.append(df_3, ignore_index=True);
            print(result);

            plt.imsave("./posterior-anterior/MAE/%s-%d" % (method, (run+1)), pred_image, cmap='Greys_r');
            gvxr.saveLastXRayImage("./posterior-anterior/MAE/%s_%d.mha" % (method, (run+1)));
    # PRS
    method = 'PRS';
    start = time.time();
    objective_function = HandFunction(target_image);
    optimiser = PRS.PureRandomSearch(len(objective_function.boundaries),
                                        objective_function.boundaries,
                                        objective_function,
                                        0,
                                        1000);
    end=time.time();
    computing_time = end-start;
    pred_image, df_4 = display_metrics(method, optimiser.best_solution, target, computing_time);
    df4 = df4.append(df_4, ignore_index=True);

    plt.imsave("./posterior-anterior/MAE/%s-%d.png" % (method, (run+1)), pred_image, cmap='Greys_r');
    gvxr.saveLastXRayImage("./posterior-anterior/MAE/%s-%d.mha" % (method, (run+1)));

    # SA
    method = 'SA';
    start = time.time();
    objective_function = HandFunction(target_image);
    optimiser = SimulatedAnnealing(objective_function, 20000, 0.01, initial_guess=initial_guess);
    optimiser.run();

    end=time.time();
    computing_time = end-start;
    pred_image, df_5 = display_metrics(method, optimiser.best_solution.parameter_set, target, computing_time);
    df5 = df5.append(df_5, ignore_index=True);
    print(df);
    plt.imsave("./posterior-anterior/MAE/%s-%d.png" % (method, (run+1)), pred_image, cmap='Greys_r');
    gvxr.saveLastXRayImage("./posterior-anterior/MAE/%s-%d.mha" % (method, (run+1)));

    # EA
    method = 'EA';
    g_number_of_individuals = 50;
    g_iterations            = 10;

    g_max_mutation_sigma = 0.1;
    g_min_mutation_sigma = 0.01;

    start = time.time();
    objective_function = HandFunction(target_image);
    optimiser = EvolutionaryAlgorithm(objective_function, g_number_of_individuals, initial_guess=initial_guess);
    optimiser.setSelectionOperator(RankSelection());

    # Create the genetic operators
    elitism = ElitismOperator(0.1);
    new_blood = NewBloodOperator(0.1);
    gaussian_mutation = GaussianMutationOperator(0.1, 0.2);
    blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);

    # Add the genetic operators to the EA
    optimiser.addGeneticOperator(new_blood);
    optimiser.addGeneticOperator(gaussian_mutation);
    optimiser.addGeneticOperator(blend_cross_over);
    optimiser.addGeneticOperator(elitism);

    for i in range(g_iterations):
        # Compute the value of the mutation variance
        sigma = g_min_mutation_sigma + (g_iterations - 1 - i) / (g_iterations - 1) * (g_max_mutation_sigma - g_min_mutation_sigma);

        # Set the mutation variance
        gaussian_mutation.setMutationVariance(sigma);

        # Run the optimisation loop
        optimiser.runIteration();

        # Print the current state in the console
        optimiser.printCurrentStates(i + 1);

    end=time.time();
    computing_time = end-start;

    pred_image, df_6 = display_metrics(method, optimiser.best_solution.genes, target, computing_time);
    df6 = df6.append(df_6, ignore_index=True);

    plt.imsave("./posterior-anterior/MAE/%s-%d.png" % (method, (run+1)), pred_image, cmap='Greys_r');
    gvxr.saveLastXRayImage("./posterior-anterior/MAE/%s-%d.mha" % (method, (run+1)));

print("\nNelder-Mead: Mean(MAE):%8f, STD(MAE):%8f, Mean(ZNCC):%8f, STD(ZNCC):%8f \n" % (df1["MAE"].mean(), df1["MAE"].std(), df1["ZNCC"].mean(), df1["ZNCC"].std()));
print("\nNelder-Mead: Mean(Entropy):%8f, STD(Entropy):%8f, Mean(Time):%8f, STD(Time):%8f \n" % (df1["Entropy"].mean(), df1["Entropy"].std(), df1["Time"].mean(), df1["Time"].std()));

print("\nCG: Mean(MAE):%8f, STD(MAE):%8f, Mean(ZNCC):%8f, STD(ZNCC):%8f \n" % (df2["MAE"].mean(), df2["MAE"].std(), df2["ZNCC"].mean(), df2["ZNCC"].std()));
print("\nCG: Mean(Entropy):%8f, STD(Entropy):%8f, Mean(Time):%8f, STD(Time):%8f \n" % (df2["Entropy"].mean(), df2["Entropy"].std(), df2["Time"].mean(), df2["Time"].std()));

print("\nL-BFGS-B: Mean(MAE):%8f, STD(MAE):%8f, Mean(ZNCC):%8f, STD(ZNCC):%8f \n" % (df3["MAE"].mean(), df3["MAE"].std(), df3["ZNCC"].mean(), df3["ZNCC"].std()));
print("\nL-BFGS-B: Mean(Entropy):%8f, STD(Entropy):%8f, Mean(Time):%8f, STD(Time):%8f \n" % (df3["Entropy"].mean(), df3["Entropy"].std(), df3["Time"].mean(), df3["Time"].std()));

print("\nPRS: Mean(MAE):%8f, STD(MAE):%8f, Mean(ZNCC):%8f, STD(ZNCC):%8f \n" % (df4["MAE"].mean(), df4["MAE"].std(), df4["ZNCC"].mean(), df4["ZNCC"].std()));
print("\nPRS: Mean(Entropy):%8f, STD(Entropy):%8f, Mean(Time):%8f, STD(Time):%8f \n" % (df4["Entropy"].mean(), df4["Entropy"].std(), df4["Time"].mean(), df4["Time"].std()));

print("\nSA: Mean(MAE):%8f, STD(MAE):%8f, Mean(ZNCC):%8f, STD(ZNCC):%8f \n" % (df5["MAE"].mean(), df5["MAE"].std(), df5["ZNCC"].mean(), df5["ZNCC"].std()));
print("\nSA: Mean(Entropy):%8f, STD(Entropy):%8f, Mean(Time):%8f, STD(Time):%8f \n" % (df5["Entropy"].mean(), df5["Entropy"].std(), df5["Time"].mean(), df5["Time"].std()));

print("\nEA: Mean(MAE):%8f, STD(MAE):%8f, Mean(ZNCC):%8f, STD(ZNCC):%8f \n" % (df6["MAE"].mean(), df6["MAE"].std(), df6["ZNCC"].mean(), df6["ZNCC"].std()));
print("\nEA: Mean(Entropy):%8f, STD(Entropy):%8f, Mean(Time):%8f, STD(Time):%8f \n" % (df6["Entropy"].mean(), df6["Entropy"].std(), df6["Time"].mean(), df6["Time"].std()));

df = df.append(df1, ignore_index=True);
df = df.append(df2, ignore_index=True);
df = df.append(df3, ignore_index=True);
df = df.append(df4, ignore_index=True);
df = df.append(df5, ignore_index=True);
df = df.append(df6, ignore_index=True);

df.to_csv("./posterior-anterior/MAE/results.csv");
