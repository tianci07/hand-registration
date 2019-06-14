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

# Searching hyper-parameters and select best hyper-parameters based on metrics.
# Selecting number of iterations iteration for Pure Random search.
hyper_parameter_number_of_iterations = np.arange(100, 1100, 100); # run 10 searches
df = pd.DataFrame();

for i in range(len(hyper_parameter_number_of_iterations)):

    start = time.time();
    objective_function = HandFunction(target_image);
    optimiser = PRS.PureRandomSearch(len(objective_function.boundaries),
                                        objective_function.boundaries,
                                        objective_function,
                                        0,
                                        hyper_parameter_number_of_iterations[i]);
    end=time.time();
    computing_time = end-start;
    pred_image, df2 = display_metrics(optimiser.best_solution,
                                    target,
                                    computing_time);
    df = df.append(df2, ignore_index=True);

    plt.imsave("./hyper-parameter-selection/RMSE/PRS-%d.png" % (i+1), pred_image, cmap='Greys_r');
    gvxr.saveLastXRayImage("./hyper-parameter-selection/RMSE/PRS-%d.mha" % (i+1));

# Selecting temperature and cooling rate for Simulated Annealing.
# run 150 searches
hyper_parameter_temperature = np.linspace(5000, 15000, 1000);
hyper_parameter_cooling_rate = np.linspace(0.01, 0.1, 10);

for i in range(len(hyper_parameter_temperature)):
    for j in range(len(hyper_parameter_cooling_rate)):

        hyper_parameter_set = [];
        hyper_parameter_set.append(hyper_parameter_temperature[i]);
        hyper_parameter_set.append(hyper_parameter_cooling_rate[j]);

        start = time.time();
        objective_function = HandFunction(target_image);
        optimiser = SimulatedAnnealing(objective_function, hyper_parameter_set[0], hyper_parameter_set[1]);
        optimiser.run();

        end=time.time();
        computing_time = end-start;
        pred_image, df2 = display_metrics(optimiser.best_solution.parameter_set,
                                        target,
                                        computing_time);
        df = df.append(df2, ignore_index=True);
        print(df);
        plt.imsave("./hyper-parameter-selection/RMSE/SA-%d.png" % (i+1), pred_image, cmap='Greys_r');
        gvxr.saveLastXRayImage("./hyper-parameter-selection/RMSE/SA-%d.mha" % (i+1));


# Selecting number of individuals and generations for Evolutionary Algorithm.
# Run 100 searches
hyper_parameter_g_number_of_individuals = np.linspace(50, 500, 10);
hyper_parameter_g_iterations = np.linspace(40, 400, 10);

g_max_mutation_sigma = 0.1;
g_min_mutation_sigma = 0.01;


for i in range(len(hyper_parameter_g_number_of_individuals)):

    for j in range(len(hyper_parameter_g_iterations)):

        hyper_parameter_set = [];
        hyper_parameter_set.append(hyper_parameter_g_number_of_individuals[i]);
        hyper_parameter_set.append(hyper_parameter_g_iterations[j]);

        start = time.time();
        objective_function = HandFunction(target_image);
        optimiser = EvolutionaryAlgorithm(objective_function, g_number_of_individuals);
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

        for i in range(hyper_parameter_g_iterations):
            # Compute the value of the mutation variance
            sigma = g_min_mutation_sigma + (hyper_parameter_g_iterations - 1 - i) / (hyper_parameter_g_iterations - 1) * (g_max_mutation_sigma - g_min_mutation_sigma);

            # Set the mutation variance
            gaussian_mutation.setMutationVariance(sigma);

            # Run the optimisation loop
            optimiser.runIteration();

            # Print the current state in the console
            optimiser.printCurrentStates(i + 1);

    end=time.time();
    computing_time = end-start;

    pred_image, df2 = display_metrics(optimiser.best_solution.genes,
                                    target,
                                    computing_time);
    df = df.append(df2, ignore_index=True);

print("Saving selection results to csv files ...");
df.to_csv("./hyper-parameter-selection/RMSE/Selection.csv");
