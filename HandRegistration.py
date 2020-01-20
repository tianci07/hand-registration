import numpy as np
import cv2
from Metrics import *
from RotateBones import *
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
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def plot(target, prediction, aParameter, anIteration):

    iter = anIteration;
    target_image = target;
    temp_image = prediction;
    param = aParameter;

    plt.figure(figsize=(20, 10));
    gs = gridspec.GridSpec(2, 4);

    plt.subplot(gs[0, 0]);
    plt.title("Target");
    plt.imshow(target_image, cmap=plt.cm.Greys_r);

    plt.axis('off');

    MAE = mean_absolute_error(target_image, temp_image);
    ZNCC = zero_mean_normalised_cross_correlation(target_image, temp_image);

    plt.imsave("./00382-s1-neg2/MAE-adaptive/best-individuals-3/ind-%s-%d" % (param, i+1), temp_image, cmap='Greys_r');

    plt.subplot(gs[0, 1]);
    plt.title("Current best");
    img2 = plt.imshow(temp_image, cmap=plt.cm.Greys_r);

    plt.axis('off');

    row = [[temp_best_solution, MAE, ZNCC]];
    df2 = pd.DataFrame(row, columns=['Parameters','MAE', 'ZNCC']);
    df_2 = df_2.append(df2, ignore_index=True);

    error_map_EA = np.log(abs(target_image-temp_image));

    plt.subplot(gs[0, 2]);
    plt.title("Error map");
    img3 = plt.imshow(error_map_EA, cmap=plt.cm.Greys_r);

    plt.axis('off');
    # plt.imsave("./00382-s1-neg2/MAE-adaptive/best-individuals-3/error-%s-%d.png" % (param, i+1), error_map_EA, cmap='Greys_r');

    correlation_map_EA = np.log(target_image*temp_image);

    plt.subplot(gs[0, 3]);
    plt.title("Correlation map");
    img4 = plt.imshow(correlation_map_EA, cmap=plt.cm.Greys_r);

    plt.axis('off');
    # plt.imsave("./00382-s1-neg2/MAE-adaptive/best-individuals-3/correlation-%s-%d.png" % (param, i+1) , correlation_map_EA, cmap='Greys_r');

    temp_df = df_2.T;
    plt.subplot(gs[1, 0:2]);
    plt.title("MAE: %.4f" % MAE);
    plt.plot(temp_df.loc['MAE'], 'b-', label='MAE');

    plt.subplot(gs[1, 2:4]);
    plt.title("ZNCC:%.4f" % ZNCC);
    plt.plot(temp_df.loc['ZNCC'], 'g-', label='ZNCC');

    fname = "./00382-s1-neg2/MAE-adaptive/best-individuals-3/ind-%d.png" % iter;
    plt.suptitle("Optimising %s " % param, fontsize='x-large');
    plt.tight_layout();
    plt.savefig(fname);

    plt.close('all');


# target_image, target = createTarget();
setXRayEnvironment();

target_image = cv2.imread("./00382-s1-neg2.png", 0);
target_image = preprocessing.scale(target_image);

# ------------------------------------------------------------------------------
# Optimising 7 sections and updating previous solution if better solution found
# ------------------------------------------------------------------------------

df = pd.DataFrame();
# df_2 = pd.DataFrame();
run = 0;

# Repeat optimisation for 15 times to gather statistically meanful results
for run in range(15):
    params = ['Distance', 'Root', 'Thumb', 'Index', 'Middle', 'Ring', 'Little'];
    number_of_params = 24;
    number_of_angles = 22;
    number_of_distances = number_of_params-number_of_angles;
    d = 2;
    overall_computing_time = 0;
    j=1;
    temp_best_solution=[];

    # Run 7 optimisations
    for param in params:

        method = 'EA';
        best_finger = [];
        if param != 'Distance':

            if param == 'Root' or param == 'Thumb':
                d += 3;
            else:
                d += 4;

        initial_guess = copy.deepcopy(temp_best_solution);
        while len(initial_guess) < d:
            initial_guess.append(0.);

        g_number_of_individuals = 20+d*10;
        g_iterations            = 20+d*10;

        g_max_mutation_sigma = 0.1;
        g_min_mutation_sigma = 0.01;

        matrix_set = getLocalTransformationMatrixSet();

        start = time.time();

        objective_function = HandFunction(target_image, d);
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

            # temp_best_solution = copy.deepcopy(optimiser.best_solution.parameter_set);
            # SOD = temp_best_solution[0]*temp_best_solution[1];
            # SDD = temp_best_solution[1];
            # setXRayParameters(SOD, SDD);
            #
            # temp_objective = -optimiser.best_solution.objective;
            # best_angle = [];
            # if param != 'Distance':
            #
            #     for a in range(d-number_of_distances):
            #         best_angle.append(temp_best_solution[a+number_of_distances]);
            #
            # while len(best_angle) < number_of_angles:
            #     best_angle.append(0.)
            #
            # temp_image = bone_rotation(best_angle, 'All');
            #
            # plot(target_image, temp_image, param, j);
            # j+=1;

        end=time.time();
        computing_time = end-start;

        overall_computing_time += computing_time;
        best_solution = copy.deepcopy(optimiser.best_solution.parameter_set);

    SOD = best_solution[0]*best_solution[1];
    SDD = best_solution[1];
    setXRayParameters(SOD, SDD);
    for a in range(number_of_angles):
        best_angle.append(best_solution[a+number_of_distances]);

    pred_image = bone_rotation(best_angle, 'All');
    plt.imsave("./00382-s1-neg2/MAE-adaptive/EA-%d" % (run+1), pred_image, cmap='Greys_r');

    MAE = -optimiser.best_solution.objective;
    ZNCC = zero_mean_normalised_cross_correlation(target_image, pred_image);
    row = [[best_solution, MAE, ZNCC, overall_computing_time]];
    df2 = pd.DataFrame(row, columns=['Parameters','MAE', 'ZNCC', 'Time']);
    df = df.append(df2, ignore_index=True);

    error_map_EA = np.log(abs(target_image-pred_image));
    plt.imsave("./00382-s1-neg2/MAE-adaptive/error-map-EA-%d.png" % (run+1), error_map_EA, cmap='Greys_r');

    correlation_map_EA = np.log(target_image*pred_image);
    plt.imsave("./00382-s1-neg2/MAE-adaptive/correlation-map-EA-%d.png" % (run+1) , correlation_map_EA, cmap='Greys_r');

df.to_csv("./00382-s1-neg2/MAE-adaptive/results.csv");
# df_2.to_csv("./00382-s1-neg2/MAE-adaptive/best-individuals-3/results.csv");
