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
import argparse
import os

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

    # plt.imsave("./00382-s1-neg2/MAE-restart/ind-%s-%d" % (param, i+1), temp_image, cmap='Greys_r');

    plt.subplot(gs[0, 1]);
    plt.title("Current best");
    img2 = plt.imshow(temp_image, cmap=plt.cm.Greys_r);

    plt.axis('off');

    error_map = np.log(abs(target_image-temp_image));

    plt.subplot(gs[0, 2]);
    plt.title("Error map");
    img3 = plt.imshow(error_map, cmap=plt.cm.Greys_r);

    plt.axis('off');
    # plt.imsave("./00382-s1-neg2/MAE-restart/error-%s-%d.png" % (param, i+1), error_map, cmap='Greys_r');

    correlation_map = np.log(target_image*temp_image);

    plt.subplot(gs[0, 3]);
    plt.title("Correlation map");
    img4 = plt.imshow(correlation_map, cmap=plt.cm.Greys_r);

    plt.axis('off');
    # plt.imsave("./00382-s1-neg2/MAE-restart/correlation-%s-%d.png" % (param, i+1) , correlation_map, cmap='Greys_r');

    temp_df = df_2.T;
    plt.subplot(gs[1, 0:2]);
    plt.title("MAE: %.4f" % MAE);
    plt.plot(temp_df.loc['MAE'], 'b-', label='MAE');

    plt.subplot(gs[1, 2:4]);
    plt.title("ZNCC:%.4f" % ZNCC);
    plt.plot(temp_df.loc['ZNCC'], 'g-', label='ZNCC');

    # fname = "./00382-s1-neg2/MAE-restart/ind-2/ind-%d.png" % iter;
    fname = ind_folder + "/ind-%d.png" % iter;
    plt.suptitle("Optimising %s " % param, fontsize='x-large');
    plt.tight_layout();
    plt.savefig(fname);

    plt.close('all');

parser = argparse.ArgumentParser();

parser.add_argument("--target", help="Input file name", required=True);
parser.add_argument("--output", help="Path to output folder", required=True);
parser.add_argument("--restart", help="Restart times", type=int, required=True);

parser.add_argument("--initial_guess", help="Initial guess", nargs='*');
parser.add_argument("--plot_metrics", help="Plot metrics during optimisation");

parser.add_argument("--parameters", help="Number of parameters to optimise", type=int, required=True);
parser.add_argument("--angles", help="Number of angles for rotation", type=int, required=True);

parser.add_argument("--max_mutation_sigma", help="For mutation variance", type=float, required=True);
parser.add_argument("--min_mutation_sigma", help="For mutation variance", type=float, required=True);
parser.add_argument("--individuals", help="Number of individuals", type=int, required=True);
parser.add_argument("--generations", help="Number of generations", type=int, required=True);
parser.add_argument("--elitism", help="Operator's probability", type=float, required=True);
parser.add_argument("--new_blood", help="Operator's probability", type=float, required=True);
parser.add_argument("--gaussian_mutation", help="Operator's probability", type=float, nargs=2, required=True);
parser.add_argument("--blend_cross_over", help="Operator's probability", type=float, required=True);

args = parser.parse_args();

setXRayEnvironment();

# target_image = cv2.imread("./00382-s1-neg2.png", 0);
target_image = cv2.imread(args.target, 0);
target_image = preprocessing.scale(target_image);

if not os.path.exists(args.output):
    os.mkdir(args.output);

ind_folder = args.output + "/ind-%d" % args.restart;
if not os.path.exists(ind_folder):
    os.mkdir(ind_folder);

# Optimising 7 sections and updating previous solution if better solution found
df = pd.DataFrame();
df_2 = pd.DataFrame();

params = ['Distance', 'Root', 'Thumb', 'Index', 'Middle', 'Ring', 'Little'];
number_of_params = args.parameters;
number_of_angles = args.angles;
number_of_distances = number_of_params-number_of_angles;
d = 2;
overall_computing_time = 0;
j=1;
temp_best_solution=[];
start = time.time();

# Optimising whole hand
if args.initial_guess:

    initial_guess =[];
    for ini in range(len(args.initial_guess)):
        initial_guess.append(float(args.initial_guess[ini]));
    param = 'All';

    g_number_of_individuals = args.individuals;
    g_iterations            = args.generations;

    g_max_mutation_sigma = args.max_mutation_sigma;
    g_min_mutation_sigma = args.min_mutation_sigma;

    start = time.time();

    objective_function = HandFunction(target_image, number_of_params);
    optimiser = EvolutionaryAlgorithm(objective_function, g_number_of_individuals, initial_guess=initial_guess);
    optimiser.setSelectionOperator(RankSelection());

    # Create the genetic operators
    elitism = ElitismOperator(args.elitism);
    new_blood = NewBloodOperator(args.new_blood);
    gaussian_mutation = GaussianMutationOperator(args.gaussian_mutation[0], args.gaussian_mutation[1]);
    blend_cross_over = BlendCrossoverOperator(args.blend_cross_over, gaussian_mutation);

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

        if args.plot_metrics:

            temp_best_solution = copy.deepcopy(optimiser.best_solution.parameter_set);

            SOD = temp_best_solution[0]*temp_best_solution[1];
            SDD = temp_best_solution[1];
            setXRayParameters(SOD, SDD);

            temp_objective = -optimiser.best_solution.objective;
            best_angle = [];

            for a in range(number_of_angles):
                best_angle.append(temp_best_solution[a+number_of_distances]);

            temp_image = bone_rotation(best_angle, 'All');

            MAE = temp_objective;
            ZNCC = zero_mean_normalised_cross_correlation(target_image, temp_image);
            row = [[temp_best_solution, MAE, ZNCC]];
            df2 = pd.DataFrame(row, columns=['Parameters','MAE', 'ZNCC']);
            df_2 = df_2.append(df2, ignore_index=True);

            plot(target_image, temp_image, param, j);
            j+=1;

else:
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

        g_number_of_individuals = args.individuals+d*10;
        g_iterations            = args.generations+d*10;

        g_max_mutation_sigma = args.max_mutation_sigma;
        g_min_mutation_sigma = args.min_mutation_sigma;

        objective_function = HandFunction(target_image, d);
        optimiser = EvolutionaryAlgorithm(objective_function, g_number_of_individuals, initial_guess=initial_guess);
        optimiser.setSelectionOperator(RankSelection());

        # Create the genetic operators
        elitism = ElitismOperator(args.elitism);
        new_blood = NewBloodOperator(args.new_blood);
        gaussian_mutation = GaussianMutationOperator(args.gaussian_mutation[0], args.gaussian_mutation[1]);
        blend_cross_over = BlendCrossoverOperator(args.blend_cross_over, gaussian_mutation);

        # g_number_of_individuals = 20+d*10;
        # g_iterations            = 20+d*10;
        #
        # g_max_mutation_sigma = 0.1;
        # g_min_mutation_sigma = 0.01;
        #
        # start = time.time();
        #
        # objective_function = HandFunction(target_image, d);
        # optimiser = EvolutionaryAlgorithm(objective_function, g_number_of_individuals, initial_guess=initial_guess);
        # optimiser.setSelectionOperator(RankSelection());
        #
        # # Create the genetic operators
        # elitism = ElitismOperator(0.1);
        # new_blood = NewBloodOperator(0.1);
        # gaussian_mutation = GaussianMutationOperator(0.1, 0.2);
        # blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);

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

            if args.plot_metrics:

                temp_best_solution = copy.deepcopy(optimiser.best_solution.parameter_set);
                SOD = temp_best_solution[0]*temp_best_solution[1];
                SDD = temp_best_solution[1];
                setXRayParameters(SOD, SDD);

                temp_objective = -optimiser.best_solution.objective;
                best_angle = [];
                if param != 'Distance':

                    for a in range(d-number_of_distances):
                        best_angle.append(temp_best_solution[a+number_of_distances]);

                while len(best_angle) < number_of_angles:
                    best_angle.append(0.)

                temp_image = bone_rotation(best_angle, 'All');

                MAE = temp_objective;
                ZNCC = zero_mean_normalised_cross_correlation(target_image, temp_image);
                row = [[temp_best_solution, MAE, ZNCC]];
                df2 = pd.DataFrame(row, columns=['Parameters','MAE', 'ZNCC']);
                df_2 = df_2.append(df2, ignore_index=True);

                plot(target_image, temp_image, param, j);
                j+=1;

end=time.time();
computing_time = end-start;

best_solution = copy.deepcopy(optimiser.best_solution.parameter_set);

SOD = best_solution[0]*best_solution[1];
SDD = best_solution[1];
setXRayParameters(SOD, SDD);
for a in range(number_of_angles):
    best_angle.append(best_solution[a+number_of_distances]);

pred_image = bone_rotation(best_angle, 'All');
plt.imsave(args.output + "/EA-%d.png" % args.restart, pred_image, cmap='Greys_r');

MAE = -optimiser.best_solution.objective;
ZNCC = zero_mean_normalised_cross_correlation(target_image, pred_image);
row = [[best_solution, MAE, ZNCC, overall_computing_time]];
df2 = pd.DataFrame(row, columns=['Parameters', 'MAE', 'ZNCC', 'Time']);
df = df.append(df2, ignore_index=True);

error_map = abs(target_image-pred_image);
plt.imsave(args.output + "/error-map.png-%d.png" % args.restart, error_map, cmap='Greys_r');

correlation_map = target_image*pred_image;
plt.imsave(args.output + "/correlation-map-%d.png" % args.restart, correlation_map, cmap='Greys_r');

df.to_csv(args.output + "/results-%d.csv" % args.restart);
df_2.to_csv(ind_folder+ "/results.csv");
