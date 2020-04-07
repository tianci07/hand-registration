#!/usr/bin/env python3

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize
import numpy as np
from HandFunction import *
from HR import *
import pandas as pd
from RotateBones import *
import gvxrPython3 as gvxr
import matplotlib.pyplot as plt
import cv2
import time
import argparse

# parser = argparse.ArgumentParser();
# parser.add_argument("--input", help="Input labels");
# args = parser.parse_args();
#
# hand_pose = args.input;
# print(args.input);

setXRayEnvironment();

# average_hand();

# plt.imsave("./02212-2-right/GA-sobel/SSIM/target.png", target_image, cmap='Greys_r');

# problems to be solved
problem = HR();

# Using Genetic Algorithm
algorithm = GA(
    pop_size=200,
    eliminate_duplicates=True
    );

start = time.time();

# Set up an optimiser
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 200),
    seed=1,
    verbose=True
    )
end = time.time();
total_time = end-start;

# Print best solution
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F));

number_of_angles = 22;
number_of_distances = 2;
best_angle = [];
for a in range(number_of_angles):
    best_angle.append(res.X[a+number_of_distances])

pred_image = bone_rotation(best_angle, 'All');
plt.imsave("./02212-2-right/GA-sobel/SSIM/pred.png", pred_image, cmap='Greys_r');

# ZNCC = zero_mean_normalised_cross_correlation(target_image, pred_image);
# MAE = res.F[0];
SSIM = -res.F[0];
ZNCC = zero_mean_normalised_cross_correlation(target_image, pred_image);
MAE = mean_absolute_error(target_image, pred_image);
RMSE = root_mean_squared_error(target_image, pred_image);
NRMSE = normalised_root_mean_squared_error(target_image, pred_image);
# SSIM = structural_similarity(target_image, pred_image);
PSNR = peak_signal_to_noise_ratio(target_image, pred_image);

df = pd.DataFrame();
row = [[res.X, MAE, RMSE, NRMSE, ZNCC, SSIM, PSNR, total_time]];
df2 = pd.DataFrame(row, columns=['Parameters', 'MAE', 'RMSE', 'NRMSE', 'ZNCC', 'SSIM', 'PSNR', 'Time(s)']);
df = df.append(df2, ignore_index=True);

error_map = abs(target_image-pred_image);
plt.imsave("./02212-2-right/GA-sobel/SSIM/error-map.png", error_map, cmap='Greys_r');

correlation_map = target_image*pred_image;
plt.imsave("./02212-2-right/GA-sobel/SSIM/correlation-map.png", correlation_map, cmap='Greys_r');

df.to_csv("./02212-2-right/GA-sobel/SSIM/results.csv" );
