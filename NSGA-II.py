#!/usr/bin/env python3

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from HR import *
import pandas as pd
import gvxrPython3 as gvxr
import cv2
import time
import matplotlib.pyplot as plt

from HandFunction import *
from RotateBones import *


setXRayEnvironment();

average_hand();

# plt.imsave("./02212-2-right/NSGA-II-rescaled-sobel/MAE-SSIM/target.png", target_image, cmap='Greys_r');

# problems to be solved
problem = MHR();

# Using NSGA II
algorithm = NSGA2(
    pop_size=200,
    eliminate_duplicates=True
    );

start = time.time();

# Set up an optimiser
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True);
end = time.time();
total_time = end-start;
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F));

df = pd.DataFrame();
number_of_angles = 22;
number_of_distances = 2;

for r in range(len(res.X[:,0])):

    best_angle = [];
    for a in range(number_of_angles):
        best_angle.append(res.X[r, a+number_of_distances]);

    pred_image = bone_rotation(best_angle, 'All');
    plt.imsave("./02212-2-right/NSGA-II-rescaled-sobel/MAE-SSIM/pred-%d.png" % r, pred_image, cmap='Greys_r');

    # MAE = res.F[r,0];
    # RMSE = res.F[r,1];
    # NRMSE = res.F[r,2];
    # ZNCC = -res.F[r,3];
    # SSIM = -res.F[r,4];
    # PSNR = -res.F[r,5];

    MAE = res.F[r,0];
    SSIM = -res.F[r,1];
    # ZNCC = -res.F[r, 1];
    ZNCC = zero_mean_normalised_cross_correlation(target_image, pred_image);
    RMSE = root_mean_squared_error(target_image, pred_image);
    NRMSE = normalised_root_mean_squared_error(target_image, pred_image);
    # SSIM = structural_similarity(target_image, pred_image);
    PSNR = peak_signal_to_noise_ratio(target_image, pred_image);

    row = [[r, res.X[r,:], MAE, RMSE, NRMSE, ZNCC, SSIM, PSNR, total_time]]
    df2 = pd.DataFrame(row, columns=['image','Parameters', 'MAE', 'RMSE', 'NRMSE', 'ZNCC', 'SSIM', 'PSNR', 'Time(s)']);
    df = df.append(df2, ignore_index=True);

    error_map = abs(target_image-pred_image);
    plt.imsave("./02212-2-right/NSGA-II-rescaled-sobel/MAE-SSIM/error-map-%d.png" % r, error_map, cmap='Greys_r');

    correlation_map = target_image*pred_image;
    plt.imsave("./02212-2-right/NSGA-II-rescaled-sobel/MAE-SSIM/correlation-map-%d.png" % r, correlation_map, cmap='Greys_r');
df.to_csv("./02212-2-right/NSGA-II-rescaled-sobel/MAE-SSIM/results.csv" );
