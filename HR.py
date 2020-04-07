import numpy as np
import gvxrPython3 as gvxr

from pymoo.model.problem import Problem
from Metrics import *
from RotateBones import bone_rotation
from sklearn.metrics import mean_absolute_error
import cv2
from HandFunction import *
from skimage import filters
import matplotlib.pyplot as plt

global target_image

target_image = cv2.imread("./02212-2-right.png", 0);
target_image = (target_image-target_image.mean())/target_image.std();
target_image[np.isnan(target_image)]=0;
target_image[target_image > 1E308] = 0.;
plt.imsave("./02212-2-right/NSGA-II-rescaled-sobel/MAE-SSIM/target.png", target_image, cmap='Greys_r');

hand_pose = np.loadtxt('./prediction.text', dtype=np.str);
print(hand_pose);

if hand_pose == 'left':
    target_image = cv2.flip(target_image, 1);
    plt.imsave("./02212-2-right/NSGA-II-rescaled-sobel/MAE-SSIM/target-flip.png", target_image, cmap='Greys_r');

def getMetrics(prediction):

    obj_list = [];
    for s in range(len(prediction[:, 0])):
        SOD = prediction[s,0]*prediction[s,1];
        SDD = prediction[s,1]

        setXRayParameters(SOD, SDD);

        angle_list = [];

        for i in range(len(prediction[0, :])-2):

            angle_list.append(prediction[s, i+2]);

        pred_image = bone_rotation(angle_list, 'All');
        target_image_sobel = filters.sobel(target_image);
        pred_image_sobel =  filters.sobel(pred_image);

        # plt.imsave("./02212-2-right/GA-sobel/SSIM/target-sobel.png", target_image_sobel, cmap='Greys_r');
        # plt.imsave("./02212-2-right/GA-sobel/SSIM/pred-sobel.png", pred_image_sobel, cmap='Greys_r');
        # obj_value = mean_absolute_error(target_image_sobel, pred_image_sobel);

        obj_value = -structural_similarity(target_image_sobel, pred_image_sobel);
        obj_list.append(obj_value);

    return obj_list

def getTwoMetrics(prediction):

    MAE_list = [];
    SSIM_list = [];
    # ZNCC_list = [];

    for s in range(len(prediction[:, 0])):
        SOD = prediction[s,0]*prediction[s,1];
        SDD = prediction[s,1];

        setXRayParameters(SOD, SDD);

        angle_list = [];

        for i in range(len(prediction[0, :])-2):

            angle_list.append(prediction[s, i+2]);

        pred_image = bone_rotation(angle_list, 'All');

        target_image_sobel = filters.sobel(target_image);
        pred_image_sobel =  filters.sobel(pred_image);

        MAE = mean_absolute_error(target_image_sobel, pred_image_sobel);
        SSIM = -structural_similarity(target_image_sobel, pred_image_sobel);
        # ZNCC = -zero_mean_normalised_cross_correlation(target_image, pred_image);

        MAE_list.append(MAE);
        SSIM_list.append(SSIM);
        # ZNCC_list.append(ZNCC);

    return np.column_stack([MAE_list, SSIM_list])
    # return np.column_stack([MAE_list, ZNCC_list])

def getAllMetrics(prediction):

    MAE_list = [];
    RMSE_list = [];
    NRMSE_list = [];
    ZNCC_list = [];
    SSIM_list = [];
    PSNR_list = [];

    for s in range(len(prediction[:, 0])):
        SOD = prediction[s,0]*prediction[s,1];
        SDD = prediction[s,1]

        setXRayParameters(SOD, SDD);

        angle_list = [];

        for i in range(len(prediction[0, :])-2):

            angle_list.append(prediction[s, i+2]);

        pred_image = bone_rotation(angle_list, 'All');

        MAE = mean_absolute_error(target_image, pred_image);
        RMSE = root_mean_squared_error(target_image, pred_image);
        NRMSE = normalised_root_mean_squared_error(target_image, pred_image);
        ZNCC = -zero_mean_normalised_cross_correlation(target_image, pred_image);
        SSIM = -structural_similarity(target_image, pred_image);
        PSNR = -peak_signal_to_noise_ratio(target_image, pred_image);

        MAE_list.append(MAE);
        RMSE_list.append(RMSE);
        NRMSE_list.append(NRMSE);
        ZNCC_list.append(ZNCC);
        SSIM_list.append(SSIM);
        PSNR_list.append(PSNR);

    return np.column_stack([MAE_list, RMSE_list, NRMSE_list, ZNCC_list, SSIM_list, PSNR_list])

class HR(Problem):
    '''
    Hand Registration with single objective function.
    '''
    def __init__(self):
        super().__init__(n_var=24, n_obj=1, n_constr=0, type_var=np.float32);
        self.xl = np.array([0.7, 10., -90., -90., -90., -20., -90., -20., -20., -90., -90., -90.,
                            -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.]);
        self.xu = np.array([0.95, 1000., 90., 90., 90., 20., 0., 20., 20., 0., 0., 0.,
                            20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 20., 0.]);

    def _evaluate(self, x, out, *args, **kwargs):

        objective = getMetrics(x);
        out["F"] = np.array(objective);

class MHR(Problem):
    '''
    Hand Registration with multiple objective functions.
    '''
    def __init__(self):
        super().__init__(n_var=24, n_obj=6, n_constr=0, type_var=np.float32);
        self.xl = np.array([0.7, 10., -90., -90., -90., -20., -90., -20., -20., -90., -90., -90.,
                            -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.]);
        self.xu = np.array([0.95, 1000., 90., 90., 90., 20., 0., 20., 20., 0., 0., 0.,
                            20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 20., 0.]);

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = getTwoMetrics(x);
        # out["F"] = getAllMetrics(x);
