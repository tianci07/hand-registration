import numpy as np
import gvxrPython3 as gvxr

from pymoo.model.problem import Problem
from Metrics import *
from RotateBones import bone_rotation
from sklearn.metrics import mean_absolute_error
import cv2
from HandFunction import *

global target_image;

target_image = cv2.imread("./00382-s1-neg2.png", 0);
target_image = (target_image-target_image.mean())/target_image.std();
target_image[np.isnan(target_image)]=0;
target_image[target_image > 1E308] = 0.;

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

        obj_value = mean_absolute_error(target_image, pred_image);
        obj_list.append(obj_value);

    return obj_list

def getTwoMetrics(prediction):

    MAE_list = [];
    ZNCC_list = [];

    for s in range(len(prediction[:, 0])):
        SOD = prediction[s,0]*prediction[s,1];
        SDD = prediction[s,1]

        setXRayParameters(SOD, SDD);

        angle_list = [];

        for i in range(len(prediction[0, :])-2):

            angle_list.append(prediction[s, i+2]);

        pred_image = bone_rotation(angle_list, 'All');

        MAE = mean_absolute_error(target_image, pred_image);
        ZNCC = -zero_mean_normalised_cross_correlation(target_image, pred_image);
        MAE_list.append(MAE);
        ZNCC_list.append(ZNCC);

    return np.column_stack([MAE_list, ZNCC_list])

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
        self.xl = np.array([0.7, 10., -20., -20., -20., -20., -90., -20., -20., -90., -90., -90.,
                            -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.]);
        self.xu = np.array([0.95, 1000., 20., 20., 20., 20., 0., 20., 20., 0., 0., 0.,
                            20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 0., 0.]);

    def _evaluate(self, x, out, *args, **kwargs):

        objective = getMetrics(x);
        out["F"] = np.array(objective);

class MHR(Problem):
    '''
    Hand Registration with multiple objective functions.
    '''
    def __init__(self):
        super().__init__(n_var=24, n_obj=2, n_constr=0, type_var=np.float32);
        self.xl = np.array([0.7, 10., -20., -20., -20., -20., -90., -20., -20., -90., -90., -90.,
                            -20., -90., -90., -90., -20., -90., -90. ,-90., -20., -90., -90., -90.]);
        self.xu = np.array([0.95, 1000., 20., 20., 20., 20., 0., 20., 20., 0., 0., 0.,
                            20., 0., 0., 0., 20., 0., 0., 0., 20., 0., 0., 0.]);

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = getTwoMetrics(x);
        # out["F"] = getAllMetrics(x);
