import numpy as np
import math
from skimage.measure import compare_ssim, compare_mse, compare_psnr, compare_nrmse
from sklearn.metrics import mean_absolute_error, mean_squared_error

def relative_error(y_true, y_pred):

    s = np.sum((y_true-y_pred)/y_true)/(y_true.shape[0]*y_true.shape[1]);
    return s

def zero_mean_normalised_cross_correlation(y_true, y_pred):
    '''
    ZNCC = (1/n)*(1/(std(target_image)*std(est_image)))* SUM_n_by_n{(target_image-
            mean(target_image))*(est_image-mean(est_image))}
    '''

    z = np.sum((y_true-y_true.mean())*(y_pred-y_pred.mean()));

    z /= (y_true.shape[0]*y_true.shape[1]*y_true.std()*y_pred.std());

    return z

def root_mean_squared_error(y_true, y_pred):

    return math.sqrt(mean_squared_error(y_true, y_pred))

def structural_similarity(y_true, y_pred):
    return compare_ssim(y_true, y_pred);

def mean_squared_error(y_true, y_pred):
    return compare_mse(y_true, y_pred);

def peak_signal_to_noise_ratio(y_true, y_pred):
    return compare_psnr(y_true, y_pred, data_range=1);

def normalised_root_mean_squared_error(y_true, y_pred):
    return compare_nrmse(y_true, y_pred);
