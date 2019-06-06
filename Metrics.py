import numpy as np
import math
from skimage.measure import compare_ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error

def relative_error(y_true, y_pred):
    s = 0;
    pix1 = 1536;
    pix2 = 1536;

    for i in range(pix1):
        for j in range(pix2):
            s += abs((y_true[i,j] - y_pred[i,j])/y_true[i,j]);
    s = s/(pix1*pix2);

    return s

def zero_mean_normalised_cross_correlation(y_true, y_pred):
    '''
    ZNCC = (1/n)*(1/(std(target_image)*std(est_image)))* SUM_n_by_n{(target_image-
            mean(target_image))*(est_image-mean(est_image))}
    '''
    z = 0;
    pix1 = 1536;
    pix2 = 1024;
    mean1 = np.mean(y_true);
    mean2 = np.mean(y_pred);
    std1 = np.std(y_true);
    std2 = np.std(y_pred);

    for i in range(pix1):
        for j in range(pix2):
            z += (y_true[i,j]-mean1)*(y_pred[i,j]-mean2);
    z = z/(pix1*pix2*std1*std2);

    return z

def root_mean_squared_error(y_true, y_pred):

    return math.sqrt(mean_squared_error(y_true, y_pred))

def structural_similarity(y_true, y_pred):

    return compare_ssim(y_true, y_pred)
