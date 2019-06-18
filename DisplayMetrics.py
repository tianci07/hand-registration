import numpy as np
from skimage.measure import compare_ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from RotateBones import bone_rotation
from Metrics import*
import pandas as pd
import matplotlib.pyplot as plt
import gvxrPython3 as gvxr
from scipy import stats

def setXRayParameters(SOD, SDD):
    # Compute the source position in 3-D from the SOD
    gvxr.setSourcePosition(SOD,  0.0, 0.0, "cm");
    gvxr.setDetectorPosition(SOD - SDD, 0.0, 0.0, "cm");
    gvxr.usePointSource();

def display_metrics(Method, Prediction, Target, computingTime):

    number_of_angles = 22;
    number_of_distances = 2;

    prediction = Prediction;
    target = Target;
    method = Method;
    prediction[0] = round(prediction[0]*prediction[1]);

    pred_angles = np.zeros(number_of_angles);
    target_angles = np.zeros(number_of_angles);

    for i in range(len(pred_angles)):
        pred_angles[i] = prediction[i+number_of_distances];
        target_angles[i] = target[i+number_of_distances];

    setXRayParameters(prediction[0], prediction[1]);
    pred_image = bone_rotation(pred_angles);

    setXRayParameters(target[0], target[1]);
    target_image = bone_rotation(target_angles);

    diff = [];

    for i in range(number_of_distances):
        diff.append(abs(prediction[i]-target[i]));

    shannon_entropy = stats.entropy(pred_image);
    SSIM = structural_similarity(pred_image, target_image);
    MAE = mean_absolute_error(target_image, pred_image);
    RMSE = root_mean_squared_error(target_image, pred_image);
    RE = relative_error(target_image, pred_image);
    ZNCC = zero_mean_normalised_cross_correlation(target_image, pred_image);
    computing_time = computingTime;

    print('Prediction:' , prediction);
    print('Target: ', target);
    print('SOD and SDD errors: ', diff);
    print('Metrics: \n SSIM: %.8f \t MAE: %.8f \t RMSE: %.8f \t RE: %.8f \t ZNCC: %.8f \
            \t shannon_entropy' %(SSIM, MAE, RMSE, RE, ZNCC));



    row = [[method, prediction[0], diff[0], prediction[1], diff[1], pred_angles,
            shannon_entropy, SSIM, MAE, RMSE, RE, ZNCC, computing_time]];

    df2 = pd.DataFrame(row, columns=['Methods', 'SOD', 'SOD Error','SDD',
                                    'SDD Error', 'Rotating Angles', 'Entropy',
                                    'SSIM', 'MAE', 'RMSE','Relative Error',
                                    'ZNCC', 'Time']);


    return pred_image, df2
