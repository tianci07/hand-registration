#!/usr/bin/env python3

from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import cv2
import argparse
from skimage.feature import hog
from skimage.transform import resize

global y_pred

def support_vector_classifier(X, Y, X_test):
    '''
    @Input: Training samples and labels.
           Test samples needs to be predicted.
    @Output: Predicted labels.
    '''
    X = np.array(X);
    Y = np.array(Y);

    X_test = np.array(X_test);

    # use support vector classifier with radial basis function
    clf = svm.SVC(C=1.0, kernel='rbf');

    clf.fit(X, Y);

    Y_pred = clf.predict(X_test);

    return Y_pred

def training_test_split(aTrainingSet, ATestSet):
    '''
    @Input: training and test csv file;
    @Output: Training and test samples and labels;
    '''
    X_train = [];
    Y_train = [];
    for i in range(len(aTrainingSet.loc['feature descriptor'])):
        fd = np.loadtxt(aTrainingSet.loc['feature descriptor'][i]);
        X_train.append(fd);
        Y_train.append(aTrainingSet.loc['labels'][i]);

    X_test = [];
    Y_test = [];

    for j in range(len(ATestSet.loc['feature descriptor'])):
        fd = np.loadtxt(ATestSet.loc['feature descriptor'][j]);
        X_test.append(fd);
        Y_test.append(ATestSet.loc['labels'][j]);

    return (X_train, Y_train, X_test, Y_test)

def single_sample_fd(anImage):

    img = cv2.imread(anImage);
    img = resize(img, (64, 32, 3), anti_aliasing=True);
    fd, hog_image = hog(img,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys',
                        visualize=True,
                        multichannel=True);

    return np.array([fd])

parser = argparse.ArgumentParser();
parser.add_argument("--input", help="Input file name");
args = parser.parse_args();

# import data from csv file
training = pd.read_csv("./front-view/training.csv");
test = pd.read_csv("./front-view/test.csv");
training = training.T;
test = test.T;

# split data
x_train, y_train, x_test, y_test = training_test_split(training, test);

# Make prediction on single samples
x_val = single_sample_fd(args.input);
x_val.reshape(1, -1);

# make prediction on test samples
y_pred = support_vector_classifier(x_train, y_train, x_val);

# # print out prediction and score the accuracy
print(y_pred);
np.savetxt('./prediction.text', y_pred, fmt='%s');
# print("Accuracy:", accuracy_score(y_test, y_pred));
