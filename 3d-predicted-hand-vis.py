#!/usr/bin/env python3

import numpy as np
import argparse
import gvxrPython3 as gvxr
import pandas as pd
from RotateBones import *
from HandFunction import *


def dataFrameToFloat(aString):
    string = aString.replace("[", "");
    string = string.replace("]", "");
    to_float = np.fromstring(string, dtype=float, sep=" ")

    return to_float

# Read results and visualise predicted 3D hand
parser = argparse.ArgumentParser();

parser.add_argument("--results_csv", help="Result csv files");
args = parser.parse_args();

setXRayEnvironment();

number_of_distances = 2;
number_of_angles = 22;
input_csv = pd.read_csv(args.results_csv, usecols=['Parameters']);

prediction = dataFrameToFloat(input_csv['Parameters'][0]);

SOD = prediction[0]*prediction[1];
SDD = prediction[1];
setXRayParameters(SOD, SDD);

angles = [];

for a in range(number_of_angles):
    angles.append(prediction[a+number_of_distances]);

updateLocalTransformationMatrixSet(angles, 'All');

gvxr.computeXRayImage();

gvxr.displayScene();
gvxr.renderLoop();
