import gvxrPython3 as gvxr
from ObjectiveFunction import *
from PureRandomSearch import *

from RotateBones import bone_rotation
from Metrics     import *

import random
import math
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import time
import PureRandomSearch as PRS
from DisplayMetrics import *
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

number_of_angles = 0;


def setXRayParameters(SOD, SDD):
    # Compute the source position in 3-D from the SOD
    gvxr.setSourcePosition(SOD,  0.0, 0.0, "cm");
    gvxr.setDetectorPosition(SOD - SDD, 0.0, 0.0, "cm");
    gvxr.usePointSource();

def setXRayEnvironment():

    gvxr.createWindow();
    gvxr.setWindowSize(512, 512);

    #gvxr.usePointSource();
    gvxr.setMonoChromatic(80, "keV", 1000);

    gvxr.setDetectorUpVector(0, 0, -1);
    gvxr.setDetectorNumberOfPixels(768, 1024);
    gvxr.setDetectorPixelSize(0.5, 0.5, "mm"); # 5 dpi

    setXRayParameters(10.0, 100.0);

    gvxr.loadSceneGraph("./hand.dae", "m");
    node_label_set = [];
    node_label_set.append('root');


    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        last_node = node_label_set[-1];

        # Initialise the material properties
        # print("Set ", label, "'s Hounsfield unit");
        # gvxr.setHU(label, 1000)
        Z = gvxr.getElementAtomicNumber("H");
        gvxr.setElement(last_node, gvxr.getElementName(Z));

        # Change the node colour to a random colour
        gvxr.setColour(last_node, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1.0);

        # Remove it from the list
        node_label_set.pop();

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(last_node)):
            node_label_set.append(gvxr.getChildLabel(last_node, i));

    gvxr.moveToCentre('root');
    gvxr.disableArtefactFiltering();
    gvxr.rotateNode('root', -90, 1, 0, 0);


class DistanceAndRootFunction(ObjectiveFunction):

    def __init__(self, aTargetImage):
        self.root_angles = 3;
        self.number_of_distances=2;
        self.boundaries = [];
        self.boundaries.append([0.7, 0.95]);
        self.boundaries.append([10, 1000]);

        self.boundaries.append([-20, 20]);
        self.boundaries.append([-20, 20]);
        self.boundaries.append([-5, 5]);

        super().__init__(len(self.boundaries),
                         self.boundaries,
                         self.objectiveFunction,
                         1);

        self.target_image = aTargetImage;

    def objectiveFunction(self, aSolution):

        SOD = aSolution[0]*aSolution[1];
        SDD = aSolution[1];

        angle_list = [];

        for i in range(self.root_angles):

            angle_list.append(aSolution[i + self.number_of_distances]);

        setXRayParameters(SOD, SDD);

        pred_image = bone_rotation(angle_list, 'root');

        MAE = mean_absolute_error(self.target_image, pred_image);

        return MAE

class AngleFunction(ObjectiveFunction):

    def __init__(self, aTargetImage, aFinger):

        '''
        Optimising each finger, 3 parameters for Thumb and 4 parameters for others.

        Maximising the negative of MAE error.

        '''

        self.finger = aFinger;
        self.boundaries = [];

        if self.finger == 'Thumb':

            self.boundaries.append([-20, 0]);
            self.boundaries.append([-20, 20]);
            self.boundaries.append([-5, 5]);

            self.number_of_angles = 3;

        else:

            self.boundaries.append([-5, 5]);
            self.boundaries.append([-20, 0]);
            self.boundaries.append([-20, 0]);
            self.boundaries.append([-20, 0]);

            self.number_of_angles = 4;

        super().__init__(len(self.boundaries),
                         self.boundaries,
                         self.objectiveFunction,
                         1);

        self.target_image = aTargetImage;

    def objectiveFunction(self, aSolution):

        angle_list = [];

        for i in range(self.number_of_angles):

            angle_list.append(aSolution[i]);

        pred_image = bone_rotation(angle_list, self.finger);

        MAE = mean_absolute_error(self.target_image, pred_image);

        return MAE

class HandFunction(ObjectiveFunction):

    def __init__(self, aTargetImage, aNumberOfDimension):

        self.number_of_dimensions = aNumberOfDimension;
        self.number_of_distances = 2;
        self.number_of_angles = self.number_of_dimensions-self.number_of_distances;

        self.boundaries = [];
        while len(self.boundaries)<self.number_of_dimensions:
            self.boundaries.append([0.7, 0.95]);
            self.boundaries.append([10, 1000]);

            self.boundaries.append([-20, 20]);
            self.boundaries.append([-20, 20]);
            self.boundaries.append([-20, 20]);

            self.boundaries.append([-20, 20]);
            self.boundaries.append([-90, 0]);
            self.boundaries.append([-20, 20]);

            while len(self.boundaries) < self.number_of_dimensions:

                self.boundaries.append([-20, 20]);
                self.boundaries.append([-90, 0]);
                self.boundaries.append([-90, 0]);
                self.boundaries.append([-90, 0]);

        super().__init__(self.number_of_dimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         1);

        self.target_image = aTargetImage;

    def objectiveFunction(self, aSolution):

        SOD = aSolution[0]*aSolution[1];
        SDD = aSolution[1];

        setXRayParameters(SOD, SDD);

        angle_list = [];

        if self.number_of_angles == 0:
            pred_image = bone_rotation(angle_list, 'None');
        else:

            for i in range(self.number_of_angles):

                angle_list.append(aSolution[i+self.number_of_distances]);

            while len(angle_list) < 22:
                angle_list.append(0.);

            pred_image = bone_rotation(angle_list, 'All');

        MAE = mean_absolute_error(self.target_image, pred_image);
        # RMSE = root_mean_squared_error(self.target_image, pred_image);
        # SSIM = structural_similarity(self.target_image, pred_image);
        # RE = relative_error(self.target_image, pred_image);
        # ZNCC = zero_mean_normalised_cross_correlation(self.target_image, pred_image);

        return MAE


def createTarget():
    global target;

    target_SOD = 100;
    target_SDD = 140;
    target_angles_pa = [0, 20, 0, -10, 0, 0,
                        5, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0
                        ];

    target = [];
    target.append(target_SOD);
    target.append(target_SDD);
    for i in range(len(target_angles_pa)):
        target.append(target_angles_pa[i]);

    gvxr.createWindow();
    gvxr.setWindowSize(512, 512);

    #gvxr.usePointSource();
    gvxr.setMonoChromatic(80, "keV", 1000);

    gvxr.setDetectorUpVector(0, 0, -1);
    gvxr.setDetectorNumberOfPixels(1536, 1536);
    gvxr.setDetectorPixelSize(0.5, 0.5, "mm"); # 5 dpi
    setXRayParameters(target_SOD, target_SDD);

    gvxr.loadSceneGraph("./hand.dae", "m");
    node_label_set = [];
    node_label_set.append('root');

    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        last_node = node_label_set[-1];

        # Initialise the material properties
        # print("Set ", label, "'s Hounsfield unit");
        # gvxr.setHU(label, 1000)
        Z = gvxr.getElementAtomicNumber("H");
        gvxr.setElement(last_node, gvxr.getElementName(Z));

        # Change the node colour to a random colour
        gvxr.setColour(last_node, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1.0);

        # Remove it from the list
        node_label_set.pop();

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(last_node)):
            node_label_set.append(gvxr.getChildLabel(last_node, i));

    gvxr.moveToCentre('root');
    gvxr.disableArtefactFiltering();

    target_image = bone_rotation(target_angles_pa);
    plt.imsave("./posterior-anterior/RMSE/target.png", target_image, cmap='Greys_r');

    return target_image, target;
