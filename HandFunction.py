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


number_of_angles = 0;


def setXRayParameters(SOD, SDD):
    # Compute the source position in 3-D from the SOD
    gvxr.setSourcePosition(SOD,  0.0, 0.0, "cm");
    gvxr.setDetectorPosition(SOD - SDD, 0.0, 0.0, "cm");
    gvxr.usePointSource();


# class SOD_SDD_ObjectiveFunction(ObjectiveFunction):
#
#     def __init__(self, aTargetImage):
#
#         boundaries = [];
#         boundaries.append([0.7, 0.95]);
#         boundaries.append([10, 1000]);
#
#         super().__init__(len(boundaries),
#                          boundaries,
#                          self.objectiveFunctionSOD_SDD,
#                          1);
#
#         self.target_image = aTargetImage;
#         self.target_entropy = shannon_entropy(aTargetImage, 10);
#
#
#     def objectiveFunctionSOD_SDD(self, aSolution):
#
#         SOD = aSolution[0] * aSolution[1];
#         SDD = aSolution[1];
#
#         setXRayParameters(SOD, SDD);
#
#         x_ray_image = gvxr.computeXRayImage();
#         pred_image = np.array(x_ray_image);
#
#         entropy  = shannon_entropy(pred_image, 10);
#
#         return abs(self.target_entropy - entropy)

class HandFunction(ObjectiveFunction):

    def __init__(self, aTargetImage):

        global number_of_angles;

        self.number_of_dimensions = 50;
        self.number_of_angles = 48;
        self.number_of_distances = 2;

        self.boundaries = [];
        self.boundaries.append([0.7, 0.95]);
        self.boundaries.append([10, 1000]);

        while len(self.boundaries) < self.number_of_dimensions:
            self.boundaries.append([-20, 20]);

        super().__init__(len(self.boundaries),
                         self.boundaries,
                         self.objectiveFunction,
                         1);

        self.target_image = aTargetImage;

    def objectiveFunction(self, aSolution):

        SOD = aSolution[0]*aSolution[1];
        SDD = aSolution[1];

        angle_list = [];

        for i in range(self.number_of_angles):

            angle_list.append(aSolution[i + self.number_of_distances]);

        setXRayParameters(SOD, SDD);

        pred_image = bone_rotation(angle_list);

        RMSE = root_mean_squared_error(self.target_image, pred_image);

        return RMSE

def createTarget():
    global target;

    target_SOD = 100;
    target_SDD = 140;
    target_angles_pa = [0, 20, 0, -10, 0, 0,
                        5, 0, 0, 5, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0
                        ];
    #
    # target_angles_pa = [-90, 20, 0, -10, 0, 0,
    #                     5, 0, 0, 5, 0, 0,
    #                     0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0,
    #                     0, 0, 0, 0, 0, 0
    #                     ];
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

# target_image = createTarget();
# df = pd.DataFrame();
#
# number_of_iterations = 10
# number_of_dimensions = 50;
#
# boundaries = [];
# boundaries.append([0.7, 0.95]);
# boundaries.append([10, 1000]);
#
# while len(boundaries) < number_of_dimensions:
#     boundaries.append([-90, 90]);
#
# for i in range(number_of_iterations):
#     start = time.time();
#     optimiser = PRS.PureRandomSearch(number_of_dimensions,
#                                         boundaries,
#                                         objectiveFunction,
#                                         0
#                                         );
#     end = time.time();
#     computing_time = end-start;
#
#     pred_image, df2 = display_metrics(optimiser.best_solution, target, computing_time);
#     df = df.append(df2, ignore_index=True);
#
#     print(df);
#
#     plt.imsave("./posterior-anterior/RMSE/prediction-PRS-%d.png" % (i+1), pred_image, cmap='Greys_r');
#     gvxr.saveLastXRayImage("./posterior-anterior/RMSE/prediction-PRS-%d.mha" % (i+1));
