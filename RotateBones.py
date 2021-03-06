import gvxrPython3 as gvxr
import numpy as np
from sklearn import preprocessing

def getLocalTransformationMatrixSet():
    # Parse the scenegraph
    matrix_set = {};

    node_label_set = [];
    node_label_set.append('root');

    # The list is not empty
    while (len(node_label_set)):

        # Get the last node
        last_node = node_label_set[-1];

        # Get its local transformation
        matrix_set[last_node] = gvxr.getLocalTransformationMatrix(last_node);

        # Remove it from the list
        node_label_set.pop();

        # Add its Children
        for i in range(gvxr.getNumberOfChildren(last_node)):
            node_label_set.append(gvxr.getChildLabel(last_node, i));

    return matrix_set;

def setLocalTransformationMatrixSet(aMatrixSet):
    # Restore the initial local transformation matrices
    for key in aMatrixSet:
        gvxr.setLocalTransformationMatrix(key, aMatrixSet[key]);

def updateLocalTransformationMatrixSet(angles,  finger):

        if finger == 'Root':

            gvxr.rotateNode('root', angles[0], 1, 0, 0);
            gvxr.rotateNode('root', angles[1], 0, 1, 0);
            gvxr.rotateNode('root', angles[2], 0, 0, 1);

        elif finger == 'Thumb':

            gvxr.rotateNode('node-Thu_Meta', angles[0], 1, 0, 0);
            gvxr.rotateNode('node-Thu_Meta', angles[1], 0, 1, 0);
            gvxr.rotateNode('node-Thu_Prox', angles[2], 1, 0, 0);

        elif finger == 'Little':

            gvxr.rotateNode('node-Lit_Prox', angles[0], 1, 0, 0);
            gvxr.rotateNode('node-Lit_Prox', angles[1], 0, 1, 0);
            gvxr.rotateNode('node-Lit_Midd', angles[2], 0, 1, 0);
            gvxr.rotateNode('node-Lit_Dist', angles[3], 0, 1, 0);

        elif finger == 'Ring':
            gvxr.rotateNode('node-Thi_Prox', angles[0], 1, 0, 0);
            gvxr.rotateNode('node-Thi_Prox', angles[1], 0, 1, 0);
            gvxr.rotateNode('node-Thi_Midd', angles[2], 0, 1, 0);
            gvxr.rotateNode('node-Thi_Dist', angles[3], 0, 1, 0);

        elif finger == 'Middle':
            gvxr.rotateNode('node-Mid_Prox', angles[0], 1, 0, 0);
            gvxr.rotateNode('node-Mid_Prox', angles[1], 0, 1, 0);
            gvxr.rotateNode('node-Mid_Midd', angles[2], 0, 1, 0);
            gvxr.rotateNode('node-Mid_Dist', angles[3], 0, 1, 0);

        elif finger == 'Index':
            gvxr.rotateNode('node-Ind_Prox', angles[0], 1, 0, 0);
            gvxr.rotateNode('node-Ind_Prox', angles[1], 0, 1, 0);
            gvxr.rotateNode('node-Ind_Midd', angles[2], 0, 1, 0);
            gvxr.rotateNode('node-Ind_Dist', angles[3], 0, 1, 0);

        elif finger == 'All':

            gvxr.rotateNode('root', angles[0], 1, 0, 0);
            gvxr.rotateNode('root', angles[1], 0, 1, 0);
            gvxr.rotateNode('root', angles[2], 0, 0, 1);

            gvxr.rotateNode('node-Thu_Meta', angles[3], 1, 0, 0);
            gvxr.rotateNode('node-Thu_Meta', angles[4], 0, 1, 0);
            gvxr.rotateNode('node-Thu_Prox', angles[5], 1, 0, 0);

            gvxr.rotateNode('node-Ind_Prox', angles[6], 1, 0, 0);
            gvxr.rotateNode('node-Ind_Prox', angles[7], 0, 1, 0);
            gvxr.rotateNode('node-Ind_Midd', angles[8], 0, 1, 0);
            gvxr.rotateNode('node-Ind_Dist', angles[9], 0, 1, 0);

            gvxr.rotateNode('node-Mid_Prox', angles[10], 1, 0, 0);
            gvxr.rotateNode('node-Mid_Prox', angles[11], 0, 1, 0);
            gvxr.rotateNode('node-Mid_Midd', angles[12], 0, 1, 0);
            gvxr.rotateNode('node-Mid_Dist', angles[13], 0, 1, 0);

            gvxr.rotateNode('node-Thi_Prox', angles[14], 1, 0, 0);
            gvxr.rotateNode('node-Thi_Prox', angles[15], 0, 1, 0);
            gvxr.rotateNode('node-Thi_Midd', angles[16], 0, 1, 0);
            gvxr.rotateNode('node-Thi_Dist', angles[17], 0, 1, 0);

            gvxr.rotateNode('node-Lit_Prox', angles[18], 1, 0, 0);
            gvxr.rotateNode('node-Lit_Prox', angles[19], 0, 1, 0);
            gvxr.rotateNode('node-Lit_Midd', angles[20], 0, 1, 0);
            gvxr.rotateNode('node-Lit_Dist', angles[21], 0, 1, 0);

def average_hand():
    # re-scale Thumb
    gvxr.scaleNode('node-Thu_Prox', 1, 1, 1.086, 'mm');
    gvxr.scaleNode('node-Thu_Dist', 1, 1, 0.897, 'mm');

    # re-scale Index
    gvxr.scaleNode('node-Ind_Prox', 1, 1, 0.969, 'mm');
    gvxr.scaleNode('node-Ind_Midd', 1, 1, 1.065, 'mm');
    gvxr.scaleNode('node-Ind_Dist', 1, 1, 1.141, 'mm');

    # re-scale Middle
    gvxr.scaleNode('node-Mid_Prox', 1, 1, 0.962, 'mm');
    gvxr.scaleNode('node-Mid_Midd', 1, 1, 1.080, 'mm');
    gvxr.scaleNode('node-Mid_Dist', 1, 1, 1.053, 'mm');

    # re-scale Ring
    gvxr.scaleNode('node-Thi_Prox', 1, 1, 1.017, 'mm');
    gvxr.scaleNode('node-Thi_Midd', 1, 1, 1.084, 'mm');
    gvxr.scaleNode('node-Thi_Dist', 1, 1, 1.056, 'mm');

    # re-scale Little
    gvxr.scaleNode('node-Lit_Prox', 1, 1, 1.034, 'mm');
    gvxr.scaleNode('node-Lit_Midd', 1, 1, 1.126, 'mm');
    gvxr.scaleNode('node-Lit_Dist', 1, 1, 1.070, 'mm');


# def updateLocalTransformationMatrixSet(angles):
#         gvxr.rotateNode('root', angles[0], 1, 0, 0);
#         gvxr.rotateNode('root', angles[1], 0, 1, 0);
#         gvxr.rotateNode('root', angles[2], 0, 0, 1);
#
#         gvxr.rotateNode('node-Thu_Meta', angles[3], 1, 0, 0);
#         gvxr.rotateNode('node-Thu_Meta', angles[4], 0, 1, 0);
#         # gvxr.rotateNode('node-Thu_Meta', angles[5], 0, 0, 1);
#
#         gvxr.rotateNode('node-Thu_Prox', angles[6], 1, 0, 0);
#         # gvxr.rotateNode('node-Thu_Prox', angles[7], 0, 1, 0);
#         # gvxr.rotateNode('node-Thu_Prox', angles[8], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Thu_Dist', angles[9], 1, 0, 0);
#         # gvxr.rotateNode('node-Thu_Dist', angles[10], 0, 1, 0);
#         # gvxr.rotateNode('node-Thu_Dist', angles[11], 0, 0, 1);
#
#         gvxr.rotateNode('node-Lit_Prox', angles[12], 1, 0, 0);
#         gvxr.rotateNode('node-Lit_Prox', angles[13], 0, 1, 0);
#         # gvxr.rotateNode('node-Lit_Prox', angles[14], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Lit_Midd', angles[15], 1, 0, 0);
#         gvxr.rotateNode('node-Lit_Midd', angles[16], 0, 1, 0);
#         # gvxr.rotateNode('node-Lit_Midd', angles[17], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Lit_Dist', angles[18], 1, 0, 0);
#         gvxr.rotateNode('node-Lit_Dist', angles[19], 0, 1, 0);
#         # gvxr.rotateNode('node-Lit_Dist', angles[20], 0, 0, 1);
#
#         gvxr.rotateNode('node-Thi_Prox', angles[21], 1, 0, 0);
#         gvxr.rotateNode('node-Thi_Prox', angles[22], 0, 1, 0);
#         # gvxr.rotateNode('node-Thi_Prox', angles[23], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Thi_Midd', angles[24], 1, 0, 0);
#         gvxr.rotateNode('node-Thi_Midd', angles[25], 0, 1, 0);
#         # gvxr.rotateNode('node-Thi_Midd', angles[26], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Thi_Dist', angles[27], 1, 0, 0);
#         gvxr.rotateNode('node-Thi_Dist', angles[28], 0, 1, 0);
#         # gvxr.rotateNode('node-Thi_Dist', angles[29], 0, 0, 1);
#
#         gvxr.rotateNode('node-Mid_Prox', angles[30], 1, 0, 0);
#         gvxr.rotateNode('node-Mid_Prox', angles[31], 0, 1, 0);
#         # gvxr.rotateNode('node-Mid_Prox', angles[32], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Mid_Midd', angles[33], 1, 0, 0);
#         gvxr.rotateNode('node-Mid_Midd', angles[34], 0, 1, 0);
#         # gvxr.rotateNode('node-Mid_Midd', angles[35], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Mid_Dist', angles[36], 1, 0, 0);
#         gvxr.rotateNode('node-Mid_Dist', angles[37], 0, 1, 0);
#         # gvxr.rotateNode('node-Mid_Dist', angles[38], 0, 0, 1);
#
#         gvxr.rotateNode('node-Ind_Prox', angles[39], 1, 0, 0);
#         gvxr.rotateNode('node-Ind_Prox', angles[40], 0, 1, 0);
#         # gvxr.rotateNode('node-Ind_Prox', angles[41], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Ind_Midd', angles[42], 1, 0, 0);
#         gvxr.rotateNode('node-Ind_Midd', angles[43], 0, 1, 0);
#         # gvxr.rotateNode('node-Ind_Midd', angles[44], 0, 0, 1);
#
#         # gvxr.rotateNode('node-Ind_Dist', angles[45], 1, 0, 0);
#         gvxr.rotateNode('node-Ind_Dist', angles[46], 0, 1, 0);
#         # gvxr.rotateNode('node-Ind_Dist', angles[47], 0, 0, 1);
#
#
def bone_rotation(angles, finger):
    '''
    @Params:
        Angles: list of rotating angles.

        finger: choice of "Root", "Thumb", "Index", "Middle", "Ring", "Little",
                "None" or "All"
    '''

    matrix_set = getLocalTransformationMatrixSet();

    if finger != 'None':
        updateLocalTransformationMatrixSet(angles, finger);

    x_ray_image = gvxr.computeXRayImage();
    image = np.array(x_ray_image);
    image = (image-image.mean())/image.std();
    image[np.isnan(image)]=0.;
    image[image > 1E308] = 0.;

    setLocalTransformationMatrixSet(matrix_set);

    return image
