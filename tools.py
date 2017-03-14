#!coding:utf-8

import cv2 as cv
import numpy as np
import os
from chainer import optimizers,serializers
import chainer.links as L
import math


def visualize_filters(model_dir,model_name = "gradient_cnn.npz",scale = 5,width = 500,cap_height = 20,normalize = 2,normalizing_factor = 10):
    #normalize - 0:false, 1:over all layers, 2:for each layer, 3:specify number
    height = 0
    if normalize == 3: maxabs = normalizing_factor

    root, exe = os.path.splitext(model_name)
    modelname_file = os.path.join(model_dir, root + "_modelname.txt")
    f = open(modelname_file, "r")
    cnn_classname = f.readline()
    # load cnn class dynamically
    mod = __import__("cnn_structure", fromlist=[cnn_classname])
    class_def = getattr(mod, cnn_classname)
    cnn_architecture = class_def()
    model = L.Classifier(cnn_architecture)

    serializers.load_npz(os.path.join(model_dir, model_name), model)

    children_names = model.predictor._children
    children_names.sort()
    for i in range(len(children_names)):
        child = model.predictor.__dict__[children_names[i]]
        if isinstance(child, L.Convolution2D):
            height += cap_height
            patchsize = child.W.data.shape[2] * scale
            height += (patchsize + 1) * math.ceil(child.W.data.shape[1] / int(width / (patchsize + 1))) * \
                      child.W.data.shape[0]
            if normalize == 1:
                abs_of_max = math.ceil(np.max(child.W.data))
                abs_of_min = math.ceil(np.min(child.W.data))
                _maxabs = abs_of_max if abs_of_max > abs_of_min else abs_of_min
                maxabs = _maxabs if _maxabs > maxabs else maxabs
    filters_image = np.ones((height, width), np.float32)
    write_pointer = [0, 0]  # opencv coordinate
    for i in range(len(children_names)):
        child = model.predictor.__dict__[children_names[i]]
        if isinstance(child, L.Convolution2D):
            if normalize == 2:
                abs_of_max = math.ceil(np.max(child.W.data))
                abs_of_min = math.ceil(np.min(child.W.data))
                maxabs = abs_of_max if abs_of_max > abs_of_min else abs_of_min
            patchsize = child.W.data.shape[2] * scale
            write_pointer[0] += cap_height
            cv.putText(filters_image,
                       children_names[i] + " shape:" + str(child.W.shape) + " normalize factor:" + str(maxabs),
                       (write_pointer[1], write_pointer[0] - 5), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            for k in range(child.W.shape[0]):
                for l in range(child.W.shape[1]):
                    filters_image[write_pointer[0]:write_pointer[0] + patchsize,
                    write_pointer[1]:write_pointer[1] + patchsize] \
                        = cv.resize(child.W.data[k, l], (patchsize, patchsize), interpolation=cv.INTER_NEAREST) / maxabs
                    write_pointer[1] = write_pointer[1] + patchsize + 1
                    if write_pointer[1] + patchsize > width:
                        write_pointer[0] += patchsize + 1
                        write_pointer[1] = 0
                if write_pointer[1] != 0:
                    write_pointer[0] += patchsize + 1
                    write_pointer[1] = 0

    return filters_image

if __name__ == "__main__":
    filters_image = visualize_filters("model/vd_bg350_rot_noBING_Adam_dropout2_whole")

    cv.imshow("filter", filters_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("filter_vis.jpg",(filters_image*255).astype(np.int32))