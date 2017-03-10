#!coding:utf-8

import cv2 as cv
import numpy as np
import math
import time
from concurrent import futures
import os
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers,cuda,serializers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from datetime import datetime
import logging
import csv
#from original source
from make_datasets import make_bboxeslist
from cnn_structure import vehicle_classify_CNN

class slidingwindow():
    def __init__(self,img,x,y,windowsize,slidestep = 0.5,efactor = 1.414,locatedistance=0.45):
        self.x = x #x:horizontal,y:vertical upper left corner
        self.y = y
        self.windowsize = windowsize
        self.slidestep = int(self.windowsize * slidestep)
        self.efactor = efactor
        self.repeat = False
        self.bVcover = False
        self.bVdetect = False
        self.locatedistance = locatedistance
        self.result = None
        self.result_probability = None
        self.mesh_idx_x = None
        self.mesh_idx_y = None
        self.overlap = 0
        self.overlap_windows = []

        self.movetocentroid(img)
        self.x -= int((self.windowsize * efactor - self.windowsize)/2)
        self.y -= int((self.windowsize * efactor - self.windowsize)/2)
        self.windowsize = int(round(self.windowsize * efactor))
        self.movetocentroid(img)
        self.checkedwindows = []
        self.connections = []
        self.connections2 = []
        self.cVehicle = None
        self.detectVehicle = None # set vehicle object that this window detected

    def movetocentroid(self,img):
        img_height,img_width = img.shape
        img_xmin,img_ymin,img_xmax,img_ymax = self.x,self.y,self.x+self.windowsize,self.y+self.windowsize
        if img_xmin < 0:img_xmin = 0
        if img_ymin < 0:img_ymin = 0
        if img_xmax > img_width:img_xmax = img_width
        if img_ymax > img_height:img_ymax = img_height

        centerY, centerX = calccentroid(img[img_ymin:img_ymax, img_xmin:img_xmax])
        self.x, self.y = self.x + int(centerX - (img_xmax - img_xmin) / 2), self.y + int(centerY - (img_ymax - img_ymin) / 2)  # directly move to centroid
        #step = self.slidestep / math.sqrt((centerX - (img_xmax - img_xmin)/2)**2 + (centerY - (img_ymax - img_ymin)/2)**2)
        #self.x , self.y = self.x + int(centerX -  (img_xmax - img_xmin)/2)*step, self.y + int(centerY - (img_ymax - img_ymin)/2)*step #move as much as slidestep

    def draw(self,img, flags,show_probability):
        draw = False
        if flags == "TESTONLY":
            color = (132, 33, 225)
            cv.rectangle(img, (self.x, self.y), (self.x + self.windowsize - 1, self.y + self.windowsize - 1),
                             color)
            draw =True
        else:
            if flags["FN"]:
                if self.result == 0 and self.bVcover == True: #False Negative with green
                    color = (0, 255, 0)
                    cv.rectangle(img, (self.x, self.y), (self.x + self.windowsize - 1, self.y + self.windowsize - 1),
                                 color)
                    draw = True
            if flags["TP"]:
                if self.result == 1 and self.bVcover == True: #True Positive with red
                    color = (0, 0, 255)
                    cv.rectangle(img, (self.x, self.y), (self.x+self.windowsize-1, self.y+self.windowsize-1), color)
                    draw = True
            if flags["FP"]:
                if self.result == 1 and self.bVcover == False: #False Positive with blue
                    color = (255, 0, 0)
                    cv.rectangle(img, (self.x, self.y), (self.x + self.windowsize - 1, self.y + self.windowsize - 1),
                                 color)
                    draw = True
        if draw and show_probability:
            cv.putText(img, "{0:.4f}".format(self.result_probability),
                       (self.x, self.y - 1 if self.y - 2 >= 0 else self.y), cv.FONT_HERSHEY_PLAIN, 0.6, color)

    def draw_(self,img):
        cv.rectangle(img, (self.x, self.y), (self.x + self.windowsize - 1, self.y + self.windowsize - 1),
                     (0, 255, 0))

    def windowimg(self,img,raw = False): #arg:RGB image
        img_height,img_width,channnel = img.shape
        xmin = self.x
        ymin = self.y
        xmax = self.x + self.windowsize
        ymax = self.y + self.windowsize
        if xmin < 0:xmin = 0
        if ymin < 0:ymin = 0
        if xmax > img_width:xmax = img_width
        if ymax > img_height:ymax = img_height
        if raw == True: return img[ymin:ymax,xmin:xmax,:]
        return cv.resize(img[ymin:ymax,xmin:xmax,:],(48,48)).transpose(2,0,1)/255.

    def cover(self,bbox):
        bboxcenter = bbox[0] + int((bbox[2]-bbox[0])/2),bbox[1] + int((bbox[3]-bbox[1])/2)
        windowcenter = self.x + int(self.windowsize/2),self.y + int(self.windowsize/2)
        distance = math.sqrt((bboxcenter[0]-windowcenter[0])**2 + (bboxcenter[1]-windowcenter[1])**2)
        if distance < self.windowsize*self.locatedistance:
            #self.bVcover = True
            return distance
        else:
            return -1

    def getCenter(self):
        self.center = self.x + int(math.ceil(self.windowsize / 2)), self.y + int(math.ceil(self.windowsize / 2))
        return self.center

class vehicle():
    def __init__(self,bbox):
        self.bbox = bbox
        self.c_x = int(math.floor(self.bbox[0] - 1 + (self.bbox[2] - self.bbox[0] + 1)/2))
        self.c_y = int(math.floor(self.bbox[1] - 1 + (self.bbox[3] - self.bbox[1] + 1)/2))
        self.connections = []
        self.connections2 = []
        self.covered = False
        self.detected = False

    def windowimg(self,img):
        img_height, img_width, channnel = img.shape
        xmin = self.bbox[0] - 1
        ymin = self.bbox[1] - 1
        xmax = self.bbox[2]
        ymax = self.bbox[3]
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > img_width: xmax = img_width
        if ymax > img_height: ymax = img_height
        return img[ymin:ymax, xmin:xmax, :]

class connection():
    def __init__(self,obj1,obj2,distance):
        self.obj1 = obj1
        self.obj2 = obj2
        self.distance = distance
        self.valid = True

def _calcgrad(img):
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, 1)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, 1)
    img = grad_x**2 + grad_y**2
    img = np.sqrt(img)
    #img = img / 255
    img = img.astype(np.uint8)
    return img
    #img = np.sqrt(grad_x*grad_x + grad_y*grad_y)

def calcgrad(img):
    img_b, img_g, img_r = cv.split(img)
    img_b = _calcgrad(img_b)
    img_g = _calcgrad(img_g)
    img_r = _calcgrad(img_r)
    img = np.maximum(np.maximum(img_b, img_g), img_r)
    return img

def calccentroid(img):
    x,y = img.shape
    int_x_sum = 0
    int_y_sum = 0
    center_x = int(x/2)
    center_y = int(y/2)

    x_mask = np.empty(img.shape)
    y_mask = np.empty(img.shape)
    for i in range(0,x):
        x_mask[i,:] = i
    for i in range(0,y):
        y_mask[:,i] = i
    int_x_sum = (img * x_mask).sum()
    int_y_sum = (img * y_mask).sum()

    # for i in range(0,x):
    #     for j in range(0,y):
    #         int_x_sum += i * img[i,j]
    #         int_y_sum += j * img[i,j]

    if img.sum() != 0:
        center_x = int(int_x_sum / img.sum())
        center_y = int(int_y_sum / img.sum())
    return center_x , center_y

def img_thresholding(img,threshold,direction=0):
    if direction == 0:#threshold below
        img_b, img_g, img_r = cv.split(img)
        img_b[img_b > threshold] = threshold
        img_g[img_g > threshold] = threshold
        img_r[img_r > threshold] = threshold
        img = cv.merge((img_b, img_g, img_r))
    else:#threshold from
        img_b, img_g, img_r = cv.split(img)
        img_b[img_b < threshold] = threshold
        img_g[img_g < threshold] = threshold
        img_r[img_r < threshold] = threshold
        img = cv.merge((img_b, img_g, img_r))
    return img

def makeslidingwindows(img,windowsize,slide_param,slide=0.5): #input image:grayscale
    img_height,img_width = img.shape
    slidewindows = []
    x, y = 0,0
    step = int(windowsize * slide)
    for i in range(math.ceil(img_height/step)):
        for j in range(math.ceil(img_width/step)):
            slidewindows.append(slidingwindow(img,j*step,i*step,windowsize,efactor=slide_param["efactor"],locatedistance=slide_param["locatedistance"]))
    return slidewindows

def getslidewindows(img,windowsize,meshsize, slide_param,overlap_sort_reverse, slide=0.5,mindistance = 0.15,thre1 = 60,thre2 = 100,searchrange = 5):
    img_thre1 = img_thresholding(img,thre1,0)
    img_thre2 = img_thresholding(img,thre2,1)
    img_org = calcgrad(img)
    img_thre1 = calcgrad(img_thre1)
    img_thre2 = calcgrad(img_thre2)
    # cv.imwrite("grad_orijinal.jpg",img_org)
    # cv.imwrite("grad_thre1.jpg", img_thre1)
    # cv.imwrite("grad_thre2.jpg", img_thre2)
    # print("making sliding windows for each gradient image...")
    # start = time.time()
    values = [img_org,img_thre1,img_thre2]
    windows1 = None
    windows2 = None
    windows3 = None

    multiprocess = 0 #マルチプロセス 1:有効化　ただしデバッグ使用不可
    if multiprocess == 1:
        with futures.ProcessPoolExecutor() as executor:     #マルチプロセス処理
            mappings = {executor.submit(makeslidingwindows,n,windowsize,slide_param): n for n in values}
            for future in futures.as_completed(mappings):
                target = mappings[future]
                if (target == img_org).all() :windows1 = future.result()
                if (target == img_thre1).all():windows2 = future.result()
                if (target == img_thre2).all():windows3 = future.result()
    else:
        windows1 = makeslidingwindows(img_org,windowsize,slide_param)   #シングルプロセス処理
        windows2 = makeslidingwindows(img_thre1,windowsize,slide_param)
        windows3 = makeslidingwindows(img_thre2,windowsize,slide_param)
    # end = time.time()
    # time_makingslides = end - start
    print("number of all sliding windows:" + str(len(windows1)+len(windows2)+len(windows3)))
    # print("finished.(%.3f seconds)" %time_makingslides)
    print("discarding repetitive images...")
    img_height, img_width, channel = img.shape

    windowsize = int(round(windowsize * slide_param["efactor"]))
    h_windowsize = int(math.ceil(windowsize / 2))
    r_windows = windows1 + windows2 + windows3
    m_width = int(math.ceil(img_width / h_windowsize))
    m_height = int(math.ceil(img_height / h_windowsize))
    r_meshsize = m_width * m_height
    r_mesh = []
    connections = []
    for i in range(r_meshsize):
        r_mesh.append([])
    for i in r_windows:
        center_x = i.x + h_windowsize - 1
        center_y = i.y + h_windowsize - 1
        if center_x < 0: center_x = 0
        if center_y < 0: center_y = 0
        if center_x >= img_width: center_x = img_width - 1
        if center_y >= img_height: center_y = img_height - 1
        idx_x = int(math.floor(center_x / h_windowsize))
        idx_y = int(math.floor(center_y / h_windowsize))
        i.mesh_idx_x = idx_x
        i.mesh_idx_y = idx_y
        r_mesh[m_width * idx_y + idx_x].append(i)
    for i in r_windows:
        idx_width = [i.mesh_idx_x]
        idx_height = [i.mesh_idx_y]
        if i.mesh_idx_x != 0: idx_width.append(i.mesh_idx_x - 1)
        if i.mesh_idx_x != m_width - 1: idx_width.append(i.mesh_idx_x + 1)
        if i.mesh_idx_y != 0: idx_height.append(i.mesh_idx_y - 1)
        if i.mesh_idx_y != m_height - 1: idx_height.append(i.mesh_idx_y + 1)
        search_idx = []
        for k in idx_width:
            for l in idx_height:
                if not (k == i.mesh_idx_x and l == i.mesh_idx_y):
                    search_idx.append([k, l])
        for j in r_mesh[m_width * i.mesh_idx_y + i.mesh_idx_x]:
            if i is not j:
                if not i in j.checkedwindows:
                    i.checkedwindows.append(j)
                    j.checkedwindows.append(i)
                    distance = math.sqrt((i.x - j.x) ** 2 + (i.y - j.y) ** 2)
                    if distance < windowsize * mindistance:
                        c = connection(i,j,distance)
                        connections.append(c)
                        i.connections.append(c)
                        j.connections.append(c)
                        i.overlap += 1
                        j.overlap += 1
                        # i.overlap_windows.append(j)
                        # j.overlap_windows.append(i)
        for k in search_idx:
            for j in r_mesh[m_width * k[1] + k[0]]:
                if not i in j.checkedwindows:
                    i.checkedwindows.append(j)
                    j.checkedwindows.append(i)
                    distance = math.sqrt((i.x - j.x) ** 2 + (i.y - j.y) ** 2)
                    if distance < windowsize * mindistance:
                        c = connection(i, j, distance)
                        connections.append(c)
                        i.connections.append(c)
                        j.connections.append(c)
                        i.overlap += 1
                        j.overlap += 1
                        # i.overlap_windows.append(j)
                        # j.overlap_windows.append(i)
    connections.sort(key=lambda x: x.distance)
    for i in r_windows:
        i.connections.sort(key=lambda x: x.distance)
    for c in connections:
        if c.valid == True:
            c.valid = False
            window1 = c.obj1
            window2 = c.obj2
            window1.connections = [x for x in window1.connections if x.valid == True]
            window2.connections = [x for x in window2.connections if x.valid == True]
            cnumber_min = len(window1.connections) if len(window1.connections) <= len(window2.connections) else len(window2.connections)
            loser = None
            for i in range(cnumber_min):
                if window1.connections[i].distance < window2.connections[i].distance:
                    loser = window2
                    break
                elif window1.connections[i].distance > window2.connections[i].distance:
                    loser = window1
                    break
            if loser == None:
                if window1.overlap > window2.overlap: loser = window2
                elif window1.overlap < window2.overlap: loser = window1
                else: loser = window2
            loser.repeat = True
            for i in loser.connections:
                i.valid = False
    slidewindows = []
    for i in r_windows:
        if i.repeat == False:
            slidewindows.append(i)

    # #legacy 2nd
    # windowsize = int(round(windowsize*slide_param["efactor"]))
    # h_windowsize = int(math.ceil(windowsize/2))
    # r_windows = windows1 + windows2 + windows3
    # m_width = int(math.ceil(img_width/h_windowsize))
    # m_height = int(math.ceil(img_height/h_windowsize))
    # r_meshsize = m_width * m_height
    # r_mesh = []
    # for i in range(r_meshsize):
    #     r_mesh.append([])
    # for i in r_windows:
    #     center_x = i.x + h_windowsize - 1
    #     center_y = i.y + h_windowsize - 1
    #     if center_x < 0: center_x = 0
    #     if center_y < 0: center_y = 0
    #     if center_x >= img_width: center_x = img_width - 1
    #     if center_y >= img_height: center_y = img_height - 1
    #     idx_x = int(math.floor(center_x / h_windowsize))
    #     idx_y = int(math.floor(center_y / h_windowsize))
    #     i.mesh_idx_x = idx_x
    #     i.mesh_idx_y = idx_y
    #     r_mesh[m_width * idx_y + idx_x].append(i)
    # for i in r_windows:
    #     idx_width = [i.mesh_idx_x]
    #     idx_height = [i.mesh_idx_y]
    #     if i.mesh_idx_x != 0: idx_width.append(i.mesh_idx_x - 1)
    #     if i.mesh_idx_x != m_width - 1: idx_width.append(i.mesh_idx_x + 1)
    #     if i.mesh_idx_y != 0: idx_height.append(i.mesh_idx_y - 1)
    #     if i.mesh_idx_y != m_height - 1: idx_height.append(i.mesh_idx_y + 1)
    #     search_idx = []
    #     for k in idx_width:
    #         for l in idx_height:
    #             if not (k == i.mesh_idx_x and l == i.mesh_idx_y):
    #                 search_idx.append([k,l])
    #     for j in r_mesh[m_width * i.mesh_idx_y + i.mesh_idx_x]:
    #         if i is not j:
    #             if math.sqrt((i.x - j.x) ** 2 + (i.y - j.y) ** 2) < windowsize * mindistance:
    #                 i.overlap += 1
    #                 j.overlap += 1
    #                 i.overlap_windows.append(j)
    #                 j.overlap_windows.append(i)
    #     for k in search_idx:
    #         for j in r_mesh[m_width * k[1] + k[0]]:
    #             if math.sqrt((i.x - j.x) ** 2 + (i.y - j.y) ** 2) < windowsize * mindistance:
    #                 i.overlap += 1
    #                 j.overlap += 1
    #                 i.overlap_windows.append(j)
    #                 j.overlap_windows.append(i)
    # r_windows.sort(key=lambda x: x.overlap,reverse=overlap_sort_reverse)
    # for i in r_windows:
    #     if i.repeat == False:
    #         for j in i.overlap_windows:
    #             j.repeat = True
    # slidewindows = []
    # for i in r_windows:
    #     if i.repeat == False:
    #         slidewindows.append(i)

    ##legacy 1st
    # step = int(windowsize * slide)
    # width = math.ceil(img_width/step)
    # height = math.ceil(img_height/step)
    # for i in range(len(windows1)):
    #     posX = i % width
    #     posY = int((i - posX)/width)
    #     xmin,ymin,xmax,ymax = posX-searchrange,posY-searchrange,posX+searchrange,posY+searchrange
    #     if xmin < 0:xmin = 0
    #     if ymin <0:ymin = 0
    #     if xmax >= width:xmax = width - 1
    #     if ymax >= height:ymax = height - 1
    #     for j in range(ymin,ymax+1):
    #         for k in range(xmin,xmax+1):
    #             l = j*width+k
    #             if windows1[i].repeat == False:
    #                 if i != l:
    #                     if math.sqrt((windows1[i].x - windows1[l].x)**2 + (windows1[i].y - windows1[l].y)**2) < windowsize * mindistance:
    #                         windows1[l].repeat = True
    #                 if math.sqrt((windows1[i].x - windows2[l].x)**2 + (windows1[i].y - windows2[l].y)**2) < windowsize * mindistance:
    #                     windows2[l].repeat = True
    #                 if math.sqrt((windows1[i].x - windows3[l].x) ** 2 + (windows1[i].y - windows3[l].y) ** 2) < windowsize * mindistance:
    #                     windows3[l].repeat = True
    #             if windows2[i].repeat == False:
    #                 if i != l:
    #                     if math.sqrt((windows2[i].x - windows2[l].x) ** 2 + (
    #                         windows2[i].y - windows2[l].y) ** 2) < windowsize * mindistance:
    #                         windows2[l].repeat = True
    #                 if math.sqrt((windows2[i].x - windows3[l].x) ** 2 + (
    #                     windows2[i].y - windows3[l].y) ** 2) < windowsize * mindistance:
    #                     windows3[l].repeat = True
    #             if windows3[i].repeat == False:
    #                 if i != l:
    #                     if math.sqrt((windows3[i].x - windows3[l].x) ** 2 + (
    #                                 windows3[i].y - windows3[l].y) ** 2) < windowsize * mindistance:
    #                         windows3[l].repeat = True
    # slidewindows = []
    # for i in windows1:
    #     if i.repeat == False:slidewindows.append(i)
    # for i in windows2:
    #     if i.repeat == False:slidewindows.append(i)
    # for i in windows3:
    #     if i.repeat == False:slidewindows.append(i)

    mesh_width = math.ceil(img_width/meshsize)
    mesh_height = math.ceil(img_height/meshsize)
    slidewindows_mesh = []
    meshlen = int(mesh_width * mesh_height)
    for i in range(meshlen):
        slidewindows_mesh.append([])
    for i in slidewindows:
        center_x = i.x - 1 + int(math.floor(i.windowsize/2))
        center_y = i.y - 1 + int(math.floor(i.windowsize/2))
        if center_x <= 0 : center_x = 1
        if center_y <= 0 : center_y = 1
        if center_x > img_width: center_x = img_width
        if center_y > img_height: center_y = img_height
        idx_x = int(math.ceil(center_x/meshsize))
        idx_y = int(math.ceil(center_y/meshsize))
        slidewindows_mesh[mesh_width * (idx_y - 1) + idx_x - 1].append(i)
    return slidewindows, slidewindows_mesh

def predictor(data,cnn_path,batch,gpu = 0):
    logger = logging.getLogger(__name__)
    cnn_classifier = os.path.join(cnn_path[0], cnn_path[1])
    cnn_optimizer = os.path.join(cnn_path[0], cnn_path[2])
    root, exe = os.path.splitext(cnn_classifier)
    modelname_file = root + "_modelname.txt"
    try:
        f = open(modelname_file, "r")
        cnn_classname = f.readline()
    except:
        logger.warn("No cnn-class-name file. ""vehicle_classify_CNN"" will be used.")
        cnn_classname = "vehicle_classify_CNN"
    # load cnn class dynamically
    mod = __import__("cnn_structure", fromlist=[cnn_classname])
    class_def = getattr(mod, cnn_classname)
    cnn_architecture = class_def()
    model = L.Classifier(cnn_architecture)
    optimizer = optimizers.SGD()
    serializers.load_npz(cnn_classifier, model)
    optimizer.setup(model)
    serializers.load_npz(cnn_optimizer, optimizer)
    model.predictor.train=False

    if gpu == 1:
        model.to_gpu()
    r = list(range(0, len(data), batch))
    r.pop()
    for i in r:
        if gpu == 1:x = cuda.to_gpu(data[i:i+batch])
        else:x = data[i:i+batch]
        result = F.softmax(model.predictor(x).data).data  #.argmax(axis=1)
        if gpu == 1:result = cuda.to_cpu(result)
        if i == 0:
            results = result
        else:
            results = np.concatenate((results, result), axis=0)
    if len(r) == 0:j=0
    else:j = i + batch
    if gpu == 1:x = cuda.to_gpu(data[j:])
    else:x = data[j:]
    result = F.softmax(model.predictor(x).data).data  #.argmax(axis=1)
    if gpu == 1: result = cuda.to_cpu(result)
    if len(r) == 0:
        results = result
    else:
        results = np.concatenate((results, result), axis=0)
    return results

def main():
    TestOnly = False
    procDIR = True  # ディレクトリ内ファイル一括処理
    showImage = False  # 処理後画像表示　ディレクトリ内処理の場合はオフ
    imgpath = "C:/work/vehicle_detection/images/test/kurume_yumetown.tif"  # 単一ファイル処理
    test_dir = "../vehicle_detection/images/test/" #"e:/work/yangon_satimage/test_2"
    result_dir = "../vehicle_detection/images/test/sHDNN/rot350" #"e:/work/yangon_satimage/test_2/2color"
    cnn_dir = "model/vd_bg350_rot_noBING_Adam_dropout2_whole"
    cnn_classifier = "gradient_cnn.npz"
    cnn_optimizer = "gradient_optimizer.npz"
    mean_image_dir = ""
    mean_image_file = "mean_image.npy"
    logfile_name = "gradient_cnn.log"
    windowsize_default = 18 #18,35
    gpuEnable = 1  # 1:有効化
    batchsize = 100
    efactor = 1.414
    locatedistance = 0.45
    overlap_sort_reverse = True
    meshsize = 50

    show_probability = False
    geoRef = True
    shpOutput = True
    Output_For_Debug = True

    if result_dir == "":
        if procDIR: result_dir = test_dir
        else: result_dir = os.path.dirname(imgpath)
    if mean_image_dir == "": mean_image_dir = cnn_dir
    mean_image_path = os.path.join(mean_image_dir,mean_image_file)
    cnn_path = [cnn_dir, cnn_classifier, cnn_optimizer]
    slide_param ={"efactor":efactor, "locatedistance":locatedistance}

    date = datetime.now()
    startdate = date.strftime('%Y/%m/%d %H:%M:%S')
    f_startdate = date.strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(result_dir, "result_sHDNN_" + f_startdate)
    logfile_path = os.path.join(result_dir, logfile_name)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler(logfile_path)
    f_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    logger.debug("All Execution Start:" + startdate)
    logger.debug("Test Only          :%s", str(TestOnly))
    logger.debug("Process Directory  :%s",str(procDIR))

    img_files = []
    img_files_ignored = []

    if procDIR: #処理可の画像ファイルを確認
        tmp = os.listdir(test_dir)
        files = sorted([os.path.join(test_dir, x) for x in tmp if os.path.isfile(os.path.join(test_dir,x))])
        for i in files:
            root, ext = os.path.splitext(i)
            if ext == ".tif" or ext == ".jpg" or ext == ".png":
                cfgpath = root + ".cfg"
                gt_file = root + ".txt"
                if not TestOnly:
                    if not os.path.isfile(gt_file):
                        logger.debug("No groundtruth file for validation: %s is ignored.", i)
                        img_files_ignored.append(i)
                    else:
                        if not os.path.isfile(cfgpath):
                            logger.warn("No windowsize .cfg: Default windowsize is used for %s.", i)
                        img_files.append(i)
                else:
                    img_files.append(i)
        logger.debug("%d target file(s):", len(img_files))
        for i in img_files:
            logger.debug(" " + i)
        logger.debug("%d ignored file(s):", len(img_files_ignored))
        all_exec_time = time.time()
    else:
        img_files.append(imgpath)

    logger.debug("CNN classifier dir:%s", cnn_dir)
    logger.debug("GPU Enable:%d", gpuEnable)
    logger.debug("Batchsize:%d", batchsize)
    logger.debug("Enlarge Factor:%f", efactor)
    logger.debug("Positive Window Distance:%f", locatedistance)
    #logger.debug("Overlap Sort Reverse:%s", str(overlap_sort_reverse))

    results_stat = []
    root, exe = os.path.splitext(os.path.join(cnn_dir,cnn_classifier))
    modelname_file = root + "_modelname.txt"
    try:
        f = open(modelname_file, "r")
        cnn_classname = f.readline()
    except:
        cnn_classname = "vehicle_classify_CNN"
    results_stat.append([cnn_classname,cnn_dir])
    results_stat.append(["","GroundTruth","DR","FAR","PR","RR","TP","FP","FN","TN","Accuracy"])
    overAcc = 0

    for imgpath in img_files:

        date = datetime.now()
        startdate = date.strftime('%Y/%m/%d %H:%M:%S')
        f_startdate = date.strftime('%Y%m%d_%H%M%S')

        logger.debug("Execution:" + startdate)
        exec_time = time.time()

        logger.debug("Image: " + imgpath)

        img = cv.imread(imgpath)

        logger.debug("img shape:"+str(img.shape))

        root,ext = os.path.splitext(imgpath)
        gt_file = root + ".txt"
        cfgpath = root+".cfg"
        if not os.path.isfile(cfgpath):
            logger.warn("No windowsize .cfg: Default windowsize %d is used for %s.", windowsize_default, imgpath)
            gtwindowsize = windowsize_default
        else:
            cfg = open(cfgpath,"r")
            gtwindowsize = int(cfg.readline())
        v_windowsize = gtwindowsize - 1

        init_slidewindowsize = int(round(gtwindowsize / efactor))
        slidewindowsize = int(round(init_slidewindowsize * efactor))

        logger.debug("making sliding windows for each gradient image...")
        start = time.time()
        slidewindows, slidewindows_mesh = getslidewindows(img,init_slidewindowsize,meshsize,slide_param,overlap_sort_reverse)
        end = time.time()
        time_makingslides = end - start
        logger.debug("finished.(%.3f seconds)" % time_makingslides)

        # img_ = np.array(img) #処理途中のスライドウィンドウ可視化
        # for i in slidewindows:
        #     i.draw_(img_)
        # cv.imwrite("gradient_slidingwindows.jpg",img_)
        # w = 0.6
        # x,y,c = img_.shape
        # x = int(x*w)
        # y = int(y*w)
        # img_ = cv.resize(img_,(y,x))
        # cv.imshow("test",img_)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        windowimgs = []
        for i in slidewindows:
            windowimgs.append(i.windowimg(img))

        npwindows = np.array(windowimgs,np.float32)
        #np.save("windows.npy",npwindows)

        logger.debug("number of windows:%s size:%d",str(len(slidewindows)),slidewindowsize)

        mean_image = np.load(mean_image_path) #平均画像ロード
        npwindows -= mean_image

        #predict windows
        logger.debug("predicting windows...")
        start = time.time()
        results_probability = predictor(npwindows,cnn_path,batchsize,gpu=gpuEnable)
        results = results_probability.argmax(axis=1)
        end = time.time()
        time_predicting = end - start
        logger.debug("finished.(%.3f seconds)" % time_predicting)
        for i in range(len(slidewindows)):
            slidewindows[i].result = results[i]
            slidewindows[i].result_probability = results_probability[i][1]

        if TestOnly:
            result_testonly = np.array(img)
            detectobjects = len([x for x in slidewindows if x.result == 1])
            for i in slidewindows:
                if i.result == 1:
                    i.draw(result_testonly,"TESTONLY",show_probability)
        else:  # Detection Result Validation
            logger.debug("analyzing results...")
            start = time.time()
            vehicle_list = make_bboxeslist(gt_file)
            vehicle_detected = [False]*len(vehicle_list)
            vehicle_connection = []
            for i in vehicle_list:
                vehicle_connection.append([])

            y, x, channel = img.shape
            mesh_width = int(math.ceil(x / meshsize))
            mesh_height = int(math.ceil(y / meshsize))

            for i in vehicle_list:   #groundtruthの矩形を正方形に拡張
                if (i[2]-i[0]) < v_windowsize:
                    if i[0] == 0:i[0] = i[2] - v_windowsize
                    else:i[2] = i[0] + v_windowsize
                if (i[3] - i[1]) < v_windowsize:
                    if i[1] == 0:
                        i[1] = i[3] - v_windowsize
                    else:
                        i[3] = i[1] + v_windowsize

            #one window can detect only one vehicle
            #if a vehicle is detected by multiple windows, the closest window wins
            #windows which lost can detect other vehicles
            gt_vehicles = []
            for i in vehicle_list:
                gt_vehicles.append(vehicle(i))
            connections1 = []
            connections2 = []

            for i in slidewindows:
                i.connections = []

            for i in gt_vehicles:
                # gt_x = int(math.floor(vehicle_list[i][0] - 1 + (vehicle_list[i][2] - vehicle_list[i][0] + 1)/2))
                # gt_y = int(math.floor(vehicle_list[i][1] - 1 + (vehicle_list[i][3] - vehicle_list[i][1] + 1)/2))
                idx_width = [int(math.ceil(i.c_x / meshsize))]
                idx_height = [int(math.ceil(i.c_y / meshsize))]
                if idx_width[0] != 1:idx_width.append(idx_width[0]-1)
                if idx_width[0] != mesh_width:idx_width.append(idx_width[0]+1)
                if idx_height[0] != 1:idx_height.append(idx_height[0]-1)
                if idx_height[0] != mesh_height:idx_height.append(idx_height[0]+1)
                for k in idx_width:
                    for l in idx_height:
                        for j in slidewindows_mesh[(l-1) * mesh_width + k - 1]:
                            distance = j.cover(i.bbox)
                            if distance >= 0:
                                j.bVcover = True
                                c = connection(i,j,distance) #obj: vehicle, window
                                connections1.append(c)
                                i.connections.append(c)
                                j.connections.append(c)
                                c = connection(i, j, distance) #duplicate
                                connections2.append(c)
                                i.connections2.append(c)
                                j.connections2.append(c)
            connections1.sort(key=lambda x: x.distance)
            connections2.sort(key=lambda x: x.distance)
            for i in slidewindows:
                i.connections.sort(key=lambda x: x.distance)
            # for i in gt_vehicles:
            #     i.connections1.sort(key=lambda x: x.distance)
            allow_only_closest = False
            if allow_only_closest:
                for i in slidewindows:
                    if not i.connections == []:
                        i.connections[0].obj1.covered = True
                        if i.result == 1:
                            i.connections[0].obj1.detected = True
                            i.bVdetect = True
            else:
                #coverage
                for c in connections1:
                    if c.valid == True:
                        c.obj1.covered = True
                        for i in c.obj1.connections:
                            i.valid = False
                        for i in c.obj2.connections:
                            i.valid = False
                #detection
                for c in connections2:
                    if c.valid == True:
                        if c.obj2.result == 1:
                            c.obj2.bVdetect = True
                            if c.obj1.detected == False:
                                c.obj1.detected = True
                                i.detectVehicle = c.obj1
                                for i in c.obj2.connections2:
                                    i.valid = False

            # #allow 1 window to detect multiple vehicles
            # for i in gt_vehicles:
            #     # gt_x = int(math.floor(vehicle_list[i][0] - 1 + (vehicle_list[i][2] - vehicle_list[i][0] + 1)/2))
            #     # gt_y = int(math.floor(vehicle_list[i][1] - 1 + (vehicle_list[i][3] - vehicle_list[i][1] + 1)/2))
            #     idx_width = [int(math.ceil(i.c_x / meshsize))]
            #     idx_height = [int(math.ceil(i.c_y / meshsize))]
            #     if idx_width[0] != 1: idx_width.append(idx_width[0] - 1)
            #     if idx_width[0] != mesh_width: idx_width.append(idx_width[0] + 1)
            #     if idx_height[0] != 1: idx_height.append(idx_height[0] - 1)
            #     if idx_height[0] != mesh_height: idx_height.append(idx_height[0] + 1)
            #     for k in idx_width:
            #         for l in idx_height:
            #             for j in slidewindows_mesh[(l - 1) * mesh_width + k - 1]:
            #                 distance = j.cover(i.bbox)
            #                 if distance >= 0:
            #                     j.bVcover = True
            #                     i.covered = True
            #                     if j.result == 1:
            #                         i.detected = True

            # #legacy
            # for i in range(len(vehicle_list)):
            #     gt_x = int(math.floor(vehicle_list[i][0] - 1 + (vehicle_list[i][2] - vehicle_list[i][0] + 1)/2))
            #     gt_y = int(math.floor(vehicle_list[i][1] - 1 + (vehicle_list[i][3] - vehicle_list[i][1] + 1)/2))
            #     idx_width = [int(math.ceil(gt_x / meshsize))]
            #     idx_height = [int(math.ceil(gt_y / meshsize))]
            #     if idx_width[0] != 1:idx_width.append(idx_width[0]-1)
            #     if idx_width[0] != mesh_width:idx_width.append(idx_width[0]+1)
            #     if idx_height[0] != 1:idx_height.append(idx_height[0]-1)
            #     if idx_height[0] != mesh_height:idx_height.append(idx_height[0]+1)
            #     for k in idx_width:
            #         for l in idx_height:
            #             for j in slidewindows_mesh[(l-1) * mesh_width + k - 1]:
            #                 if j.cover(vehicle_list[i]):
            #                     if j.result == 1:
            #                         vehicle_detected[i] = True

            end = time.time()
            time_analysis = end - start

            logger.debug('finished.(%.3f seconds)' % time_analysis)

            TP,TN,FP,FN = 0,0,0,0
            detectobjects = 0

            result_img1 = np.array(img)
            result_img2 = np.array(img)

            # for i in slidewindows:
            #     if i.result == 1 and i.vehiclecover == True:TP += 1
            #     elif i.result == 0 and i.vehiclecover == False:TN += 1
            #     elif i.result == 1 and i.vehiclecover == False:FP += 1
            #     else:FN += 1
            #     if i.result == 1:detectobjects += 1
            #     i.draw(result_img1, {"TP":True, "FP":True, "FN":True})
            #     i.draw(result_img2, {"TP": True, "FP": True, "FN": False})

            for i in slidewindows:
                if i.result == 1 and i.bVcover == True:
                    TP += 1
                elif i.result == 0 and i.bVcover == False:
                    TN += 1
                elif i.result == 1 and i.bVcover == False:
                    FP += 1
                else:
                    FN += 1
                if i.result == 1: detectobjects += 1
                i.draw(result_img1, {"TP": True, "FP": True, "FN": True},show_probability)
                i.draw(result_img2, {"TP": True, "FP": True, "FN": False},show_probability)
            nGT = len(gt_vehicles)
            DR = len([x for x in gt_vehicles if x.covered == True]) / nGT
            n_detected_vehicles = len([x for x in gt_vehicles if x.detected == True])
            FAR = FP / nGT
            PR = n_detected_vehicles/detectobjects if detectobjects != 0 else None
            RR = n_detected_vehicles/nGT if nGT != 0 else None
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            results_stat.append([imgpath,nGT,DR,FAR,PR,RR,TP,FP,FN,TN,Accuracy])
            overAcc += Accuracy

        exec_time = time.time() - exec_time

        logger.debug("---------result--------")
        logger.debug("Overall Execution time  :%.3f seconds", exec_time)
        logger.debug("Detected Objects        :%d", detectobjects)

        if not TestOnly:
            logger.debug("GroundTruth vehicles    :%d", nGT)
            logger.debug("DR(Detection Rate)      :%.3f", DR)
            logger.debug("FAR(False alarm rate)   :%.3f", FAR)
            logger.debug("PR(d vehicles/d objects):%d/%d %s", n_detected_vehicles,detectobjects,str(PR))
            logger.debug("RR(detected vehicles)   :%d/%d %s", n_detected_vehicles,nGT,str(RR))
            logger.debug("TP,TN,FP,FN             :%d,%d,%d,%d", TP, TN, FP, FN)
            logger.debug("Accuracy:%.3f", Accuracy)


        if geoRef:
            from osgeo import gdal
            from geoproc import saveGeoTiff, getCoords, savePointshapefile
            gimg = gdal.Open(imgpath)
            SpaRef = gimg.GetProjection()
            geoTransform = gimg.GetGeoTransform()
            if SpaRef == "":
                geoRef = False

        img_bsname = os.path.basename(imgpath)  # 結果画像出力
        root, exe = os.path.splitext(img_bsname)
        exe = ".tif" if geoRef else ".jpg"
        result_testonly_path = os.path.join(result_dir, root + "_sHDNN_TESTONLY" + f_startdate + exe)
        result_img1_path = os.path.join(result_dir, root + "_sHDNN_TP_FP_FN" + f_startdate + exe)
        result_img2_path = os.path.join(result_dir, root + "_sHDNN_TP_FP" + f_startdate + exe)
        shpdir = os.path.join(result_dir,"shp")
        shppath = os.path.join(shpdir, root + "sHDNN_vc_detected" + f_startdate + ".shp")

        #generate result visualization output for evaluation
        if Output_For_Debug and (not TestOnly):
            debug_file_path = os.path.join(result_dir,"debug")
            if not os.path.isdir(debug_file_path):
                os.makedirs(debug_file_path)
            eval_img_width_max = 1000
            #for TP
            tile_columns = int(eval_img_width_max / slidewindowsize)
            eval_img_width_max = tile_columns * slidewindowsize
            tile_rows = math.ceil(TP / tile_columns)
            eval_img = np.zeros((tile_rows*slidewindowsize,tile_columns*slidewindowsize,3),np.uint8)
            write_pointer = [0,0] #opencv coordinate
            for i in slidewindows:
                if i.result == 1 and i.bVcover == True:
                    img_patch = i.windowimg(img,raw=True)
                    eval_img[write_pointer[0]:write_pointer[0]+img_patch.shape[0],
                    write_pointer[1]:write_pointer[1]+img_patch.shape[1],
                    :] = img_patch
                    write_pointer[1] = (write_pointer[1] + slidewindowsize) % eval_img_width_max
                    if write_pointer[1] == 0: write_pointer[0] += slidewindowsize
            cv.imwrite(os.path.join(debug_file_path,root + "_sHDNN_TPpatch_visualization" + f_startdate + ".jpg"),eval_img)
            #for FP
            tile_rows = math.ceil(FP / tile_columns)
            eval_img = np.zeros((tile_rows * slidewindowsize, tile_columns * slidewindowsize, 3), np.uint8)
            write_pointer = [0, 0]  # opencv coordinate
            for i in slidewindows:
                if i.result == 1 and i.bVcover == False:
                    img_patch = i.windowimg(img, raw=True)
                    eval_img[write_pointer[0]:write_pointer[0] + img_patch.shape[0],
                    write_pointer[1]:write_pointer[1] + img_patch.shape[1],
                    :] = img_patch
                    write_pointer[1] = (write_pointer[1] + slidewindowsize) % eval_img_width_max
                    if write_pointer[1] == 0: write_pointer[0] += slidewindowsize
            cv.imwrite(os.path.join(debug_file_path, root + "_sHDNN_FPpatch_visualization" + f_startdate + ".jpg"),
                       eval_img)
            #for undetected GT and associated FN
            tile_columns = 12  #1 GT and tile_columns-2 FNs
            tile_rows = 0
            eval_img_width_max = tile_columns * gtwindowsize
            for i in gt_vehicles:
                if i.detected == False:
                    #tile_rows += math.ceil(len(filter(lambda x: x.obj2.result == 0,i.connections))/(tile_columns-2))
                    tile_rows += math.ceil(len([x for x in i.connections if x.obj2.result == 0])/(tile_columns-2))
            eval_img = np.zeros((tile_rows * gtwindowsize, tile_columns * gtwindowsize, 3), np.uint8)
            write_pointer = [0, 0]  # opencv coordinate
            for i in gt_vehicles:
                if i.detected == False:
                    img_patch = i.windowimg(img)
                    eval_img[write_pointer[0]:write_pointer[0] + img_patch.shape[0],
                    write_pointer[1]:write_pointer[1] + img_patch.shape[1],
                    :] = img_patch
                    write_pointer[1] += gtwindowsize * 2
                    FNs = [x.obj2 for x in [x for x in i.connections if x.obj2.result == 0]]
                    for j in FNs:
                        img_patch = j.windowimg(img,raw=True)
                        eval_img[write_pointer[0]:write_pointer[0] + img_patch.shape[0],
                        write_pointer[1]:write_pointer[1] + img_patch.shape[1],
                        :] = img_patch
                        write_pointer[1] = (write_pointer[1] + gtwindowsize) % eval_img_width_max
                        if write_pointer[1] == 0:
                            write_pointer[1] += gtwindowsize * 2
                            write_pointer[0] += gtwindowsize
                    if write_pointer[1] != gtwindowsize * 2: write_pointer[0] += gtwindowsize
                    write_pointer[1] = 0
            cv.imwrite(os.path.join(debug_file_path, root + "_sHDNN_undetectedGT_FN_visualization" + f_startdate + ".jpg"),
                       eval_img)

        if geoRef:
            if TestOnly:
                saveGeoTiff(result_testonly, result_testonly_path, geoTransform, SpaRef)
            else:
                saveGeoTiff(result_img1,result_img1_path,geoTransform,SpaRef)
                saveGeoTiff(result_img2, result_img2_path,geoTransform,SpaRef)
            if shpOutput:
                if not os.path.isdir(shpdir):
                    os.makedirs(shpdir)
                vehicle_points_raw = []
                for i in slidewindows:
                    if i.result == 1:vehicle_points_raw.append(i.getCenter())
                if len(vehicle_points_raw) > 0:
                    vehicle_points = getCoords(geoTransform, vehicle_points_raw)
                    savePointshapefile(vehicle_points,"vehicles",SpaRef,shppath)

        if not geoRef:
            if TestOnly:
                cv.imwrite(result_testonly_path,result_testonly)
            else:
                cv.imwrite(result_img1_path, result_img1)
                cv.imwrite(result_img2_path,result_img2)

        if not(procDIR) and showImage: #結果画像表示
            if TestOnly:
                img = result_testonly
            else:
                img = result_img1
            w = 0.6
            x,y,c = img.shape
            x = int(x*w)
            y = int(y*w)
            img = cv.resize(img,(y,x))
            cv.imshow("test",img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        logger.debug("")

    if procDIR:
        all_exec_time = time.time() - all_exec_time
        overAcc = overAcc / len(img_files)
        results_stat.append(["overAcc",overAcc])
        csvfile = os.path.join(result_dir,"result.csv")
        with open(csvfile, 'w') as f:
            writer = csv.writer(f,lineterminator="\n")
            writer.writerows(results_stat)
        logger.debug("all exec time:%.3f seconds" % all_exec_time)
        logger.debug("Overall Accuracy:%.3f", overAcc)

if __name__ == "__main__":
    main()
