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
        self.vehiclecover = False
        self.locatedistance = locatedistance
        self.result = None

        self.movetocentroid(img)
        self.x -= int((self.windowsize * efactor - self.windowsize)/2)
        self.y -= int((self.windowsize * efactor - self.windowsize)/2)
        self.windowsize = int(round(self.windowsize * efactor))
        self.movetocentroid(img)

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

    def draw(self,img, flags):
        if flags["FN"]:
            if self.result == 0 and self.vehiclecover == True: #False Negative with green
                cv.rectangle(img, (self.x, self.y), (self.x + self.windowsize - 1, self.y + self.windowsize - 1),
                             (0, 255, 0))
        if flags["TP"]:
            if self.result == 1 and self.vehiclecover == True: #True Positive with red
                cv.rectangle(img, (self.x, self.y), (self.x+self.windowsize-1, self.y+self.windowsize-1), (0, 0, 255))
        if flags["FP"]:
            if self.result == 1 and self.vehiclecover == False: #False Positive with blue
                cv.rectangle(img, (self.x, self.y), (self.x + self.windowsize - 1, self.y + self.windowsize - 1),
                             (255, 0, 0))

    def draw_(self,img):
        cv.rectangle(img, (self.x, self.y), (self.x + self.windowsize - 1, self.y + self.windowsize - 1),
                     (0, 255, 0))

    def windowimg(self,img): #arg:RGB image
        img_height,img_width,channnel = img.shape
        xmin = self.x
        ymin = self.y
        xmax = self.x + self.windowsize
        ymax = self.y + self.windowsize
        if xmin < 0:xmin = 0
        if ymin < 0:ymin = 0
        if xmax > img_width:xmax = img_width
        if ymax > img_height:ymax = img_height
        return cv.resize(img[ymin:ymax,xmin:xmax,:],(48,48)).transpose(2,0,1)/255.

    def cover(self,bbox):
        bboxcenter = bbox[0] + int((bbox[2]-bbox[0])/2),bbox[1] + int((bbox[3]-bbox[1])/2)
        windowcenter = self.x + int(self.windowsize/2),self.y + int(self.windowsize/2)
        if math.sqrt((bboxcenter[0]-windowcenter[0])**2 + (bboxcenter[1]-windowcenter[1])**2) < self.windowsize*self.locatedistance:
            self.vehiclecover = True
            return True
        else:
            return False

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

def getslidewindows(img,windowsize,meshsize, slide_param, slide=0.5,mindistance = 0.15,thre1 = 60,thre2 = 100,searchrange = 5):
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

    multiprocess = 1 #マルチプロセス 1:有効化　ただしデバッグ使用不可
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
    step = int(windowsize * slide)
    width = math.ceil(img_width/step)
    height = math.ceil(img_height/step)
    for i in range(len(windows1)):
        posX = i % width
        posY = int((i - posX)/width)
        xmin,ymin,xmax,ymax = posX-searchrange,posY-searchrange,posX+searchrange,posY+searchrange
        if xmin < 0:xmin = 0
        if ymin <0:ymin = 0
        if xmax >= width:xmax = width - 1
        if ymax >= height:ymax = height - 1
        for j in range(ymin,ymax+1):
            for k in range(xmin,xmax+1):
                l = j*width+k
                if windows1[i].repeat == False:
                    if i != l:
                        if math.sqrt((windows1[i].x - windows1[l].x)**2 + (windows1[i].y - windows1[l].y)**2) < windowsize * mindistance:
                            windows1[l].repeat = True
                    if math.sqrt((windows1[i].x - windows2[l].x)**2 + (windows1[i].y - windows2[l].y)**2) < windowsize * mindistance:
                        windows2[l].repeat = True
                    if math.sqrt((windows1[i].x - windows3[l].x) ** 2 + (windows1[i].y - windows3[l].y) ** 2) < windowsize * mindistance:
                        windows3[l].repeat = True
                if windows2[i].repeat == False:
                    if i != l:
                        if math.sqrt((windows2[i].x - windows2[l].x) ** 2 + (
                            windows2[i].y - windows2[l].y) ** 2) < windowsize * mindistance:
                            windows2[l].repeat = True
                    if math.sqrt((windows2[i].x - windows3[l].x) ** 2 + (
                        windows2[i].y - windows3[l].y) ** 2) < windowsize * mindistance:
                        windows3[l].repeat = True
                if windows3[i].repeat == False:
                    if i != l:
                        if math.sqrt((windows3[i].x - windows3[l].x) ** 2 + (
                                    windows3[i].y - windows3[l].y) ** 2) < windowsize * mindistance:
                            windows3[l].repeat = True
    slidewindows = []
    for i in windows1:
        if i.repeat == False:slidewindows.append(i)
    for i in windows2:
        if i.repeat == False:slidewindows.append(i)
    for i in windows3:
        if i.repeat == False:slidewindows.append(i)

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
    cnn_classifier = os.path.join(cnn_path[0], cnn_path[1])
    cnn_optimizer = os.path.join(cnn_path[0], cnn_path[2])
    model = L.Classifier(vehicle_classify_CNN())
    optimizer = optimizers.SGD()
    serializers.load_npz(cnn_classifier, model)
    optimizer.setup(model)
    serializers.load_npz(cnn_optimizer, optimizer)

    if gpu == 1:
        model.to_gpu()
    r = list(range(0, len(data), batch))
    r.pop()
    for i in r:
        if gpu == 1:x = cuda.to_gpu(data[i:i+batch])
        else:x = data[i:i+batch]
        result = F.softmax(model.predictor(x).data).data.argmax(axis=1)
        if gpu == 1:result = cuda.to_cpu(result)
        if i == 0:
            results = result
        else:
            results = np.concatenate((results, result), axis=0)
    if len(r) == 0:j=0
    else:j = i + batch
    if gpu == 1:x = cuda.to_gpu(data[j:])
    else:x = data[j:]
    result = F.softmax(model.predictor(x).data).data.argmax(axis=1)
    if gpu == 1: result = cuda.to_cpu(result)
    if len(r) == 0:
        results = result
    else:
        results = np.concatenate((results, result), axis=0)
    return results

def main():
    imgpath = "C:/work/gspace_yangon/vehicle/test/yangon_test1.tif"  # 単一ファイル処理
    showImage = True  # 処理後画像表示　ディレクトリ内処理の場合はオフ
    procDIR = False  # ディレクトリ内ファイル一括処理
    test_dir = "../vehicle_detection/images/test/"
    result_dir = "" #""../vehicle_detection/images/result/"
    cnn_dir = "model/yangon_vd_161114"
    cnn_classifier = "gradient_cnn.npz"
    cnn_optimizer = "gradient_optimizer.npz"
    gpuEnable = 1 #1:有効化
    batchsize = 50
    mean_image_dir = ""
    mean_image_file = "mean_image.npy"
    logfile_name = "gradient_cnn.log"

    meshsize = 50

    efactor = 1.414
    locatedistance = 0.45

    if result_dir == "":
        if procDIR: result_dir = test_dir
        else: result_dir = os.path.dirname(imgpath)
    if mean_image_dir == "": mean_image_dir = cnn_dir
    mean_image_path = os.path.join(mean_image_dir,mean_image_file)
    cnn_path = [cnn_dir, cnn_classifier, cnn_optimizer]
    slide_param ={"efactor":efactor, "locatedistance":locatedistance}


    #slidewindowsize = 35 #18,35


    img_files = []
    img_files_ignored = []

    date = datetime.now()
    startdate = date.strftime('%Y/%m/%d %H:%M:%S')
    f_startdate = date.strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(result_dir,"result_sHDNN_"+f_startdate)
    logfile_path = os.path.join(result_dir, logfile_name)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if procDIR: #処理可の画像ファイルを確認
        tmp = os.listdir(test_dir)
        files = sorted([os.path.join(test_dir, x) for x in tmp if os.path.isfile(test_dir + x)])
        for i in files:
            root, ext = os.path.splitext(i)
            if ext == ".tif" or ext == ".jpg" or ext == ".png":
                cfgpath = root + ".cfg"
                gt_file = root + ".txt"
                if os.path.isfile(cfgpath) and os.path.isfile(gt_file):
                    img_files.append(i)
                else:
                    img_files_ignored.append(i)
        logfile = open(logfile_path, "a")
        print("all execution start:" + startdate)
        print("all execution start:" + startdate,file=logfile)
        print("%d target file(s):" % len(img_files))
        print("%d target file(s):" % len(img_files),file=logfile)
        for i in img_files:
            print(" " + i)
            print(" " + i,file=logfile)
        print("%d ignored file(s):" % len(img_files_ignored))
        print("%d ignored file(s):" % len(img_files_ignored),file=logfile)
        for i in img_files_ignored:
            print(" " + i)
            print(" " + i,file=logfile)
        print("")
        print("",file=logfile)
        logfile.close()
        all_exec_time = time.time()
    else:
        img_files.append(imgpath)

    logfile = open(logfile_path, "a")
    print("CNN classifire dir:%s" % cnn_dir)
    print("GPU Enable:%d" % gpuEnable)
    print("Batchsize:%d" % batchsize)
    print("Enlarge Factor:%f" %efactor)
    print("Min Window Distance:%f" %locatedistance)

    print("CNN classifire dir:%s" % cnn_dir, file=logfile)
    print("GPU Enable:%d" % gpuEnable, file=logfile)
    print("Batchsize:%d" % batchsize, file=logfile)
    print("Enlarge Factor:%f" %efactor, file=logfile)
    print("Min Window Distance:%f" %locatedistance, file=logfile)
    logfile.close()

    for imgpath in img_files:

        date = datetime.now()
        startdate = date.strftime('%Y/%m/%d %H:%M:%S')
        f_startdate = date.strftime('%Y%m%d_%H%M%S')

        logfile = open(logfile_path, "a")
        print("execution:" + startdate)
        print("execution:" + startdate,file=logfile)
        exec_time = time.time()

        print("image:"+imgpath)
        print("image:" + imgpath,file=logfile)

        img = cv.imread(imgpath)
        print("img shape:"+str(img.shape))
        print("img shape:" + str(img.shape),file=logfile)

        root,ext = os.path.splitext(imgpath)
        gt_file = root + ".txt"
        cfgpath = root+".cfg"
        cfg = open(cfgpath,"r")
        gtwindowsize = int(cfg.readline())
        v_windowsize = gtwindowsize - 1

        init_slidewindowsize = int(round(gtwindowsize / efactor))
        slidewindowsize = int(round(init_slidewindowsize * efactor))

        print("making sliding windows for each gradient image...")
        print("making sliding windows for each gradient image...",file=logfile)
        start = time.time()
        slidewindows, slidewindows_mesh = getslidewindows(img,init_slidewindowsize,meshsize,slide_param)
        end = time.time()
        time_makingslides = end - start
        print("finished.(%.3f seconds)" % time_makingslides)
        print("finished.(%.3f seconds)" % time_makingslides,file=logfile)

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

        print("number of windows:%s size:%d" %(str(len(slidewindows)),slidewindowsize))
        print("number of windows:%s size:%d" %(str(len(slidewindows)),slidewindowsize),file=logfile)

        mean_image = np.load(mean_image_path) #平均画像ロード
        npwindows -= mean_image

        print("predicting windows...")
        print("predicting windows...",file=logfile)
        start = time.time()
        results = predictor(npwindows,cnn_path,batchsize,gpu=gpuEnable)
        end = time.time()
        time_predicting = end - start
        print("finished.(%.3f seconds)" % time_predicting)
        print("finished.(%.3f seconds)" % time_predicting,file=logfile)

        vehicle_list = make_bboxeslist(gt_file)
        vehicle_detected = [False]*len(vehicle_list)

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

        for i in range(len(slidewindows)):
            slidewindows[i].result = results[i]

        print("analyzing results...")
        print("analyzing results...",file=logfile)
        start = time.time()

        for i in range(len(vehicle_list)):
            gt_x = int(math.floor(vehicle_list[i][0] - 1 + (vehicle_list[i][2] - vehicle_list[i][0] + 1)/2))
            gt_y = int(math.floor(vehicle_list[i][1] - 1 + (vehicle_list[i][3] - vehicle_list[i][1] + 1)/2))
            idx_width = [int(math.ceil(gt_x / meshsize))]
            idx_height = [int(math.ceil(gt_y / meshsize))]
            if idx_width[0] != 1:idx_width.append(idx_width[0]-1)
            if idx_width[0] != mesh_width:idx_width.append(idx_width[0]+1)
            if idx_height[0] != 1:idx_height.append(idx_height[0]-1)
            if idx_height[0] != mesh_height:idx_height.append(idx_height[0]+1)
            for k in idx_width:
                for l in idx_height:
                    for j in slidewindows_mesh[(l-1) * mesh_width + k - 1]:
                        if j.cover(vehicle_list[i]):
                            if j.result == 1:
                                vehicle_detected[i] = True

        end = time.time()
        time_analysis = end - start
        print('finished.(%.3f seconds)' % time_analysis)
        print('finished.(%.3f seconds)' % time_analysis,file=logfile)

        TP,TN,FP,FN = 0,0,0,0
        detectobjects = 0

        result_img1 = np.array(img)
        result_img2 = np.array(img)

        for i in slidewindows:
            if i.result == 1 and i.vehiclecover == True:TP += 1
            elif i.result == 0 and i.vehiclecover == False:TN += 1
            elif i.result == 1 and i.vehiclecover == False:FP += 1
            else:FN += 1
            if i.result == 1:detectobjects += 1
            i.draw(result_img1, {"TP":True, "FP":True, "FN":True})
            i.draw(result_img2, {"TP": True, "FP": True, "FN": False})

        exec_time = time.time() - exec_time

        print("---------result--------")
        print("Overall Execution time  :%.3f seconds" % exec_time)
        print("GroundTruth vehicles    :%d" % len(vehicle_detected))
        print("detected objects        :%d" % detectobjects)
        print("PR(d vehicles/d objects):%d/%d %f" %(vehicle_detected.count(True),detectobjects,vehicle_detected.count(True)/detectobjects))
        print("RR(detected vehicles)   :%d/%d %f" % (vehicle_detected.count(True),len(vehicle_detected),vehicle_detected.count(True)/len(vehicle_detected)))
        print("TP,TN,FP,FN             :%d,%d,%d,%d" % (TP, TN, FP, FN))
        print("")

        print("---------result--------",file=logfile)  #to logfile
        print("Overall Execution time  :%.3f seconds" % exec_time,file=logfile)
        print("GroundTruth vehicles    :%d" % len(vehicle_detected),file=logfile)
        print("detected objects        :%d" % detectobjects,file=logfile)
        print("PR(d vehicles/d objects):%d/%d %f" %(vehicle_detected.count(True),detectobjects,vehicle_detected.count(True)/detectobjects),file=logfile)
        print("RR(detected vehicles)   :%d/%d %f" % (vehicle_detected.count(True),len(vehicle_detected),vehicle_detected.count(True)/len(vehicle_detected)),file=logfile)
        print("TP,TN,FP,FN             :%d,%d,%d,%d" % (TP, TN, FP, FN),file=logfile)
        print("",file=logfile)

        logfile.close()

        img_bsname = os.path.basename(imgpath) #結果画像出力
        root,exe = os.path.splitext(img_bsname)
        result_img1_path = os.path.join(result_dir, root + "_sHDNN_TP_FP_FN" + f_startdate + ".jpg")
        cv.imwrite(result_img1_path,result_img1)
        result_img2_path = os.path.join(result_dir, root + "_sHDNN_TP_FP" + f_startdate + ".jpg")
        cv.imwrite(result_img2_path,result_img2)

        if not(procDIR) and showImage: #結果画像表示
            img = result_img1
            w = 0.6
            x,y,c = img.shape
            x = int(x*w)
            y = int(y*w)
            img = cv.resize(img,(y,x))
            cv.imshow("test",img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    if procDIR:
        logfile = open(logfile_path, "a")
        all_exec_time = time.time() - all_exec_time
        print("all exec time:%.3f seconds" % all_exec_time)
        print("all exec time:%.3f seconds" % all_exec_time, file=logfile)
        logfile.close()

if __name__ == "__main__":
    main()
