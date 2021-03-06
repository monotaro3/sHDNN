#!coding:utf-8

import cv2 as cv
import os
import numpy as np
from numpy.random import *
import math

def make_bboximg(region,img,dataset_img_size):
    ymin,xmin,ymax,xmax = region #opecvの座標系に変換
    bboximg = img[xmin:xmax+1,ymin:ymax+1,:]
    bboximg = (cv.resize(bboximg,dataset_img_size)).transpose(2, 0, 1)  #無理やりリサイズ
    return bboximg

def make_rotated_bboximg(bbox,img,bboxes_img,angles,dataset_img_size):
    xmin, ymin, xmax, ymax = bbox #横x,縦yのままで
    i_y, i_x, channel = img.shape
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    l_width = math.ceil(width * math.sqrt(2))
    l_height = math.ceil(height * math.sqrt(2))
    diff_width = math.ceil((l_width - width)/2)
    diff_height = math.ceil((l_height - height)/2)
    xmin -= diff_width
    xmax += diff_width
    ymin -= diff_height
    ymax += diff_height
    if xmin < 0 or ymin < 0 or xmax+1 > i_x or ymax+1 > i_y: return -1
    for angle in angles:
        l_img = img[ymin:ymax+1,xmin:xmax+1,:] #opencvの座標系に直す
        d_size = (l_img.shape[1],l_img.shape[0])
        center = (int(math.ceil(d_size[0] / 2)), int(math.ceil(d_size[1] / 2)))
        rmat = cv.getRotationMatrix2D(center,angle,1.0)
        l_img_r = cv.warpAffine(l_img,rmat,d_size)
        img_r = l_img_r[diff_height:-diff_height,diff_width:-diff_width,:]
        img_r = (cv.resize(img_r, dataset_img_size)).transpose(2, 0, 1)
        bboxes_img.append(img_r)

def make_bboxeslist(gt_file):
    gt_txt = open(gt_file,"r")
    bboxes = []
    line = gt_txt.readline()
    while (line):
        category, xmin, ymin, xmax, ymax = line.split(",")
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        bboxes.append([xmin, ymin, xmax, ymax])
        line = gt_txt.readline()
    return bboxes

def make_bgimg(img,size,number,bboxes,dataset_img_size):
    x,y,z = img.shape
    x_max = x-size
    y_max = y-size
    bg_imgs = []
    i = 0
    while i < number:
    #for i in range(number):
        pos_x = randint(0,x_max)
        pos_y = randint(0,y_max)
        if_bg = True
        for j in bboxes:
            if calcIoU(j,[pos_y,pos_y+size-1,pos_x,pos_x+size-1]) >0.4: #座標系をVOC用に変換
                if_bg = False
                break
        if if_bg:
            bg_imgs.append((cv.resize(img[pos_x:pos_x+size,pos_y:pos_y+size,:],dataset_img_size)).transpose(2, 0, 1))
            i += 1
    return bg_imgs

def calcIoU(a, b):  # (xmin,ymin,xmax,ymax)
    if a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3]:
        return 0
    else:
        x = [a[0], a[2], b[0], b[2]]
        y = [a[1], a[3], b[1], b[3]]
        x.sort()
        y.sort()
        x = x[1:3]
        y = y[1:3]
        intersect = (x[1] - x[0]) * (y[1] - y[0])
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersect
        IoU = intersect / union
        return IoU

def make_car_bg_images(imgfile,bg_size,bg_number,angles,dataset_img_size): #画像1枚からgroundtruthとbackgroudの画像を作成
    root,ext = os.path.splitext(imgfile)
    gt_file = root + ".txt"
    img = cv.imread(imgfile)

    bboxes = make_bboxeslist(gt_file)
    bboxes_img = []

    for bbox in bboxes:
        bboxes_img.append(make_bboximg(bbox,img,dataset_img_size))
        if angles != "": make_rotated_bboximg(bbox,img,bboxes_img,angles,dataset_img_size)
    print("  vehicle images finished.(%d imgs)"% len(bboxes_img))
    bg_imgs = make_bgimg(img,bg_size,bg_number,bboxes,dataset_img_size)
    print("  bg images finished.(%d imgs)"% len(bg_imgs))
    return bboxes_img,bg_imgs

def readBINGproposals(filepath,number):
    bing_txt = open(filepath,"r")
    bboxes = []
    line = bing_txt.readline()
    line = bing_txt.readline()
    i = 1
    while (line and i <= number):
        d, xmin, ymin, xmax, ymax = line.split(",")
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        bboxes.append([xmin, ymin, xmax, ymax])
        line = bing_txt.readline()
        i += 1
    return bboxes

def judgePN(proposals, gtboxes, pIoU):
    p_boxes = []
    n_boxes = []
    for i in proposals:
        positive = False
        for j in gtboxes:
            if calcIoU(i, j) >= pIoU:
                positive = True
                break
        if positive:
            p_boxes.append(i)
        else:
            n_boxes.append(i)
    return p_boxes, n_boxes

def make_BING_data(img_path,gt_path,bing_path,BING_pIoU,B_number,angles,dataset_img_size):
    BINGproposals = readBINGproposals(bing_path,B_number)
    gtboxes = make_bboxeslist(gt_path)
    img = cv.imread(img_path)
    p_boxes, n_boxes = judgePN(BINGproposals, gtboxes, BING_pIoU)
    p_images = []
    n_images = []
    for bbox in p_boxes:
        p_images.append(make_bboximg(bbox,img,dataset_img_size))
        if angles != "": make_rotated_bboximg(bbox,img,p_images,angles,dataset_img_size)
    for bbox in n_boxes:
        n_images.append(make_bboximg(bbox,img,dataset_img_size))
        #if angles != "": make_rotated_bboximg(bbox,img,n_images,angles,dataset_img_size)
    #debug
    for i in range(20):
        cv.imshow("BING_p", p_images[i].transpose(1,2,0))
        cv.waitKey(0)
        cv.destroyAllWindows()
    for i in range(20):
        cv.imshow("BING_n", n_images[i].transpose(1, 2, 0))
        cv.waitKey(0)
        cv.destroyAllWindows()
    return p_images, n_images

def make_datasets(img_dir,bg_ratio,angles,dataset_img_size,useBINGProposals,BING_pIoU,B_number): #フォルダからgroundtruthとbackground画像を作成
    vehicle_images = []
    bg_images = []
    tmp = os.listdir(img_dir)
    files = sorted([x for x in tmp if os.path.isfile(os.path.join(img_dir, x))])
    img_files = []
    if len(files) > 0:
        root,ext =os.path.splitext(files[0])
        img_files.append(os.path.join(img_dir,root))
        for i in files:
            root,ext = os.path.splitext(i)
            ab_path = os.path.join(img_dir,root)
            if not ab_path in img_files:
                img_files.append(ab_path)
    for i in img_files:
        img_path = i + ".tif"
        gt_path = i + ".txt"
        cfg_path = i + ".cfg"
        print(" image file:"+img_path)
        cfg = open(cfg_path,"r")
        windowsize = int(cfg.readline()) #パラメータを読み込み
        gt_num =  sum(1 for line in open(gt_path))
        bg_number = int(gt_num * bg_ratio)
        vehicle_imgs,bg_imgs = make_car_bg_images(img_path,windowsize,bg_number,angles,dataset_img_size)
        vehicle_images.extend(vehicle_imgs)
        bg_images.extend(bg_imgs)

        if useBINGProposals:
            bing_path = i + ".bng"
            BING_v_imgs, BING_b_imgs = make_BING_data(img_path,gt_path,bing_path,BING_pIoU,B_number,angles,dataset_img_size)
            vehicle_images.extend(BING_v_imgs)
            bg_images.extend(BING_b_imgs)

    return vehicle_images,bg_images

def main():
    img_dir = "C:/work/vehicle_detection/images/train/test" #"C:/work/gspace_yangon/vehicle/train"
    bg_bias = 35
    data_dir = "data/yangon_vd_161114/test"
    data_name = "data.npy"
    val_name = "val.npy"
    meanimg_name = "mean_image.npy"
    logfile_name = "traindata.log"

    angles = "" #[9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0, 81.0, 90.0]  # set "" if rotation is not necessary

    useBINGProposals = True
    BING_pIoU = 0.6
    B_number = 100000 ; # limit number of BING region proposals which are used

    dataset_img_size = (48,48)

    data_path = os.path.join(data_dir, data_name)
    val_path = os.path.join(data_dir,val_name)
    meanimg_path = os.path.join(data_dir, meanimg_name)

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    print("making data...")
    vehicle_images,bg_images = make_datasets(img_dir,bg_bias, angles,dataset_img_size, useBINGProposals,BING_pIoU, B_number)
    print("finished.")

    vehicle_class = [1]*len(vehicle_images) #車は1
    bg_class = [0]*len(bg_images) #背景は0
    data_ = vehicle_images + bg_images
    val_ = vehicle_class + bg_class
    data = []
    val = []
    indexes = np.random.permutation(len(data_))
    for i in indexes:
        data.append(data_[i])
        val.append(val_[i])
    npdata = np.array(data, np.float32) /255.
    npval = np.array(val, np.int32)

    print("training img dir:%s" %img_dir)
    print("rotation:%s" % angles)
    print("useBINGproposals:%s" % str(useBINGProposals))
    print("vehicle data   :"+str(len(vehicle_images)))
    print("background data:"+str(len(bg_images)))
    print("bacnground bias:%s" % str(bg_bias))
    print("all data       :"+str(len(data)))
    np.save(data_path, npdata)
    np.save(val_path, npval)
    print("data.npy and val.npy saved.")

    logfile = open(os.path.join(data_dir, logfile_name), "a")
    print("training img dir:%s" %img_dir, file=logfile)
    print("rotation:%s" % angles, file=logfile)
    print("useBINGproposals:%s" % str(useBINGProposals), file=logfile)
    print("vehicle data   :"+str(len(vehicle_images)), file=logfile)
    print("background data:"+str(len(bg_images)), file=logfile)
    print("bacnground bias:%s" % str(bg_bias), file=logfile)
    print("all data       :"+str(len(data)), file=logfile)

    if len(data_)>0:
        mean_image = np.zeros(data_[0].shape,dtype=np.double)
        for i in data_:
            mean_image += i/255.
        mean_image = mean_image / len(data_)
        mean_image = np.array(mean_image,dtype=np.float32)

        np.save(meanimg_path, mean_image)
        print("mean_image.npy saved.")

        # cv.imshow("test",mean_image.transpose(1,2,0))
        # cv.waitKey(0)
        # cv.destroyAllWindows()

if __name__ == "__main__":
    main()


# npdata = npdata - mean_image
#
# for i in range(10):
#     cv.namedWindow("test",cv.WINDOW_AUTOSIZE)
#     cv.moveWindow("test",1000,500)
#     print(npval[i])
#     cv.imshow("test",npdata[i].transpose(1,2,0))
#     cv.waitKey(0)
#     cv.destroyAllWindows()



# imgfile = "c:/work/vehicle_detection/images/mikawaharbor_Z19.tif"
# bg_size = 25
# bg_number = 500

# bbox_imgs,bg_imgs = make_car_bg_images(imgfile,bg_size,bg_number)

# print(len(vehicle_images))
# print(len(bg_images))
# for i in range(10):
#     print(vehicle_images[i].shape)
#     cv.namedWindow("test",cv.WINDOW_AUTOSIZE)
#     cv.moveWindow("test",1000,500)
#     cv.imshow("test",vehicle_images[i].transpose(1,2,0))
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# print(len(bboxes_img))
# print(len(bg_imgs))
# for i in range(10):
#     print(bg_imgs[i].shape)
#     cv.imshow("test",bg_imgs[i])
#     cv.waitKey(0)
#     cv.destroyAllWindows()


