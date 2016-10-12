#!coding:utf-8

import cv2 as cv
import os
import numpy as np
from numpy.random import *

def make_bboximg(region,img):
    ymin,xmin,ymax,xmax = region #opecvの座標系に変換
    bboximg = img[xmin:xmax+1,ymin:ymax+1,:]
    bboximg = (cv.resize(bboximg,(48,48))).transpose(2, 0, 1)
    return bboximg

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

def make_bgimg(img,size,number,bboxes):
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
            bg_imgs.append((cv.resize(img[pos_x:pos_x+size,pos_y:pos_y+size,:],(48,48))).transpose(2, 0, 1))
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


def make_car_bg_images(imgfile,bg_size,bg_number): #画像1枚からgroundtruthとbackgroudの画像を作成
    root,ext = os.path.splitext(imgfile)
    gt_file = root + ".txt"
    img = cv.imread(imgfile)

    bboxes = make_bboxeslist(gt_file)
    bboxes_img = []

    for bbox in bboxes:
        bboxes_img.append(make_bboximg(bbox,img))
    print("  vehicle images finished.(%d imgs)"% len(bboxes_img))
    bg_imgs = make_bgimg(img,bg_size,bg_number,bboxes)
    print("  bg images finished.(%d imgs)"% len(bg_imgs))
    return bboxes_img,bg_imgs

def make_datasets(img_dir,bg_ratio): #フォルダからgroundtruthとbackground画像を作成
    vehicle_images = []
    bg_images = []
    tmp = os.listdir(img_dir)
    files = sorted([x for x in tmp if os.path.isfile(img_dir + x)])
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
        vehicle_imgs,bg_imgs = make_car_bg_images(img_path,windowsize,bg_number)
        vehicle_images.extend(vehicle_imgs)
        bg_images.extend(bg_imgs)
    return vehicle_images,bg_images

def main():
    img_dir = "C:/work/vehicle_detection/images/train/"
    print("making data...")
    vehicle_images,bg_images = make_datasets(img_dir,35)
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

    print("vehicle data   :"+str(len(vehicle_images)))
    print("background data:"+str(len(bg_images)))
    print("all data       :"+str(len(data)))
    np.save("data.npy",npdata)
    np.save("val.npy",npval)
    print("data.npy and val.npy saved.")


    if len(data_)>0:
        mean_image = np.zeros(data_[0].shape,dtype=np.double)
        for i in data_:
            mean_image += i/255.
        mean_image = mean_image / len(data_)
        mean_image = np.array(mean_image,dtype=np.float32)

        np.save("mean_image.npy",mean_image)
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


