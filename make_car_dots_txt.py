# coding: utf-8

import cv2 as cv
import numpy as np
import os

def checkrepeat(dots,dot):
    round_dots = []
    round_dots.append([dot[0]-1,dot[1]-1])
    round_dots.append([dot[0],dot[1]-1])
    round_dots.append([dot[0]+1,dot[1]-1])
    round_dots.append([dot[0]-1,dot[1]])
    round_dots.append([dot[0],dot[1]])
    round_dots.append([dot[0]+1,dot[1]])
    round_dots.append([dot[0]-1,dot[1]+1])
    round_dots.append([dot[0],dot[1]+1])
    round_dots.append([dot[0]+1,dot[1]+1])
    repeat = False
    for i in round_dots:
        if i in dots:repeat = True
    return repeat

img_dir = "c:/work/vehicle_detection/images/database/"

img_files = []
tmp = os.listdir(img_dir)
files = sorted([x for x in tmp if os.path.isfile(img_dir + x)])
if len(files) > 0:
    root, ext = os.path.splitext(files[0])
    img_files.append(os.path.join(img_dir, root))
    for i in files:
        root, ext = os.path.splitext(i)
        ab_path = os.path.join(img_dir, root)
        if not ab_path in img_files:
            img_files.append(ab_path)

print(img_files)


for j in img_files:
    # imgpath = "c:/work/vehicle_detection/images/mikawaharbor_Z19_gt.tif"
    # root,ext = os.path.splitext(imgpath)
    # txtpath = root+".txt"
    # windowsize = 25

    imgpath = j + ".bmp"
    root,ext = os.path.splitext(imgpath)
    txtpath = root+".txt"
    cfgpath = root+".cfg"
    cfg = open(cfgpath,"r")
    windowsize = int(cfg.readline()) #パラメータ読み込み
    print("windowsize:"+str(windowsize))

    img = cv.imread(imgpath)
    img_x,img_y,channel = img.shape

    rp = np.where(img == 255)
    dot_num = rp[0].size
    dots_ = []
    for i in range(dot_num):
        dots_.append([rp[0][i],rp[1][i]])

    dots = []
    dots_round = []

    for i in dots_:
        if not checkrepeat(dots,i):
            if not checkrepeat(dots_round,i):dots.append(i)
            else:dots_round.append(i)
        else:dots_round.append(i)

    for i in range(10):
        print(img[dots[i][0],dots[i][1]])
        print(dots[i])
    len(dots)

    #テキストへの書き出し、x:横軸、y:縦軸に変換
    txt = open(txtpath,"w")
    h_window = int(windowsize/2)
    for i in dots:
        xmin = i[1]-h_window
        ymin = i[0]-h_window
        xmax = i[1]+(windowsize-h_window-1)
        ymax = i[0]+(windowsize-h_window-1)
        if xmin < 0:xmin = 0
        if ymin < 0:ymin = 0
        if xmax > img_y:xmax = img_y
        if ymax > img_x:ymax = img_x
        txt.write("car,"+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+"\n")
    txt.close()

