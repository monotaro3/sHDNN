#!coding:utf-8

import numpy as np
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers,cuda,serializers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import time
from datetime import datetime
import cv2 as cv
import os
from graph import genGraph
import cnn_structure
import logging
import math

def cnn_trainer(model,optimizer,epoch,batchsize,gpu_device,data,val,traindata_ratio,
                trainlog_path,bSnapshot = False,snapshot_interval_number = 0):

    N = (int)(len(val)*traindata_ratio)
    date = datetime.now()
    f_startdate = date.strftime('%Y%m%d_%H%M%S')

    data_train ,data_test = np.split(data,[N])
    val_train , val_test = np.split(val,[N])

    train = tuple_dataset.TupleDataset(data_train,val_train)
    test = tuple_dataset.TupleDataset(data_test,val_test)

    train_iter = iterators.SerialIterator(train,batch_size=batchsize)
    test_iter = iterators.SerialIterator(test,batch_size=batchsize,repeat=False,shuffle=False)

    updater = training.StandardUpdater(train_iter,optimizer,device=gpu_device)
    trainer = training.Trainer(updater,(epoch,"epoch"),out = trainlog_path)

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(test_iter,eval_model,device=gpu_device))
    trainer.extend(extensions.LogReport(log_name="tlog" + f_startdate + ".trainlog"))
    trainer.extend(extensions.PrintReport(["epoch","main/accuracy","validation/main/accuracy"]))
    trainer.extend(extensions.ProgressBar())

    #save snapshot of model and optimizer
    if(bSnapshot):
        snapshot_model = 'gradient_cnn_{.updater.epoch}'
        snapshot_optimizer = 'gradient_optimizer_{.updater.epoch}'
        snapshot_interval = (snapshot_interval_number, "epoch")
        trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
        trainer.extend(extensions.snapshot_object(model, snapshot_model), trigger=snapshot_interval)
        trainer.extend(extensions.snapshot_object(optimizer, snapshot_optimizer), trigger=snapshot_interval)

    trainer.run()


def cnn_train():
    modelload = False # 既存のモデルを読み込んでトレーニング

    model_dir = "model/vd_bg350_rot_noBING_Adam_dropout2_each"
    model_name = "gradient_cnn.npz"
    optimizer_name = "gradient_optimizer.npz"
    logfile_name = "cnn_train.log"
    traindata_ratio = 0.9
    gpu_Enable = True
    snapshot_interval_number = 50
    bSnapshot = True

    trainlog_dir = "trainlog"

    #if modelload == False:
    cnn_architecture = cnn_structure.CNN_dropout2()
    optimizer = optimizers.Adam()

    data_dir = "data/vd_bg350_rot_noBING"
    data_name_prefix = "data"
    val_name_prefix = "val"
    meanimg_name = "mean_image.npy"

    batchsize = 100
    epoch = 200
    #N = 140000 #split data into training and validation

    if modelload:
        root, exe = os.path.splitext(model_name)
        modelname_file = os.path.join(model_dir,root+"_modelname.txt")
        f = open(modelname_file,"r")
        cnn_classname = f.readline()
        # load cnn class dynamically
        mod = __import__("cnn_structure", fromlist=[cnn_classname])
        class_def = getattr(mod, cnn_classname)
        cnn_architecture = class_def()
    else:
        cnn_classname = cnn_architecture.__class__.__name__

    model = L.Classifier(cnn_architecture)

    model_path = os.path.join(model_dir, model_name)
    optimizer_path = os.path.join(model_dir, optimizer_name)
    logfile_path = os.path.join(model_dir, logfile_name)
    # data_path = os.path.join(data_dir, data_name_prefix)
    # val_path = os.path.join(data_dir, val_name_prefix)
    meanimg_loadpath = os.path.join(data_dir, meanimg_name)
    trainlog_path = os.path.join(model_dir, trainlog_dir)
    gpu_device = 0 if gpu_Enable else None

    data_paths = []
    val_paths = []
    data_index = 0

    while(os.path.isfile(os.path.join(data_dir, data_name_prefix+"_"+str(data_index)+".npy"))):
        data_paths.append(os.path.join(data_dir, data_name_prefix+"_"+str(data_index)+".npy"))
        if not os.path.isfile(os.path.join(data_dir, val_name_prefix+"_"+str(data_index)+".npy")):
            print("val file corresponding to data missing")
            return
        else:
            val_paths.append(os.path.join(data_dir, val_name_prefix+"_"+str(data_index)+".npy"))
        data_index += 1
    if len(data_paths) == 0:
        print("no training data in training data dir")
        return

    if (not modelload) and os.path.isfile(model_path):
        print("New model training is chosen but a model already exists at the specified location.")
        print("Process aborted.")
        return

    if not os.path.isdir(trainlog_path): os.makedirs(trainlog_path)

    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler(logfile_path)
    f_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    date = datetime.now()
    startdate = date.strftime('%Y/%m/%d %H:%M:%S')
    f_startdate = date.strftime('%Y%m%d_%H%M%S')
    exec_time = time.time()

    #calculate datasize
    datasize = 0
    traindatanum = 0
    valdatanum = 0
    for val_path in val_paths:
        val = np.load(val_path)
        datasize += len(val)
        traindatanum += (int)(len(val)*traindata_ratio)
        valdatanum += (len(val)-(int)(len(val)*traindata_ratio))

    logger.debug("execution:" + startdate)
    logger.debug("training data dir:%s", data_dir)
    logger.debug("cnn model dir:%s", model_dir)
    logger.debug("batchsize:%d", batchsize)
    logger.debug("epoch:%d", epoch)
    logger.debug("training data file number:%d",len(data_paths))
    logger.debug("loaded data size(number of images):%d",datasize)
    logger.debug("traindata amount:%d", traindatanum)
    logger.debug("validationdata amount:%d", valdatanum)

    if modelload:serializers.load_npz(model_path, model)
    optimizer.setup(model)
    if modelload:serializers.load_npz(optimizer_path, optimizer)
    if(gpu_Enable):model.to_gpu()

    #training start
    train_order_whole = False
    if len(data_paths) == 1:
        data_path = data_paths[0]
        val_path = val_paths[0]
        data = np.load(data_path)
        val = np.load(val_path)
        mean_image = np.load(meanimg_loadpath)
        data -= mean_image
        print("process start:%s (1/1)"% data_path)
        cnn_trainer(model, optimizer, epoch, batchsize, gpu_device, data, val, traindata_ratio,
                    trainlog_path, bSnapshot=bSnapshot, snapshot_interval_number = snapshot_interval_number)
    else:
        if train_order_whole:
            logger.debug("Multiple file Train Order:Whole")
            localepoch=1
            for i in range(epoch):
                print("%d / %d epoch(whole process mode)" %((i+1),epoch))
                for j in range(len(data_paths)):
                    data_path = data_paths[j]
                    val_path = val_paths[j]
                    data = np.load(data_path)
                    val = np.load(val_path)
                    mean_image = np.load(meanimg_loadpath)
                    data -= mean_image
                    print("process start:%s ", data_path)
                    cnn_trainer(model, optimizer, localepoch, batchsize, gpu_device, data, val, traindata_ratio,
                                trainlog_path)
                if (i+1)%snapshot_interval_number == 0:
                    if bSnapshot:
                        if gpu_Enable: model.to_cpu()
                        serializers.save_npz(os.path.join(trainlog_path,"gradient_cnn_"+str(i+1)+".npz"), model)
                        serializers.save_npz(os.path.join(trainlog_path,"gradient_optimizer_"+str(i+1)+".npz"), optimizer)
                        if gpu_Enable: model.to_gpu()
        else:
            logger.debug("Multiple file Train Order:Each")
            snapshot_point = []
            for i in range(len(data_paths)):snapshot_point.append([])
            if bSnapshot:
                for i in range(1,(int)(epoch/snapshot_interval_number)+1):
                    snapshot_point[math.ceil(i*snapshot_interval_number*len(data_paths)/epoch)-1].append(
                        i * snapshot_interval_number * len(data_paths) % epoch
                        if i*snapshot_interval_number*len(data_paths)%epoch != 0
                        else epoch
                    )
            for i in range(len(data_paths)):
                data_path = data_paths[i]
                val_path = val_paths[i]
                data = np.load(data_path)
                val = np.load(val_path)
                mean_image = np.load(meanimg_loadpath)
                data -= mean_image
                print("process start:%s (%d/%d)" %(data_path,i+1,len(data_paths)))
                if len(snapshot_point[i]) == 0:
                    cnn_trainer(model, optimizer, epoch, batchsize, gpu_device, data, val, traindata_ratio,
                                trainlog_path)
                else:
                    for j in range(len(snapshot_point[i])):
                        localepoch = snapshot_point[i][j] if j==0 else snapshot_point[i][j]-snapshot_point[i][j-1]
                        cnn_trainer(model, optimizer, localepoch, batchsize, gpu_device, data, val, traindata_ratio,
                                    trainlog_path)

                        if gpu_Enable: model.to_cpu()
                        serializers.save_npz(os.path.join(trainlog_path, "gradient_cnn_" + str((int)(i * epoch + snapshot_point[i][j])/len(data_paths)) + ".npz"), model)
                        serializers.save_npz(os.path.join(trainlog_path, "gradient_optimizer_" + str((int)(i * epoch + snapshot_point[i][j])/len(data_paths)) + ".npz"),
                                             optimizer)
                        if gpu_Enable: model.to_gpu()
                        if j == len(snapshot_point[i])-1 and snapshot_point[i][j]<epoch:
                            localepoch = epoch -snapshot_point[i][j]
                            cnn_trainer(model, optimizer, localepoch, batchsize, gpu_device, data, val, traindata_ratio,
                                        trainlog_path)

    # data = np.load(data_path)
    # val = np.load(val_path)
    # mean_image = np.load(meanimg_loadpath)
    # data -= mean_image

    #datasize = int(data.size/48/48/3)

    # data_train ,data_test = np.split(data,[N])
    # val_train , val_test = np.split(val,[N])
    #
    # train = tuple_dataset.TupleDataset(data_train,val_train)
    # test = tuple_dataset.TupleDataset(data_test,val_test)
    #
    # train_iter = iterators.SerialIterator(train,batch_size=batchsize)
    # test_iter = iterators.SerialIterator(test,batch_size=batchsize,repeat=False,shuffle=False)
    #
    # updater = training.StandardUpdater(train_iter,optimizer,device=gpu_device)
    # trainer = training.Trainer(updater,(epoch,"epoch"),out = trainlog_path)
    #
    # eval_model = model.copy()
    # eval_model.train = False
    #
    # trainer.extend(extensions.Evaluator(test_iter,eval_model,device=gpu_device))
    # trainer.extend(extensions.LogReport(log_name="tlog" + f_startdate + ".trainlog"))
    # trainer.extend(extensions.PrintReport(["epoch","main/accuracy","validation/main/accuracy"]))
    # trainer.extend(extensions.ProgressBar())
    #
    # #save snapshot of model and optimizer
    #
    # trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(model, snapshot_model), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(optimizer, snapshot_optimizer), trigger=snapshot_interval)
    #
    # trainer.run()

    #training end
    #save model
    if gpu_Enable:model.to_cpu()
    serializers.save_npz(model_path, model)
    serializers.save_npz(optimizer_path, optimizer)
    root, exe = os.path.splitext(model_path)
    modelname_file = root + "_modelname.txt"
    f = open(modelname_file,"w")
    f.write(cnn_classname)
    f.close()

    #save meanimage
    np.save(os.path.join(model_dir, meanimg_name), mean_image) #平均画像の保存
    root, ext = os.path.splitext(meanimg_name)
    meanimg_savepath = root + f_startdate + ext
    meanimg_savepath = os.path.join(model_dir, meanimg_savepath)
    np.save(meanimg_savepath, mean_image)

    #generate graph
    genGraph(trainlog_path)

    exec_time = time.time() - exec_time
    sec = exec_time % 60
    min = (int)(exec_time / 60) % 60
    hour = (int)(exec_time / 60 / 60) % 24
    day = (int)(exec_time / 60 / 60 / 24)
    time_str = str(sec)+" sec(s)"
    if min != 0:time_str = str(min)+" min(s) " + time_str
    if hour != 0:time_str = str(hour)+" hour(s) " + time_str
    if day != 0:time_str = str(day)+" day(s) " + time_str
    logger.debug("exex time:%f (%s)", exec_time,time_str)

if __name__ == "__main__":
    cnn_train()