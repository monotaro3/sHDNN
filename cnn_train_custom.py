#!coding:utf-8

import numpy as np
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers,cuda,serializers, Variable
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
import csv

def gen_dms_time_str(time_sec):
    sec = time_sec % 60
    min = (int)(time_sec / 60) % 60
    hour = (int)(time_sec / 60 / 60) % 24
    day = (int)(time_sec / 60 / 60 / 24)
    time_str = ""
    if day!=0: time_str = str(day) + " day(s)"
    if time_str != "" or hour != 0: time_str = time_str + " " + str(hour) + " hour(s)"
    if time_str != "" or min != 0: time_str = time_str + " " + str(min) + " min(s)"
    time_str = time_str + (" " if time_str != "" else "") + str(sec) + " sec(s)"
    return time_str

def dl_model_load(model_dir, model_name = "gradient_cnn.npz", optimizer_name = "gradient_optimizer.npz"):
    if not os.path.isdir(model_dir):
        print("Can't load - No model directory")
        return None,None
    else:
        model_path = os.path.join(model_dir, model_name)
        optimizer_path = os.path.join(model_dir, optimizer_name)
        root, exe = os.path.splitext(model_path)
        modelname_file = root + "_modelname.txt"
        try:
            f = open(modelname_file, "r")
            cnn_classname = f.readline()
        except:
            cnn_classname = "vehicle_classify_CNN"
        mod = __import__("cnn_structure", fromlist=[cnn_classname])
        class_def = getattr(mod, cnn_classname)
        cnn_architecture = class_def()

        model = L.Classifier(cnn_architecture)
        optimizer = optimizers.SGD()
        serializers.load_npz(model_path, model)
        optimizer.setup(model)
        serializers.load_npz(optimizer_path, optimizer)

        return model, optimizer

def dl_model_save(model,optimizer,model_dir,model_name = "gradient_cnn.npz", optimizer_name = "gradient_optimizer.npz", *, snapshot = None):
    #model.to_cpu()
    if snapshot != None:
        root, exe = os.path.splitext(model_name)
        model_path = os.path.join(model_dir,root + "_" + str(snapshot)+exe)
        root, exe = os.path.splitext(optimizer_name)
        optimizer_path = os.path.join(model_dir,root + "_" + str(snapshot)+exe)
    else:
        model_path = os.path.join(model_dir,model_name)
        optimizer_path = os.path.join(model_dir,optimizer_name)
    serializers.save_npz(model_path, model)
    serializers.save_npz(optimizer_path, optimizer)
    if snapshot == None:
        root, exe = os.path.splitext(model_path)
        modelname_file = root + "_modelname.txt"
        f = open(modelname_file, "w")
        f.write(model.predictor.__class__.__name__)
        f.close()

def load_datapaths(data_dir,data_name_prefix = "data", val_name_prefix = "val"):
    data_paths = []
    val_paths = []
    data_index = 0

    if not os.path.isdir(data_dir):
        print("No data dir")
        return 0,0
    while (os.path.isfile(os.path.join(data_dir, data_name_prefix + "_" + str(data_index) + ".npy"))):
        if not os.path.isfile(os.path.join(data_dir, val_name_prefix + "_" + str(data_index) + ".npy")):
            print("val file corresponding to data missing")
        else:
            data_paths.append(os.path.join(data_dir, data_name_prefix + "_" + str(data_index) + ".npy"))
            val_paths.append(os.path.join(data_dir, val_name_prefix + "_" + str(data_index) + ".npy"))
        data_index += 1
    if len(data_paths) == 0:
        print("no training data in training data dir")
        return 0,0
    else:
        return data_paths, val_paths

def dl_drain_curriculum(model_dir, data_dir, batch_learn, batch_check, epoch, snapshot_interval, from_scratch, dl_model_scratch,
                        optimizer_scratch, gpu_use = True, traindata_ratio=0.9,trainlog_dir = "trainlog",meanimg_name = "mean_image.npy",
                        *,logger = None):
    # gpu_use = True
    # epoch = 200
    # batch_learn = 100
    # batch_check = 1000
    # traindata_ratio = 0.9
    #
    # model_path = ""
    # data_dir = ""

    if from_scratch:
        model = L.Classifier(dl_model_scratch)
        optimizer = optimizer_scratch
        optimizer.setup(model)
    else:
        model,optimizer = dl_model_load(model_dir)
        if model == None:
            print("Model load failure")
            return -1

    data_paths, val_paths = load_datapaths(data_dir)
    if data_paths == 0: return -1

    meanimg_path = os.path.join(data_dir,meanimg_name)

    if gpu_use:
        model.to_gpu()#cuda.to_gpu(model)
        #cuda.to_gpu(optimizer)

    #train_stat = [] #loss_train, acc_train, loss_valid, acc_valid

    date = datetime.now()
    startdate = date.strftime('%Y/%m/%d %H:%M:%S')
    f_startdate = date.strftime('%Y%m%d_%H%M%S')
    exec_time = time.time()
    csvfile = os.path.join(model_dir, trainlog_dir, "trainlog" + f_startdate + ".csv")
    if not os.path.isdir(os.path.join(model_dir,trainlog_dir)): os.makedirs(os.path.join(model_dir,trainlog_dir))

    with open(csvfile, 'a') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows([["loss_train", "accuracy_train", "loss_validation", "accuracy_validation"]])

    # calculate datasize
    datasize = 0
    traindatanum = 0
    valdatanum = 0
    for val_path in val_paths:
        val = np.load(val_path)
        datasize += len(val)
        traindatanum += (int)(len(val) * traindata_ratio)
        valdatanum += (len(val) - (int)(len(val) * traindata_ratio))

    logger.debug("execution:" + startdate)
    logger.debug("training data dir:%s", data_dir)
    logger.debug("dl model dir:%s", model_dir)
    logger.debug("check batchsize:%d", batch_check)
    logger.debug("learn batchsize:%d", batch_learn)
    logger.debug("epoch:%d", epoch)
    logger.debug("training data file number:%d", len(data_paths))
    logger.debug("loaded data size(number of images):%d", datasize)
    logger.debug("traindata amount:%d", traindatanum)
    logger.debug("validationdata amount:%d", valdatanum)

    overall_train_loss = 0
    overall_train_accuracy = 0
    overall_validation_loss = 0
    overall_validation_accuracy = 0

    for e in range(epoch):
        print("%d / %d epoch" % ((e + 1), epoch))
        loss_train = 0
        accuracy_train = 0
        iteration_train = 0
        loss_validation = 0
        accuracy_validation = 0
        iteration_validation = 0
        for i in range(len(data_paths)):
            data_path = data_paths[i]
            val_path = val_paths[i]
            data = np.load(data_path)
            val = np.load(val_path)
            mean_image = np.load(meanimg_path)
            data -= mean_image
            N = (int)(len(val) * traindata_ratio)
            data_train, data_valid = np.split(data,[N])
            val_train, val_valid = np.split(val,[N])
            indexes = np.random.permutation(len(val_train))
            max_iter_train = math.ceil(len(val_train)/batch_check)
            max_iter_validation = math.ceil(len(val_valid)/batch_learn)
            #training
            for j in range(max_iter_train):
                if batch_check == batch_learn:
                    data_train_batch = data_train[
                        indexes[j * batch_learn:(j + 1) * batch_learn if j < max_iter_train - 1 else None]]
                    val_train_batch = val_train[
                        indexes[j * batch_learn:(j + 1) * batch_learn if j < max_iter_train - 1 else None]]
                else:
                #check loss value and select traindata
                    data_check = data_train[indexes[j*batch_check:(j+1)*batch_check if j < max_iter_train-1 else None]]
                    val_check = val_train[indexes[j*batch_check:(j+1)*batch_check if j < max_iter_train-1 else None]]
                    if gpu_use:
                        data_check = cuda.to_gpu(data_check)
                    data_check_v = Variable(data_check)
                    model.predictor.train = False
                    result_check = F.softmax(model.predictor(data_check_v).data).data
                    if gpu_use:
                        result_check = cuda.to_cpu(result_check)
                        data_check = cuda.to_cpu(data_check)
                    indexes_toploss = np.argsort(result_check[np.arange(len(result_check)),val_check])
                    data_train_batch = data_check[indexes_toploss[0:batch_learn]]
                    val_train_batch = val_check[indexes_toploss[0:batch_learn]]
                #train
                if gpu_use:
                    data_train_batch = cuda.to_gpu(data_train_batch)
                    val_train_batch = cuda.to_gpu(val_train_batch)
                data_train_batch = Variable(data_train_batch)
                val_train_batch = Variable(val_train_batch)
                model.predictor.train = True
                optimizer.update(model,data_train_batch,val_train_batch)
                loss_batch = model.loss
                accuracy_batch = model.accuracy
                loss_train += loss_batch.data
                accuracy_train += accuracy_batch.data
            #validation
            for j in range(max_iter_validation):
                data_valid_batch = data_valid[j*batch_learn:(j+1)*batch_learn if j < max_iter_validation-1 else None]
                val_valid_batch = val_valid[j*batch_learn:(j+1)*batch_learn if j < max_iter_validation-1 else None]
                if gpu_use:
                    data_valid_batch = cuda.to_gpu(data_valid_batch)
                    val_valid_batch = cuda.to_gpu(val_valid_batch)
                model.predictor.train = False
                data_valid_batch = Variable(data_valid_batch)
                val_valid_batch = Variable(val_valid_batch)
                loss_valid_batch = model(data_valid_batch,val_valid_batch)
                accuracy_valid_batch = model.accuracy
                loss_validation += loss_valid_batch.data
                accuracy_validation += accuracy_valid_batch.data
            iteration_train += max_iter_train
            iteration_validation += max_iter_validation
        loss_train = loss_train / iteration_train
        accuracy_train = accuracy_train / iteration_train
        loss_validation = loss_validation / iteration_validation
        accuracy_validation = accuracy_validation / iteration_validation
        #train_stat.append([loss_train,accuracy_train,loss_validation,accuracy_validation])
        print("train loss, train accuracy, valid loss, valid accuracy: %.3f, %.3f, %.3f, %.3f"\
                     % (loss_train, accuracy_train, loss_validation, accuracy_validation))

        with open(csvfile, 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows([[loss_train,accuracy_train,loss_validation,accuracy_validation]])

        overall_train_loss += loss_train
        overall_train_accuracy += accuracy_train
        overall_validation_loss += loss_validation
        overall_validation_accuracy += accuracy_validation

        if snapshot_interval != 0:
            if (e + 1) % snapshot_interval == 0:
                if gpu_use: model.to_cpu()
                dl_model_save(model,optimizer,model_dir,snapshot=e+1)
                print("snapshot saved")
                if gpu_use: model.to_gpu()

    overall_train_loss = overall_train_loss / epoch
    overall_train_accuracy = overall_train_accuracy / epoch
    overall_validation_loss = overall_validation_loss / epoch
    overall_validation_accuracy = overall_validation_accuracy / epoch

    logger.debug("overall train loss, train accuracy, valid loss, valid accuracy: %.3f, %.3f, %.3f, %.3f",
                 overall_train_loss, overall_train_accuracy, overall_validation_loss, overall_validation_accuracy)

    exec_time = time.time() - exec_time
    time_str = gen_dms_time_str(exec_time)
    logger.debug("exex time:%f (%s)", exec_time, time_str)
    if gpu_use: model.to_cpu()
    dl_model_save(model, optimizer, model_dir)
    print("dl model saved to: %s" % model_dir)
    np.save(os.path.join(model_dir, meanimg_name), mean_image)

if __name__ == "__main__":
    # data, val = load_datapaths("data/vd_bg35_rot_noBING_0.5m")
    # print(data)

    from_scratch = True
    gpu_use = True
    epoch = 10
    batch_learn = 100
    batch_check = 1000
    traindata_ratio = 0.9
    snapshot_interval = 2

    model_dir = "model/vd_bg350_rot_noBING_Adam_batchnorm_Henomal_whole_test"
    data_dir = "data/vd_bg35_rot_noBING_0.5m"

    if (batch_learn > batch_check):
        print("Batchsize setting error.")
        exit(0)
    if from_scratch:
        if not os.path.isdir(model_dir): os.makedirs(model_dir)
    else:
        if not os.path.isdir(model_dir):
            print("No model dir to load.")
            exit(0)

    logfile_name = "cnn_train_curricurum.log"
    logfile_path = os.path.join(model_dir, logfile_name)

    dl_model_scratch = cnn_structure.CNN_batchnorm_Henormal()
    optimizer_scratch = optimizers.Adam()

    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler(logfile_path)
    f_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    dl_drain_curriculum(model_dir, data_dir, batch_learn, batch_check, epoch, snapshot_interval, from_scratch, dl_model_scratch,
                        optimizer_scratch, logger=logger)



