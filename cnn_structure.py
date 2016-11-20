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

class vehicle_classify_CNN(Chain):
    def __init__(self):
        super(vehicle_classify_CNN,self).__init__(
            conv1=L.Convolution2D(3,20,7),
            conv2=L.Convolution2D(20,8,4),
            conv3=L.Convolution2D(8,8,4),
            fc = L.Linear(72,2)
        )
    def __call__(self,x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)),2,2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)),2,2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)),2,2)
        y = self.fc(h3)
        return y

def cnn_train():
    modelload = False  # 既存のモデルを読み込んでトレーニング

    model_dir = "model/yangon_vd_161114"
    model_name = "gradient_cnn.npz"
    optimizer_name = "gradient_optimizer.npz"
    logfile_name = "cnn_train.log"

    trainlog_dir = "trainlog"

    data_dir = "data/yangon_vd_161114"
    data_name = "data.npy"
    val_name = "val.npy"
    meanimg_name = "mean_image.npy"

    batchsize = 100
    epoch = 200
    N = 20000 #split data into training and validation

    model_path = os.path.join(model_dir, model_name)
    optimizer_path = os.path.join(model_dir, optimizer_name)
    logfile_path = os.path.join(model_dir, logfile_name)
    data_path = os.path.join(data_dir, data_name)
    val_path = os.path.join(data_dir, val_name)
    meanimg_loadpath = os.path.join(data_dir, meanimg_name)
    trainlog_path = os.path.join(model_dir, trainlog_dir)

    if (not modelload) and os.path.isfile(model_path):
        print("New model training is chosen but a model already exists at the specified location.")
        print("Process aborted.")
        return

    if not os.path.isdir(trainlog_path): os.makedirs(trainlog_path)

    logfile = open(logfile_path, "a")
    date = datetime.now()
    startdate = date.strftime('%Y/%m/%d %H:%M:%S')
    f_startdate = date.strftime('%Y%m%d_%H%M%S')
    exec_time = time.time()

    model = L.Classifier(vehicle_classify_CNN())
    optimizer = optimizers.SGD()
    if modelload:serializers.load_npz(model_path, model)
    optimizer.setup(model)
    if modelload:serializers.load_npz(optimizer_path, optimizer)
    model.to_gpu()

    data = np.load(data_path)
    val = np.load(val_path)
    mean_image = np.load(meanimg_loadpath)

    data -= mean_image

    datasize = int(data.size/48/48/3)

    print("execution:" + startdate)
    print("training data dir:%s" % data_dir)
    print("cnn model dir:%s" % model_dir)
    print("batchsize:%d" % batchsize)
    print("epoch:%d" % epoch)
    print("loaded data size(number of images):%d" %datasize)
    print("N(split number):%d" % N)

    print("execution:" + startdate, file=logfile)
    print("training data dir:%s" % data_dir, file=logfile)
    print("cnn model dir:%s" % model_dir, file=logfile)
    print("batchsize:%d" % batchsize, file=logfile)
    print("epoch:%d" % epoch, file=logfile)
    print("loaded data size(number of images):%d" %datasize, file=logfile)
    print("N(split number):%d" % N, file=logfile)

    data_train ,data_test = np.split(data,[N])
    val_train , val_test = np.split(val,[N])

    train = tuple_dataset.TupleDataset(data_train,val_train)
    test = tuple_dataset.TupleDataset(data_test,val_test)

    train_iter = iterators.SerialIterator(train,batch_size=batchsize)
    test_iter = iterators.SerialIterator(test,batch_size=batchsize,repeat=False,shuffle=False)

    updater = training.StandardUpdater(train_iter,optimizer,device=0)
    trainer = training.Trainer(updater,(epoch,"epoch"),out = trainlog_path)

    trainer.extend(extensions.Evaluator(test_iter,model,device=0))
    trainer.extend(extensions.LogReport(log_name="trainlog" + f_startdate))
    trainer.extend(extensions.PrintReport(["epoch","main/accuracy","validation/main/accuracy"]))
    trainer.extend(extensions.ProgressBar())

    #save snapshot of model and optimizer
    snapshot_interval = (200,"epoch")
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(optimizer, 'optimizer_epoch_{.updater.epoch}'), trigger=snapshot_interval)

    trainer.run()

    model.to_cpu()
    serializers.save_npz(model_path, model)
    serializers.save_npz(optimizer_path, optimizer)

    np.save(os.path.join(model_dir, meanimg_name), mean_image) #平均画像の保存
    root, ext = os.path.splitext(meanimg_name)
    meanimg_savepath = root + f_startdate + ext
    meanimg_savepath = os.path.join(model_dir, meanimg_savepath)
    np.save(meanimg_savepath, mean_image)

    genGraph(trainlog_path)

    exec_time = time.time() - exec_time
    print("exex time:%f sec"% exec_time)
    print("exex time:%f" % exec_time,file=logfile)

    logfile.close()

if __name__ == "__main__":
    cnn_train()