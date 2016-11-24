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

class CNN_dropout1(Chain):
    def __init__(self):
        super(CNN_dropout1, self).__init__(
            conv1=L.Convolution2D(3, 20, 7),
            conv2=L.Convolution2D(20, 8, 4),
            conv3=L.Convolution2D(8, 8, 4),
            fc=L.Linear(72, 2)
        )

    def __call__(self, x):
        h1 = F.dropout(F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2),ratio=0.2)
        h2 = F.dropout(F.max_pooling_2d(F.relu(self.conv2(h1)), 2, 2),ratio=0.2)
        h3 = F.dropout(F.max_pooling_2d(F.relu(self.conv3(h2)), 2, 2),ratio=0.2)
        y = self.fc(h3)
        return y