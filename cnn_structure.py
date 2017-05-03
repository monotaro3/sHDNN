#!coding:utf-8

import numpy as np
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers,cuda,serializers
import chainer.functions as F
import chainer.links as L


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
    def __init__(self,train=True):
        super(CNN_dropout1, self).__init__(
            conv1=L.Convolution2D(3, 20, 7),
            conv2=L.Convolution2D(20, 8, 4),
            conv3=L.Convolution2D(8, 8, 4),
            fc=L.Linear(72, 2)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.dropout(F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2),ratio=0.2,train=self.train)
        h2 = F.dropout(F.max_pooling_2d(F.relu(self.conv2(h1)), 2, 2),ratio=0.2,train=self.train)
        h3 = F.dropout(F.max_pooling_2d(F.relu(self.conv3(h2)), 2, 2),ratio=0.2,train=self.train)
        y = self.fc(h3)
        return y

class CNN_dropout2(Chain):
    def __init__(self,train=True):
        super(CNN_dropout2, self).__init__(
            conv1=L.Convolution2D(3, 20, 7),
            conv2=L.Convolution2D(20, 8, 4),
            conv3=L.Convolution2D(8, 8, 4),
            fc1=L.Linear(72,72),
            fc2=L.Linear(72, 2)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2, 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 2, 2)
        h4 = F.dropout(self.fc1(h3),ratio=0.5,train=self.train)
        y = self.fc2(h4)
        return y

class CNN_dropout2_LRN1(Chain):
    def __init__(self, train=True):
        super(CNN_dropout2_LRN1, self).__init__(
            conv1=L.Convolution2D(3, 20, 7),
            conv2=L.Convolution2D(20, 8, 4),
            conv3=L.Convolution2D(8, 8, 4),
            fc1=L.Linear(72, 72),
            fc2=L.Linear(72, 2)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 2, 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2, 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 2, 2)
        h4 = F.dropout(self.fc1(h3), ratio=0.5, train=self.train)
        y = self.fc2(h4)
        return y

class CNN_dropout2_LRN2(Chain):
    def __init__(self, train=True):
        super(CNN_dropout2_LRN2, self).__init__(
            conv1=L.Convolution2D(3, 20, 7),
            conv2=L.Convolution2D(20, 8, 4),
            conv3=L.Convolution2D(8, 8, 4),
            fc1=L.Linear(72, 72),
            fc2=L.Linear(72, 2)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.local_response_normalization(F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2))
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2, 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 2, 2)
        h4 = F.dropout(self.fc1(h3), ratio=0.5, train=self.train)
        y = self.fc2(h4)
        return y

class CNN_dropout2_40_16_16filters(Chain):
    def __init__(self,train=True):
        super(CNN_dropout2_40_16_16filters, self).__init__(
            conv1=L.Convolution2D(3, 40, 7),
            conv2=L.Convolution2D(40, 16, 4),
            conv3=L.Convolution2D(16, 16, 4),
            fc1=L.Linear(144,144),
            fc2=L.Linear(144, 2)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2, 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 2, 2)
        h4 = F.dropout(self.fc1(h3),ratio=0.5,train=self.train)
        y = self.fc2(h4)
        return y

class CNN_batchnorm(Chain):
    def __init__(self,train=True):
        super(CNN_batchnorm, self).__init__(
            conv1=L.Convolution2D(3, 20, 7),
            norm1=L.BatchNormalization(20),
            conv2=L.Convolution2D(20, 8, 4),
            norm2=L.BatchNormalization(8),
            conv3=L.Convolution2D(8, 8, 4),
            norm3=L.BatchNormalization(8),
            fc1=L.Linear(72,72),
            fc2=L.Linear(72, 2)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.norm1(self.conv1(x))), 2, 2)
        h2 = F.max_pooling_2d(F.relu(self.norm2(self.conv2(h1))), 2, 2)
        h3 = F.max_pooling_2d(F.relu(self.norm3(self.conv3(h2))), 2, 2)
        #h4 = F.dropout(self.fc1(h3),ratio=0.5,train=self.train)
        y = self.fc2(h3)
        return y

class CNN_batchnorm_Henormal(Chain):
    def __init__(self,train=True):
        initializer = chainer.initializers.HeNormal()
        super(CNN_batchnorm_Henormal, self).__init__(
            conv1=L.Convolution2D(3, 20, 7,initialW=initializer),
            norm1=L.BatchNormalization(20),
            conv2=L.Convolution2D(20, 8, 4,initialW=initializer),
            norm2=L.BatchNormalization(8),
            conv3=L.Convolution2D(8, 8, 4,initialW=initializer),
            norm3=L.BatchNormalization(8),
            fc1=L.Linear(72,72,initialW=initializer),
            fc2=L.Linear(72, 2,initialW=initializer)
        )
        self.train = train

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.norm1(self.conv1(x))), 2, 2)
        h2 = F.max_pooling_2d(F.relu(self.norm2(self.conv2(h1))), 2, 2)
        h3 = F.max_pooling_2d(F.relu(self.norm3(self.conv3(h2))), 2, 2)
        #h4 = F.dropout(self.fc1(h3),ratio=0.5,train=self.train)
        y = self.fc2(h3)
        return y