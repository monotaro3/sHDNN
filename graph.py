#!coding:utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from scipy import signal
import csv

def genGraph(log_dir, x_interval_rescale_divide = 1,fig_size = (8,6), mode = 0):

    smoothing = True

    range_limit = False
    limit_acc = [0.990,1]
    limit_loss = [0,0.12]

    logger = logging.getLogger("__main__." + __name__)

    tmp = os.listdir(log_dir)
    tmp = sorted([os.path.join(log_dir, x) for x in tmp if os.path.isfile(os.path.join(log_dir, x))])
    logfiles = []
    for t in tmp:
        root, exe = os.path.splitext(t)
        if mode == 0:
            if exe == ".trainlog":
                logfiles.append(t)
        elif mode == 1:
            if exe == ".csv":
                logfiles.append(t)

    if len(logfiles) == 0:
        logger.error("No log files.")
        return

    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []

    acc_fig_name = "accuracy.png"
    loss_fig_name = "loss.png"
    acc_fig_name_smoothed = "accuracy_smoothed.png"
    loss_fig_name_smoothed = "loss_smoothed.png"

    graph_dir = os.path.join(log_dir, "graph")
    if not os.path.isdir(graph_dir): os.makedirs(graph_dir)

    acc_fig_path = os.path.join(graph_dir, acc_fig_name)
    loss_fig_path = os.path.join(graph_dir, loss_fig_name)
    acc_fig_path_smoothed = os.path.join(graph_dir, acc_fig_name_smoothed)
    loss_fig_path_smoothed = os.path.join(graph_dir, loss_fig_name_smoothed)

    #f = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016100717_35t_1000/log","r")
    #f2 = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016090901_1000/log","r")

    for logfile in logfiles: # for training accuracy
        if mode == 0:
            f = open(logfile, "r")
            try:
                line = f.readline()
            except:
                logger.error("can't read line")
                return

            while(line):
                start = line.find(":")
                if start != -1:
                    start += 1
                    end = line.find(",")
                    if end == -1:
                        value = float(line[start:])
                    else:
                        value = float(line[start:end])
                    if line.find("\"main/accuracy\"") > -1:
                        train_acc.append(value)
                    elif line.find("\"validation/main/accuracy\"") > -1:
                        valid_acc.append(value)
                    elif line.find("\"main/loss\"") > -1:
                        train_loss.append(value)
                    elif line.find("\"validation/main/loss\"") > -1:
                        valid_loss.append(value)
                line = f.readline()
        elif mode == 1:
            with open(logfile, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for epoch in reader:
                    train_loss.append(epoch[0])
                    train_acc.append(epoch[1])
                    valid_loss.append(epoch[2])
                    valid_acc.append(epoch[3])

    #leveling by Savitzky-Golay filter
    if smoothing:
        train_acc_smoothed = signal.savgol_filter(np.array(train_acc), 51, 3)
        valid_acc_smoothed = signal.savgol_filter(np.array(valid_acc), 51, 3)
        train_loss_smoothed = signal.savgol_filter(np.array(train_loss), 51, 3)
        valid_loss_smoothed = signal.savgol_filter(np.array(valid_loss), 51, 3)

    x_values =[]
    for i in range(len(train_acc)):
        x_values.append((i+1)/x_interval_rescale_divide)

    plt.figure(figsize=fig_size)
    plt.title("Training / Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    if range_limit: plt.ylim(limit_acc)
    plt.plot(x_values,train_acc, label="Training")
    plt.plot(x_values,valid_acc, label="Validation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(acc_fig_path)
    #plt.show()

    plt.figure(figsize=fig_size)
    plt.title("Training / Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    if range_limit: plt.ylim(limit_loss)
    plt.plot(x_values, train_loss, label="Training")
    plt.plot(x_values, valid_loss, label="Validation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(loss_fig_path)

    if smoothing:
        plt.figure(figsize=fig_size)
        plt.title("Training / Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        if range_limit: plt.ylim(limit_acc)
        plt.plot(x_values, train_acc_smoothed, label="Training")
        plt.plot(x_values, valid_acc_smoothed, label="Validation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.7)
        plt.savefig(acc_fig_path_smoothed)
        # plt.show()

        plt.figure(figsize=fig_size)
        plt.title("Training / Validation Loss")
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        if range_limit: plt.ylim(limit_loss)
        plt.plot(x_values, train_loss_smoothed, label="Training")
        plt.plot(x_values, valid_loss_smoothed, label="Validation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.7)
        plt.savefig(loss_fig_path_smoothed)

    logger.debug("Graph was generated succesfully.")

if __name__ == "__main__":
    log_dir = "C:/work/sHDNN/model/HEM_test/test1_old_n/trainlog"

    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    #f_handler = logging.FileHandler(os.path.join(log_dir,"graph.log"))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    #logger.addHandler(f_handler)

    genGraph(log_dir,x_interval_rescale_divide=1,fig_size=(20,6),mode = 1)

