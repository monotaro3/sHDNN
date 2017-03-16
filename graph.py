#!coding:utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import logging

def genGraph(log_dir, x_interval_rescale_divide = 1):
    logger = logging.getLogger("__main__." + __name__)

    tmp = os.listdir(log_dir)
    tmp = sorted([os.path.join(log_dir, x) for x in tmp if os.path.isfile(os.path.join(log_dir, x))])
    logfiles = []
    for t in tmp:
        root, exe = os.path.splitext(t)
        if exe == ".trainlog":
            logfiles.append(t)

    if len(logfiles) == 0:
        logger.error("No log files.")
        return

    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []

    acc_fig_name = "accuracy.png"
    loss_fig_name = "valid_accuracy.png"

    graph_dir = os.path.join(log_dir, "graph")
    if not os.path.isdir(graph_dir): os.makedirs(graph_dir)

    acc_fig_path = os.path.join(graph_dir, acc_fig_name)
    loss_fig_path = os.path.join(graph_dir, loss_fig_name)

    #f = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016100717_35t_1000/log","r")
    #f2 = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016090901_1000/log","r")

    for logfile in logfiles: # for training accuracy
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

    x_values =[]
    for i in range(len(train_acc)):
        x_values.append((i+1)/x_interval_rescale_divide)

    plt.title("Training / Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.plot(x_values,train_acc, label="Training")
    plt.plot(x_values,valid_acc, label="Validation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(acc_fig_path)
    #plt.show()

    plt.figure()
    plt.title("Training / Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.plot(x_values, train_loss, label="Training")
    plt.plot(x_values, valid_loss, label="Validation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(loss_fig_path)

    logger.debug("Graph was generated succesfully.")

if __name__ == "__main__":
    log_dir = "C:/work/sHDNN/model/vd_bg350_rot_noBING_Adam_dropout2_whole/trainlog"

    logger = logging.getLogger(__name__)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    #f_handler = logging.FileHandler(os.path.join(log_dir,"graph.log"))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    #logger.addHandler(f_handler)

    genGraph(log_dir,x_interval_rescale_divide=7)

