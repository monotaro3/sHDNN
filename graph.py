#!coding:utf-8

import matplotlib.pyplot as plt
import os

def genGraph(log_dir):

    tmp = os.listdir(log_dir)
    logfiles = sorted([os.path.join(log_dir, x) for x in tmp if os.path.isfile(os.path.join(log_dir, x))])

    train_acc = []
    valid_acc = []

    fig_name = "accuracy.png"
    valid_fig_name = "valid_accuracy.png"

    graph_dir = os.path.join(log_dir, "graph")
    if not os.path.isdir(graph_dir): os.makedirs(graph_dir)

    fig_path = os.path.join(graph_dir, fig_name)
    valid_fig_path = os.path.join(graph_dir, valid_fig_name)

    #f = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016100717_35t_1000/log","r")
    #f2 = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016090901_1000/log","r")

    for logfile in logfiles: # for training accuracy
        f = open(logfile, "r")
        line = f.readline()
        while(line):
            if line.find("\"main/accuracy\"") > -1:
                start = line.find(":") + 1
                end = line.find(",")
                if end == -1:
                    train_acc.append(float(line[start:]))
                else:
                    train_acc.append(float(line[start:end]))
            line = f.readline()

    for logfile in logfiles: # for validation accuracy
        f = open(logfile, "r")
        line = f.readline()
        while(line):
            if line.find("\"validation/main/accuracy\"") > -1:
                start = line.find(":") + 1
                end = line.find(",")
                if end == -1:
                    valid_acc.append(float(line[start:]))
                else:
                    valid_acc.append(float(line[start:end]))
            line = f.readline()

    plt.title("Training / Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.plot(train_acc, label="Training")
    plt.plot(valid_acc, label="Validation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.7)
    plt.savefig(fig_path)
    #plt.show()

if __name__ == "__main__":
    log_dir = "C:/work/sHDNN/model/yangon_vd_161114/trainlog"
    genGraph(log_dir)

