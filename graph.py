#!coding:utf-8

import matplotlib.pyplot as plt

f = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016100717_35t_1000/log","r")

#f2 = open("C:/work/PycharmProjects/gradient_slide_cnn/model/2016090901_1000/log","r")
#test2_2

f.readline()
i = 0
n = []
line = f.readline()
while(line):
    if i % 8 == 5:
        line = line[35:-2]
        n.append(float(line))
    line = f.readline()
    i += 1
#
# f2.readline()
# i = 0
# m = []
# line = f2.readline()
# while(line):
#     if i % 8 == 5:
#         line = line[25:-2]
#         m.append(float(line))
#     line = f2.readline()
#     i += 1
#
# m.extend(n)
print(len(n))
print(n[1000:1100])

plt.title("Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.plot(n)
plt.show()

