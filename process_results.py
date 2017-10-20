#!/usr/bin/python
#
import sys
import numpy as np
#
f = open(sys.argv[1], "r")
#
#SOB_B_A-14-22549G-100-030-550-150.png;0;0.129226;0.004267;0.007883;0.004518;0.175048;0.000307;0.677721;0.001031;
#
t = 0
Y = []
W = []
Z = []
#
for i in f:
    if(t == 0):
        t += 1
        continue
    else:
        linha = i.split(";")
        Y.append(int(linha[1]))
        wtmp = list()
        for j in linha[2:-1]:
            wtmp.append(float(j))
        W.append(wtmp)
        Z.append(linha[0])
#
f.close()
#
#print(len(Y), len(W), len(Z))
#
pac_vot = {}
img_vot = {}
pac = {}
img = {}
#
correct = 0
total = 0
for i in range(len(Y)):
    if(Y[i] == np.argmax(W[i])):
        correct += 1
    total += 1
e = float(correct)/total
#
for i in range(len(Y)):
    img_name = Z[i].split("-")
    pac_str = img_name[0]+"-"+img_name[1]+"-"+img_name[2]
    img_str = img_name[0]+"-"+img_name[1]+"-"+img_name[2]+"-"+img_name[3]+"-"+img_name[4]
    if(img_str in img):
        a = np.add(img[img_str][1], W[i])
        img[img_str][1] = a
        img[img_str][2][np.argmax(W[i])] += 1
    else:
        a = [0 for j in range(len(W[i]))]
        np_a = np.array(a)
        np.add(np_a, W[i])
        np_b = np.array(a)
        np_b[np.argmax(W[i])] += 1
        # 1 - soma
        # 2 - voto
        img[img_str] = [Y[i], np_a, np_b]
#
img_correto_sum = 0
img_correto_vot = 0
img_total = 0
for i in img:
    #print(i)
    if(img[i][0] == np.argmax(img[i][1])):
        img_correto_sum += 1
    if(img[i][0] == np.argmax(img[i][2])):
        img_correto_vot += 1
    img_total += 1
a = float(img_correto_sum)/img_total
b = float(img_correto_vot)/img_total
#
pac = {}
#
for i in img:
    pac_name = i.split("-")
    pac_str = pac_name[0]+"-"+pac_name[1]+"-"+pac_name[2]
    correto_sum = 0
    correto_vot = 0
    if(img[i][0] == np.argmax(img[i][1])):
        correto_sum = 1
    if(img[i][0] == np.argmax(img[i][2])):
        correto_vot = 1

    if(pac_str in pac):
        pac[pac_str][0] += correto_sum
        pac[pac_str][1] += correto_vot
        pac[pac_str][2] += 1
    else:
        pac[pac_str] = [0,0,0]
        pac[pac_str][0] = correto_sum
        pac[pac_str][1] = correto_vot
        pac[pac_str][2] = 1
#
media_sum = 0
media_vot = 0
#
t = 0
#
for i in pac:
    t += 1
    media_sum += float(pac[i][0])/pac[i][2]
    media_vot += float(pac[i][1])/pac[i][2]
    #print("{} {} {} {}".format(i, pac[i][0], pac[i][1], pac[i][2] ))
#
c = float(media_sum)/t
d = float(media_vot)/t
#
print("{};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f}".format(sys.argv[1], e, a, b, c, d))

