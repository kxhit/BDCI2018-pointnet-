import os
import sys
import csv
import numpy as np
import string
# from decimal import Decimal
import time

#The txt data path. Please change the path
filelist = ['/media/data/csc105/孔/单帧/cloud100_b.txt']
#计时开始
time_start=time.time()

#遍历每个文件
for index in range(len(filelist)):
    file_name = filelist[index]

    X,Y,Z,I,L = [],[],[],[],[]

    f = np.loadtxt(file_name)
    print(f[0])
    xyz_min = np.amin(f, axis=0)[0:3]
    print('xyz_min')
    print(xyz_min)
    kx
    #pts
    csv_pts = csv.reader(open(ROOT_dir+"pts/"+file_name))
    for row in csv_pts:
        x = float(row[0])
        y = float(row[1])
        z = float(row[2])
        X.append(x)
        Y.append(y)
        Z.append(z)

    #intensity
    csv_intensity = csv.reader(open(ROOT_dir + "intensity/" + file_name))
    for row in csv_intensity:
        intensity = round(float(row[0])*255) # 0-1 -> 0-255  精度会变一点点
        I.append(intensity)

    #category
    csv_category = csv.reader(open(ROOT_dir + "category/" + file_name))
    for row in csv_category:
        category = int(row[0])  # 0-1 -> 0-255  精度会变一点点
        L.append(category)

    #output
    data_output = np.concatenate(([X],[Y],[Z],[I],[I],[I],[L]),0).T # should be NX7
    # print(data_output.shape[0])
    # print(data_output.shape[1])
    # print(data_output[0])
    xyz_min = np.amin(data_output, axis=0)[0:3]
    # print(xyz_min)
    data_output[:, 0:3] -= xyz_min

    file_prefix_name,_ =os.path.splitext(file_name)
    out_filename = file_prefix_name + ".npy"
    #Path to save the generate data. Please change the path
    train_dir = "/media/training/"
    np.save(train_dir+"train/"+out_filename,data_output) #磁盘大小不够 换个磁盘

#计时结束
time_end=time.time()
print('totally cost',time_end-time_start)
