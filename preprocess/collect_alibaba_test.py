import os
import sys
import csv
import numpy as np
import string
from decimal import Decimal
from tqdm import tqdm

#The raw data path. Please change the path
ROOT_dir = "/media/data/TestSet_2/"
filelist_pts = os.listdir(ROOT_dir + "pts_2")
filelist_intensity = os.listdir(ROOT_dir + "intensity_2")
# filelist_category = os.listdir(ROOT_dir + "category")

#遍历每个文件
for index in tqdm(range(len(filelist_pts))): #len(filelist_pts)
    file_name = filelist_pts[index]

    # temp = open(ROOT_dir + "pts/" + file_name,"r")
    # rows = len(temp.readlines())

    X,Y,Z,I,L = [],[],[],[],[]
    #pts
    csv_pts = csv.reader(open(ROOT_dir+"pts_2/"+file_name))
    for row in csv_pts:
        x = float(row[0])#保持精度 str2float 精度不够 Decimal()精度够 但是gen_indoor3d 不同意
        y = float(row[1])
        z = float(row[2])
        X.append(x)
        Y.append(y)
        Z.append(z)

    #intensity
    csv_intensity = csv.reader(open(ROOT_dir + "intensity_2/" + file_name))
    for row in csv_intensity:
        intensity = round(float(row[0])*255) # 0-1 -> 0-255  精度会变一点点
        I.append(intensity)

    #category


    #output
    data_output = np.concatenate(([X],[Y],[Z],[I],[I],[I],[[0]*len(X)]),0).T # should be NX7
    # print(data_output.shape[0])
    # print(data_output.shape[1])
    # print(data_output[0])
    xyz_min = np.amin(data_output, axis=0)[0:3]
    # print(xyz_min)
    data_output[:, 0:3] -= xyz_min

    file_prefix_name,_ =os.path.splitext(file_name)
    out_filename = file_prefix_name + ".npy"
    # Path to save the generate data. Please change the path
    np.save(ROOT_dir+"test_2/"+out_filename,data_output)
