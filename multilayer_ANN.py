# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:46:26 2018

@author: ko936
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import requests 
from tensorflow.python.framework import ops

#数据集名称
birth_weight_file = 'birth_weight.csv'

#如果当前文件夹下没有birth_weight.csv的数据集则下载dat文件并生成csv文件
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    #split分割函数，以一行作为分割函数，windows中换行符号为'\r\n'，每一行后面都有一个
    birth_data = birth_file.text.split('\r\n')
    #每一列的标题，标在第一行，即birth_data的第一个数据，并使用制表符作为划分
    birth_header = birth_data[0].split('\t')
    #数组的第一维表示遍历行从第一行开始，所以不包括标题，数组第二维遍历列（使用制表符进行分割）
    birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
    #此为list数据形式，不是numpy数组，因此不能使用np.shape函数，但是我们可以使用np.array函数将list对象转化为numpy数组后使用shape属性进行查看
    print(np.array(birth_data).shape)
    #注意，向其中写入文件时一定要去掉换行等操作符号，如果在csv中没有换行符，也会作为一行数据
    #读文件时，把csv文件读入列表中，写文件时会把列表中的元素写入到csv文件中
    with open(birth_weight_file,'w',newline = '') as f:
        #创建当前目录下birth_weight.csv文件
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(birth_data)
        f.close()
#将出生体重数据读进内存
birth_data = []
with open(birth_weight_file,newline = '') as csvfile:
    #用csv.reader读取csvfile中的文件
    csv_reader = csv.reader(csvfile)
    #读取第一行每一列的标题
    birth_header = next(csv_reader)
    #数据保存到birth_data中
    for row in csv_reader:
        birth_data.append(row)
#数据转换成浮点格式
birth_data = [[float(x) for x in row] for row in birth_data]
#对每组数据而言，第8列（序列0开始）及为标签序列-体重
y_valus = np.array([x[8] for x in birth_data])
#特征序列
colums_of_interest = ['AGE','LWT','RACE','SMOKE','PTL','HT','UI']
x_valus = np.array([[x[ix] for ix,feature in enumerate(birth_header) if feature in colums_of_interest] for x in birth_data])
