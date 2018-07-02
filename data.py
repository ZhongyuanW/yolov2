# encoding: utf-8  

""" 
@version: v1.0 
@author: ZhongyuanWu
@contact: 806282013@qq.com 
@site: Chongqing University 
@software: PyCharm 
@file: data.py 
@time: 2018/6/29 10:50 
"""

import config as cfg
import os
import numpy as np
import random
from xml.etree import ElementTree as et
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class data(object):
    def __init__(self,data_path = cfg.DATA_XML_PATH):
        self.filenames = os.listdir(data_path)
        self.filenames = np.array(self.filenames)
        random.shuffle(self.filenames)
        self.len = len(self.filenames)
        self.curr = 0


    def read(self,batch_size = cfg.BATCH_SIZE):
        #完成一个epoch就随机混乱数据集
        if(self.curr+batch_size>self.len):
            self.curr = 0
            random.shuffle(self.filenames)
        #读取数据
        batch_images = np.zeros((batch_size,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,3),np.float32)
        batch_labels = np.zeros((batch_size,cfg.MAX_NUMS,5),np.float32)
        for curr in range(self.curr,self.curr+batch_size):
            image,label = xml_parse(os.path.join(cfg.DATA_XML_PATH,self.filenames[curr]))
            batch_images[curr-self.curr] = image
            batch_labels[curr-self.curr] = label

        return batch_images,batch_labels


    def read_test(batch_size = cfg.BATCH_SIZE):
        return None

##解析xml文件
## return label:[x_center,y_center,w,h,class]
def xml_parse(xml_path):
    #label
    labels = np.zeros((cfg.MAX_NUMS,5),np.float32)

    #xml解析
    xml = et.parse(xml_path)

    #读取image
    image_name = list(xml.findall("./filename"))[0].text
    image = cv2.imread(os.path.join(cfg.DATA_ROOT_PATH,'JPEGImages',image_name))

    #放缩
    y_scale = cfg.IMAGE_SIZE / np.shape(image)[0]
    x_scale = cfg.IMAGE_SIZE / np.shape(image)[1]
    image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    image = np.array(image,np.float32)
    #最大最小值归一化
    image = (image-np.min(image))/(np.max(image) - np.min(image))

    #读取label
    objects = list(xml.findall("./object"))
    for i,obj in enumerate(objects):
        label = []
        label = np.array(label)
        #类别
        cls = obj[0].text
        Class = cfg.CLASSES.index(cls)

        #坐标
        bndbox = obj[4]
        x_min = int(bndbox[0].text) * x_scale
        y_min = int(bndbox[1].text) * y_scale
        x_max = int(bndbox[2].text) * x_scale
        y_max = int(bndbox[3].text) * y_scale

        x_center = (x_min + x_max) * 0.5
        y_center = (y_min + y_max) * 0.5
        w = x_max - x_min
        h = y_max - y_min

        label = np.append(label,[x_center,y_center,w,h,Class])
        labels[i] = label
    return image,labels



if __name__ == "__main__":
    data = data()
    imgs,labels = data.read(batch_size=2)
    #img,label =  xml_parse("D:\\Pascal VOC 2007\\VOCdevkit\\VOC2007\\Annotations\\000001.xml")
    print(labels[1])
