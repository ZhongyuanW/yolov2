#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: ZhongyuanWu
@contact: 806282013@qq.com
@site: 
@software: PyCharm
@file: config.py
@time: 2018/6/24 1:00
"""

ALPHA = 0.2
L2_REGULARIZER = 0.005

IMAGE_SIZE = 448

CLASSES_NUMS = 20
ANCHOR_NUMS = 5
GRID_SIZE = 14

CLASS_SCALE = 1
OBJECT_SCALE = 5
NOOBJECT_SCALE = 0.1
COORD_SCALE = 1

BATCH_SIZE = 64
MAX_NUMS = 30
MAX_TRAIN_STEP = 45000

SAVE_PATH = "model\\yolov2_model.ckpt"
DATA_XML_PATH = "D:\wuzhongyuan\Pascal VOC 2007\VOCdevkit\VOC2007\Annotations"
DATA_ROOT_PATH = "D:\wuzhongyuan\Pascal VOC 2007\VOCdevkit\VOC2007"

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',             #目标类别
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

GPU = "3,4,5,7"

ANCHORS = [[1,2],[3,4],[5,6],[7,8],[9,10]]
LEARNING_RATE = 1e-1