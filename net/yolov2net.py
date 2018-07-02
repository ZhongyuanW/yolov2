#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: ZhongyuanWu
@contact: 806282013@qq.com
@site: 
@software: PyCharm
@file: yolov2net.py
@time: 2018/6/24 0:13
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg
import numpy as np
import cv2
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

def inferece(x,scope="yolov2"):
    # 输入：x, [batch,width,height,channel]
    # 输出：y, [batch,GRID_W,GRID_H,(cfg.CLASSES_NUMS+5)*cfg.ANCHOR_NUMS]
    with tf.variable_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
            activation_fn=leaky_relu(),
            weights_regularizer=slim.l2_regularizer(cfg.L2_REGULARIZER),
                normalizer_fn=slim.batch_norm,
        ):
            conv0 = slim.conv2d(x,32,3,1,scope='conv_0')
            pool1 = slim.max_pool2d(conv0, 2, 2, scope='pool_1')
            conv2 = slim.conv2d(pool1, 64, 3, 1, scope='conv_2')
            pool3 = slim.max_pool2d(conv2, 2, 2, scope='pool_3')
            conv4 = slim.conv2d(pool3, 128, 3, 1, scope='conv_4')
            conv5 = slim.conv2d(conv4, 64, 1, 1, scope='conv_5')
            conv6 = slim.conv2d(conv5, 128, 3, 1, scope='conv_6')
            pool7 = slim.max_pool2d(conv6, 2, 2, scope='pool_7')
            conv8 = slim.conv2d(pool7, 256, 3, 1, scope='conv_8')
            conv9 = slim.conv2d(conv8, 128, 1, 1, scope='conv_9')
            conv10= slim.conv2d(conv9, 256, 3, 1, scope='conv_10')
            pool11= slim.max_pool2d(conv10, 2, 2, scope='pool_11')
            conv12= slim.conv2d(pool11, 512, 3, 1, scope='conv_12')
            conv13= slim.conv2d(conv12, 256, 1, 1, scope='conv_13')
            conv14= slim.conv2d(conv13, 512, 3, 1, scope='conv_14')
            conv15= slim.conv2d(conv14, 256, 1, 1, scope='conv_15')
            conv16= slim.conv2d(conv15, 512, 3, 1, scope='conv_16')
            pool17= slim.max_pool2d(conv16, 2, 2, scope='pool_17')
            conv18= slim.conv2d(pool17, 1024, 3, 1, scope='conv_18')
            conv19= slim.conv2d(conv18, 512, 1, 1, scope='conv_19')
            conv20= slim.conv2d(conv19, 1024, 3, 1, scope='conv_20')
            conv21= slim.conv2d(conv20, 512, 1, 1, scope='conv_21')
            conv22= slim.conv2d(conv21, 1024, 3, 1, scope='conv_22')
            conv23= slim.conv2d(conv22, 1024, 3, 1, scope='conv_23')
            conv24= slim.conv2d(conv23, 1024, 3, 1, scope='conv_24')
            #conv25= conv16
            conv26= slim.conv2d(conv16, 64, 1, 1, scope='conv_25')
            #隔行取样
            conv27= tf.space_to_depth(conv26,2,name='reorg_27')
            #融合第24层和27层特征图
            conv28= tf.concat([conv27,conv24],axis=3,name='concat_28')
            conv29= slim.conv2d(conv28, 1024, 3, 1, scope='conv_29')
            conv30= slim.conv2d(conv29,(cfg.CLASSES_NUMS+5)*cfg.ANCHOR_NUMS,1,1,scope='conv30')

            y = tf.reshape(conv30,shape=(-1,cfg.GRID_SIZE,cfg.GRID_SIZE,cfg.ANCHOR_NUMS,cfg.CLASSES_NUMS+5),name='y')

            #还原真实坐标, 相对于GRID X GRID
            anchors = tf.constant(cfg.ANCHORS, dtype=tf.float32)

            temp = tf.ones(shape=[cfg.GRID_SIZE,cfg.GRID_SIZE],dtype=tf.float32)
            temp *= [x for x in range(cfg.GRID_SIZE)]
            temp = tf.reshape(temp,shape=[cfg.GRID_SIZE,cfg.GRID_SIZE,1,1])
            temp_x = tf.tile(temp,[1,1,cfg.ANCHOR_NUMS,1])
            temp_y = tf.transpose(temp_x,(1,0,2,3))

            center_x = tf.sigmoid(y[:, :, :, :, 0:1]) + temp_x
            center_y = tf.sigmoid(y[:, :, :, :, 1:2]) + temp_y

            w = anchors[:, 0:1] * tf.exp(y[:, :, :, :, 2:3])
            h = anchors[:, 1:2] * tf.exp(y[:, :, :, :, 3:4])
            f = y[:,:,:,:,4:5]
            c = tf.nn.softmax(y[:,:,:,:,5:])
            net = tf.concat([center_x,center_y,w,h,f,c],axis=4)

            return net


def leaky_relu(alpha=cfg.ALPHA):
    # leaky_relu激活函数
    # 输入: alpha, float
    def op(inputs):
        return tf.nn.leaky_relu(inputs,alpha=alpha,name="leaky_relu")


def iou(boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                       boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
    #TODO 这里坐标转换好像有问题
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

    boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,

                       boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

    # calculate the left up point
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    # intersection
    intersection = rd - lu
    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
    inter_square = mask * inter_square

    # calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    return inter_square / (square1 + square2 - inter_square + 1e-6)

def losses(preduces,labels):
    #TODO 完善损失函数
    #损失函数计算
    #输入: preduces, 网络预测值, [batch,GRID_W,GRID_H,ANCHOR_NUMS,~]
    #     labels, 样本标签, [batch,max_nums,5], 5代表x,y,w,h,class
    #输出: total_loss, 总的损失值, float, 包括 坐标损失、含有obj的置信度损失、不含obj的置信度损失、分类损失

    total_loss = tf.constant(0,dtype=tf.float32)
    for i in range(cfg.BATCH_SIZE):
        starttime = time.time()
        label = labels[i]
        total_loss += get_loss(i,preduces,label)
        endtime = time.time()
        print(i," 次运行时间: ", (endtime - starttime))

    total_loss /= cfg.BATCH_SIZE
    return total_loss

def get_loss(i,preduces,label):
    total_loss = tf.constant(0, dtype=tf.float32,name = "total_loss")
    for j in range(cfg.MAX_NUMS):
        obj = label[j]
        # 判断是否
        if obj[0] == obj[1] == obj[2] == obj[3] == obj[4] == 0.0:
            break

        ### 1. 计算object, [GRID_SIZE,GRID_SIZE], 其中1的地方表示有obj存在,0的地方表示没有ibj存在 ###

        min_x = (obj[0] - obj[2] / 2) / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)
        max_x = (obj[0] + obj[2] / 2) / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)

        min_y = (obj[1] - obj[3] / 2) / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)
        max_y = (obj[1] + obj[3] / 2) / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)

        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)

        max_x = tf.ceil(max_x)
        max_y = tf.ceil(max_y)

        # 表示有obj的矩阵
        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        object = tf.ones(temp, tf.float32)

        # 填充没有obj的地方
        temp = tf.cast(tf.stack([min_y, cfg.GRID_SIZE - max_y, min_x, cfg.GRID_SIZE - max_x]), dtype=tf.int32)
        temp = tf.reshape(temp, (2, 2))
        object = tf.pad(object, temp, 'CONSTANT')

        # return object

        ### 2. 计算response, [GRID_SIZE,GRID_SIZE], 只有一个格子存在1, 表示对该obj负责, 其他的都为0  ###
        center_x = obj[0] / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)
        center_x = tf.floor(center_x)

        center_y = obj[1] / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack([center_y, cfg.GRID_SIZE - center_y - 1, center_x, cfg.GRID_SIZE - center_x - 1]),
                       tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, 'CONSTANT')

        # return response

        ### 3. 计算IOU  ###
        preduces_boxes = preduces[i, :, :, :, 0:4]

        preduces_boxes = preduces_boxes * [(cfg.IMAGE_SIZE / cfg.GRID_SIZE), (cfg.IMAGE_SIZE / cfg.GRID_SIZE),
                                           (cfg.IMAGE_SIZE / cfg.GRID_SIZE), (cfg.IMAGE_SIZE / cfg.GRID_SIZE)]

        iou_predict_truth = iou(preduces_boxes, obj[0:4])

        C = iou_predict_truth * tf.reshape(response, (cfg.GRID_SIZE, cfg.GRID_SIZE, 1))
        # return C

        ### 4. 计算每个网格内对物体负责的anchor ###
        I = iou_predict_truth * tf.reshape(response, (cfg.GRID_SIZE, cfg.GRID_SIZE, 1))

        max_I = tf.reduce_max(I, 2, keep_dims=True)
        # return max_I

        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (cfg.GRID_SIZE, cfg.GRID_SIZE, 1))
        # return I

        no_I = tf.ones_like(I, tf.float32) - I
        # return no_I

        # 每个anchor的置信度 [GRID,GRID,ANCHOR_NUMS]
        p_C = preduces[i, :, :, :, 4]
        # return p_C

        # 计算真实的x,y,sqrt_w,sqrt_h
        x = obj[0]
        y = obj[1]
        sqrt_w = tf.sqrt(tf.abs(obj[2]))
        sqrt_w = tf.cast(sqrt_w, dtype=tf.float32)
        # return sqrt_w
        sqrt_h = tf.sqrt(tf.abs(obj[3]))
        sqrt_h = tf.cast(sqrt_h, dtype=tf.float32)

        # 计算预测的p_x,p_y,p_w,p_h, [GRID,GRID,ANCHOR_NUMS]
        p_x = preduces_boxes[:, :, :, 0]
        p_y = preduces_boxes[:, :, :, 1]
        p_sqrt_w = tf.sqrt(tf.minimum(cfg.IMAGE_SIZE * 1.0, tf.maximum(0.0, preduces_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(cfg.IMAGE_SIZE * 1.0, tf.maximum(0.0, preduces_boxes[:, :, :, 3])))

        # 计算真实的分类
        P = tf.one_hot(tf.cast(obj[4], tf.int32), cfg.CLASSES_NUMS, dtype=tf.float32)

        # 计算预测的分类 [GRID,GRID,ANCHOR_NUMS,CLASSES_NUMS]
        p_P = preduces[i, :, :, :, 5:]

        ###   计算loss  ###
        # 分类误差
        class_loss = tf.nn.l2_loss(tf.tile(tf.reshape(object, (cfg.GRID_SIZE, cfg.GRID_SIZE, 1, 1)),
                                           (1, 1, cfg.ANCHOR_NUMS, cfg.CLASSES_NUMS)) * (p_P - P)) * cfg.CLASS_SCALE

        # 含有obj的置信度误差
        object_loss = tf.nn.l2_loss(I * (p_C - C)) * cfg.OBJECT_SCALE

        # 不含obj的置信度误差
        noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * cfg.NOOBJECT_SCALE

        # 坐标误差
        # TODO 这里好像有点问题
        coor_loss = (tf.nn.l2_loss(I * (p_x - x) / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)) +
                     tf.nn.l2_loss(I * (p_y - y) / (cfg.IMAGE_SIZE / cfg.GRID_SIZE)) +
                     tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / cfg.IMAGE_SIZE +
                     tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h)) / cfg.IMAGE_SIZE) * cfg.COORD_SCALE

        total_loss += class_loss + object_loss + noobject_loss + coor_loss
    return total_loss


if __name__ == "__main__":
    x = tf.placeholder(dtype=tf.float32,shape=[None,608,608,3],name='x')
    y = tf.placeholder(dtype=tf.float32,shape=[None,cfg.MAX_NUMS,5],name="y")
    preduce = inferece(x)

    total_loss = losses(preduce,y)
