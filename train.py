# encoding: utf-8  

""" 
@version: v1.0 
@author: ZhongyuanWu
@contact: 806282013@qq.com 
@site: Chongqing University 
@software: PyCharm 
@file: train.py 
@time: 2018/6/29 10:23 
"""

import tensorflow as tf
import net.yolov2net as yolonet
import config as cfg
import re
import data
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

data = data.data()

def train(model_path = None):

    starttime = time.time()

    x = tf.placeholder(tf.float32,[None,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,3])
    labels = tf.placeholder(tf.float32,[None,cfg.MAX_NUMS,5])
    tf.summary.image("input",x)

    net = yolonet.inferece(x)

    nettime = time.time()
    print("---------1.构建net花费 %f ---------------\n"%(nettime-starttime))

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=10)

        total_loss = yolonet.losses(net,labels)
        losstime = time.time()
        print("---------2.构建loss花费 %f ---------------\n" % (losstime - nettime))
        tf.summary.scalar("loss",total_loss)

        train_step = tf.train.AdamOptimizer(cfg.LEARNING_RATE).minimize(total_loss)
        traintime = time.time()
        print("---------3.反向传播花费 %f ---------------\n" % (traintime - losstime))

        # TODO 完成精确度的计算
        accuracy = tf.constant(0,tf.float32)

        tf.summary.scalar("accuracy",accuracy)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("log",sess.graph)
        test_writer = tf.summary.FileWriter("log", sess.graph)

        sess.run(tf.global_variables_initializer())

        current_step = 0
        if(model_path == None):
            print("No Model find!\n")
        else:
            if(os.path.exists(model_path)):
                saver.restore(sess,model_path[0:5])
                current_step = int(re.sub("\D","",model_path)) + 1
                print(model_path[0:5],"has be load!\n")
            else:
                print("the path is not exist!\n")

        for i in range(cfg.MAX_TRAIN_STEP-current_step):
            pretime = time.time()
            i += current_step
            batch_x,batch_y = data.read()
            datatime = time.time()
            print("---------4.读取数据花费 %f ---------------\n" % (datatime - pretime))
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_x,labels:batch_y})
            steptime = time.time()
            print("---------5.训练一个batch花费 %f ---------------\n" % (steptime - datatime))
            train_writer.add_summary(summary,i)

            if(i % 2 == 0):
                summary,train_accuracy,loss = sess.run([merged,accuracy,total_loss],feed_dict={x: batch_x, labels: batch_y})
                test_writer.add_summary(summary,i)
                print("step %d,training accuracy %g,loss %f\n" % (i, train_accuracy,loss))

            # if(i % 1000 == 0) & (i != 0):
            #     test_x,test_y = data.read_test()
            #     test_accuracy,loss = sess.run([accuracy,loss],feed_dict={x:test_x,labels:test_y})
            #     print("step %d,testing accuracy %g,loss %f\n" % (i, test_accuracy,loss))
            #
            #     save = saver.save(sess,cfg.SAVE_PATH,global_step=i)
            #     print("the model is saved at %s"%save)

if __name__ == "__main__":
    train()


