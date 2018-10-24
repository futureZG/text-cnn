#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### 加载包

from __future__ import print_function
import os
import codecs
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow.contrib import learn

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

#### 定义参数


embedding_size = 128
filter_sizes = [3, 4, 5]
num_filters = 64
l2_reg_lambda = 0.0
prob1 = 0.5l
prob2 = 1.0


#### path
file_y = "./label_list.txt"
file_x = "./segname.txt"
check_dir = "./result/cnn_softmax.ckpt"

classes_df = pd.read_csv("./label_list.txt", sep = "\t", header = None, encoding = "utf-8")
classes_list = classes_df.loc[:, 0]
num_classes = len(classes_list)



fundata = pd.read_csv("./data", sep = "\t", header = None, encoding = "utf-8", 
                       error_bad_lines = False, dtype = {"":str})
fundata = fundata.dropna()

x_text = list(codecs.open(file_x, "r", "utf-8").read().split("\n"))

max_document_length = max([len(x.split(" ")) for x in x_text])
sequence_length = max_document_length
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit_transform(x_text)
vocab_size = len(vocab_processor.vocabulary_)

#### 占位符:
inputx = tf.placeholder(tf.int32, [None, sequence_length], name = "inputx")
dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")



    
#### 卷积层
def textCNN(inputx, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):

    #l2 regularization loss
    l2_loss = tf.constant(0.0)

    #embedding layer
    with tf.device("/cpu:0"):
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1), name = "W")
        embedded_chars = tf.nn.embedding_lookup(W, inputx)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    #convolution and maxpool layer
    pooled_output = []
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, embedding_size, 1, num_filters]  #filter_height, filter_width, in_channels, out_channels
        print(filter_shape)
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
        b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "b")
        conv = tf.nn.conv2d(embedded_chars_expanded,
                            W,
                            strides = [1, 1, 1, 1],
                            padding = "VALID",
                            name = "conv")
        #apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
        #maxpooling over the outputs
        pooled = tf.nn.max_pool(h,
                                ksize = [1, sequence_length - filter_size + 1, 1, 1],
                                strides = [1, 1, 1, 1],
                                padding = "VALID",
                                name = "pool")
        print("==============         pooled              ==========", pooled.shape)
        pooled_output.append(pooled)
        print("==============         pooled_output       ==========", len(pooled_output))
    #combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_output, 3)
    print("============         h_pool    shape  ========", h_pool.shape)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total], name = "feature_vec")
    print("===========          h_pool_flat      ========", h_pool_flat.shape)
    # Add dropout
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # final socres and predictions
    W = tf.get_variable("W",
                        shape = [num_filters_total, num_classes],
                        initializer = tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "b")

    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)

    scores = tf.nn.xw_plus_b(h_drop, W, b, name = "scores")
    return l2_loss, scores    

l2_loss, scores = textCNN(inputx, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters)
predictions = tf.argmax(scores, 1)

#### 加载模型
check_dir = "./result/cnn_softmax.ckpt"
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, check_dir)
print("Model restored.")

#### 待预测商品
name = "金号纯棉浴巾吸水成人加大加厚男女情侣款儿童适用4320A白色18080cm"
seg = [" ".join(list(jieba.cut(name)))]

x = np.array(list(vocab_processor.fit_transform(seg)))
x = x.reshape(-1, sequence_length)
#phy = np.array([425.0, 281.0, 68.0, 0.60, 8.12090], dtype = np.float32)
pre = sess.run(predictions, feed_dict={inputx: x, dropout_keep_prob: prob})
print(pre)

# from sklearn import preprocessing
# #这里的list的内容需要改成label_list=list(third.loc[:, 1].unique())
# third_label_encoder = pd.read_csv("./label_list.txt", sep = "\t", header = None, encoding = "utf-8")
# label_list=list(third_label_encoder.loc[:, 0])
# le.fit(label_list)

# #### 输出预测结果
# predictions = le.inverse_transform(pre)

# print(predictions)