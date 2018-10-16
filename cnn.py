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

dev_sample_percentage = 0.05
batch_size = 500
epochs = 50
num_classes = 2246
embedding_size = 128
filter_sizes = [3, 4, 5]
num_filters = 64
l2_reg_lambda = 0.0
prob1 = 0.5
prob2 = 1.0
display_step = 100
learn_reate = 0.001


#### path
file_x = "./segname.txt"
file_y = "./labels.txt"
check_dir = "./result/cnn_softmax.ckpt"


def load_data_labels(file_x, file_y):
    x_text = list(codecs.open(file_x, "r", "utf-8").read().split("\n"))
    labels = list(codecs.open(file_y, "r", "utf-8").read().split("\n"))
    labels = [int(label) for label in labels]
    y = to_categorical(np.asarray(labels))
    print("the length of x:", len(x_text))
    print("the length of y:", len(y))
    return x_text, y
    
def create_placeholder(sequence_length, num_classes):
    inputx = tf.placeholder(tf.int32, [None, sequence_length], name = "inputx")
    inputy = tf.placeholder(tf.int32, [None, num_classes], name = "inputy")
    dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
    return inputx, inputy, dropout_keep_prob

def init_seq(x_text, y, dev_sample_percentage):
    np.random.seed(10)
    max_document_length = max([len(x.split(" ")) for x in x_text])
    sequence_length = max_document_length
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print(x[1])
    # Randomly shuffle data  
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    
    # Split train/test set
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    vocab_size = len(vocab_processor.vocabulary_)
    print("vocab_size: ", vocab_size)
    return sequence_length, vocab_size, x_train, x_dev, y_train, y_dev
    
#### 卷积层
def textCNN(inputx, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):

    #l2 regularization loss
    l2_loss = tf.constant(0.0)

    #embedding layer
    with tf.device("/cpu:0"):
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1), name = "W")
        embedded_chars = tf.nn.embedding_lookup(W, inputx)
        print("embedding matrix", embedded_chars.shape)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) #[batch, in_height, in_width, in_channels]
        print("expand embedding", embedded_chars_expanded.shape)

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

def train_ops(scores, inputy, learn_reate):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = inputy)
    loss = losses + l2_reg_lambda * l2_loss
    loss = tf.reduce_mean(loss)

    predictions = tf.argmax(scores, 1, name = "predictions")
    # Accuracy
    correct_predictions = tf.equal(predictions, tf.argmax(inputy, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")
    
    # top5 准确率
    ture_label = tf.argmax(inputy, 1)
    correct_5 = tf.nn.in_top_k(scores, ture_label, k = 5)
    accuracy_top5 = tf.reduce_mean(tf.cast(correct_5, "float"), name = "top5_accuracy")

    # Define training procedure
    global_step = tf.Variable(0, name = "global_step", trainable = False)
    optimizer = tf.train.AdamOptimizer(learn_reate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
    return loss, accuracy, accuracy_top5, train_op

def train(check_dir, x_dev, y_dev, x_train, y_train, batch_size, epochs, train_op, inputx, inputy, dropout_keep_prob, prob1,          prob2, accuracy, loss):
    #### 初始化&开始计算
    max_accuracy = 0
    init = tf.global_variables_initializer()

    test_data = x_dev.reshape(-1, sequence_length)
    print("test_data", test_data.shape)
    test_label = y_dev.reshape(-1, num_classes)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        train_data_len = len(x_train)
        steps = train_data_len/batch_size
        for epoch in range(epochs):
            print("step of epoch is: {}".format(steps))
            train_range = zip(range(0, train_data_len, batch_size), range(batch_size, train_data_len, batch_size))
            for step ,(start, end) in enumerate(train_range):
                batch_x = x_train[start:end].reshape((batch_size, sequence_length))
                batch_y = y_train[start:end].reshape((batch_size, num_classes))
                sess.run(train_op, feed_dict = {inputx: batch_x, inputy: batch_y, dropout_keep_prob: prob1})
    #             if step > 2:
    #                 break
                if step % display_step == 0:
                    acc, acc5, los = sess.run([accuracy, accuracy_top5, loss], feed_dict={inputx: batch_x, inputy: batch_y, dropout_keep_prob: prob1})
                    print("epoch: " + str(epoch + 1) + ", Iter" + str(step) + ", Minibatch Loss=" +  "{:.6f}".format(los) + ", Training Accuracy= " + "{:.5f}".format(acc) + ", top5_accuracy= " +
                          "{:.5f}".format(acc5))
                    print ("Optimization step {}".format(step))
                    
                    testacc, testacc5 = sess.run([accuracy, accuracy_top5], feed_dict={inputx: test_data, inputy: test_label, dropout_keep_prob: prob2})
                    print ("Testing accuracy:", testacc, " Testing top 5 accuracy:", testacc5)
                                    
                    if testacc > max_accuracy:
                        max_accuracy = testacc
                        saver.save(sess, check_dir)
                    print("")
                    if testacc > 0.95:
                        break
                        print("testacc is bigger than 0.95, break")
    print("max accuracy: ", max_accuracy)    
#### functions
print("Loading data...")
x_text, y = load_data_labels(file_x, file_y)
print("loading done")

sequence_length, vocab_size, x_train, x_dev, y_train, y_dev = init_seq(x_text, y, dev_sample_percentage)

inputx, inputy, dropout_keep_prob = create_placeholder(sequence_length, num_classes)

l2_loss, scores = textCNN(inputx, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters)

loss, accuracy, accuracy_top5, train_op = train_ops(scores, inputy, learn_reate)

train(check_dir, x_dev, y_dev, x_train, y_train, batch_size, epochs, train_op, inputx, inputy, dropout_keep_prob, prob1,      prob2, accuracy, loss)


