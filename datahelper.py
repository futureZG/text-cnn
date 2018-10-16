#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import codecs
import jieba
import re
from sklearn import preprocessing

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

#### 读取数据
fundata = pd.read_csv("./data", sep = "\t", header = None, encoding = "utf-8", 
                       error_bad_lines = False, dtype = {"":str})
fundata.columns = [ "item_name", "item_third_cate_name", "item_third_cate_cd"]


#### 处理数据
names = fundata.loc[:, "item_name"]
#names = list(map(lambda x : re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "", x), names))
names = list(map(lambda x : re.sub(u"[\s+\.\!\/_,$%^*(+\"\')+——()\?【】“”！，。？、~@#￥%……&*（）-]+|[\d+\.?\d*]+|[cmCMXxmgkgKGLXLXXL]+", "", x), names))

#### 商品类别标签写入本地
label_list = list(fundata.loc[:, "item_third_cate_cd"].unique())
fo = codecs.open("./label_list.txt", "w", "utf-8")
for i in range(len(label_list)):
    fo.write(str(label_list[i]))
    if (i < len(label_list) - 1):
        fo.write("\n")
fo.close()

#### 商品名称分词写入本地
print("开始分词")
#jieba.load_userdict("/home/mart_coo/chenshengtai/baojia6/brand2.txt")#导入用户自定义词典
fo = codecs.open("./segname.txt", "w", "utf-8")
for i in range(len(names)):
    segwords = list(jieba.cut(names[i], HMM = True))
    for segword in segwords:
        fo.write(segword)
        fo.write(" ")
    if (i < len(names) - 1):
        fo.write("\n")
fo.close()
print("=================分词结束===================")

with codecs.open("./label_list.txt", "r", "utf-8") as f:
    label_list = f.read().split("\n")

labels = fundata.loc[:, "item_third_cate_cd"]
le = preprocessing.LabelEncoder()
le.fit(label_list)
all_labels = le.transform(labels)

fo = codecs.open("./labels.txt", "w", "utf-8")
for i in range(len(all_labels)):
    fo.write(str(all_labels[i]))
    if (i < len(all_labels)-1):
        fo.write("\n")
fo.close()
print("=================类别标签写入本地结束=======================")
