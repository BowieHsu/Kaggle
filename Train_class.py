#encoding:utf-8
__author__ = 'bowiehsu'
from function import *

import csv
import numpy
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA

input_df = pd.read_csv('train.csv', header=0)
submit_df  = pd.read_csv('test.csv',  header=0)

# merge the two DataFrames into one
df = pd.concat([input_df, submit_df])
df = df .reset_index()
df = df.drop('index', axis=1)
df = df.reindex_axis(input_df.columns, axis=1)
features = input_df.values[:,1:]

input_df = pd.read_csv('train.csv', header=0)
submit_df  = pd.read_csv('test.csv',  header=0)

print "数据读取完成"
# merge the two DataFrames into one
df = pd.concat([input_df, submit_df])
df = df .reset_index()
df = df.drop('index', axis=1)
df = df.reindex_axis(input_df.columns, axis=1)

print "数据融合完成"
features = input_df.values[:, 1:]
labels = input_df.values[:,0]
#pca使用方法？
pca = PCA(n_components = 64)
pca.fit(df.values[:,1:])
features = pca.transform(features)
pred_data = pca.transform(submit_df.values)
print "Pca步骤完成"

#使用knn算法得到结果
clf = KNeighborsClassifier(n_neighbors=10).fit(features, labels)
print "样本训练完成"
output = clf.predict(pred_data).astype(int)
print "生成label完成"
ids = range(1, 28001)
predictions_file = open("sklearn_Knn=10_Result.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print output

