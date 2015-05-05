#encoding:utf-8
__author__ = 'bowiehsu'
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
#使用randomforest_classifier

#清洗数据，训练样本
train_df = pd.read_csv('train.csv')
#将所有数据中的string 转化为 int
train_df['Gender'] = train_df['Sex'].map({'female':0,'male':1}).astype(int)
#print train_df['Gender']
#Embarked from 'C', 'Q' ,'S'
if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
    train_df.Embarked[train_df.Embarked.isnull()] = train_df.Embarked.dropna().mode().values
#print train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))#所有Embarked的情况
Ports_dict = {name:i for i,name in Ports}
print Ports_dict
train_df.Embarked = train_df.Embarked.map(lambda x:Ports_dict[x]).astype(int)
#print train_df.Embarked

#继续清洗数据
#缺少年龄的，补上平均年龄
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
    train_df.loc[(train_df.Age.isnull()),'Age'] = median_age
train_df = train_df.drop(['Name','Sex','Ticket','Cabin','PassengerId'],axis = 1)


#********** 我是分割线 *****************

#开始清洗TEST数据
test_df = pd.read_csv('test.csv',header = 0)
#if test_df['Sex'] != 'famale' and test_df['Sex'] != 'male':
#    print test_df['Sex']
#test_gender = test_df['Sex']
#for i in test_gender:
#    if i != 'female' and i != 'male':
#        print i
test_df['Gender'] = test_df['Sex'].map({'female':0,'male':1}).astype(int)

#对于缺少embarked值的数据，将最常用的embarked值赋给它们
if len(test_df.Embarked[test_df.Embarked.isnull()]) > 0:
    test_df.Embarked[test_df.Embarked.isnull()] = test_df.Embarked.dropna().mode().values
#将CQS转化为123
test_df.Embarked = test_df.Embarked.map(lambda x:Ports_dict[x]).astype(int)

#清洗年龄数据与PassengerId数据
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
    test_df.loc[(test_df.Age.isnull()),'Age'] = median_age

ids = test_df['PassengerId'].values
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

#缺少船票的
if len(test_df.Fare[test_df.Fare.isnull()]) > 0:
    median_fare = np.zeros(3)
    for f in range(3):
        median_fare[f] = test_df[test_df.Pclass == f+1]['Fare'].dropna().median()
    for f in range(3):
        test_df.loc[(test_df.Fare.isnull())&(test_df.Pclass == f+1),'Fare'] = median_fare[f]

train_data = train_df.values
test_data = test_df.values

print 'Training...'
forest = RandomForestClassifier(n_estimators=1000, random_state=500, min_samples_leaf=3)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

print 'Predicting...'

output = forest.predict(test_data).astype(int)
'''
print output
trans_dic = {i:name for i,name in Ports}
print trans_dic
output2 = []
for i in output:
    output2.append(trans_dic[i-1])
print output2'''

prediction_file = open("result_use_random_forest.csv","wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow (["PassengerId","Survived"])
open_file_object.writerows(zip(ids,output))
prediction_file.close()
print 'Done!'