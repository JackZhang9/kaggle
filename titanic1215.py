'''
titanic
一 导包
二 读入数据
三 EDA
四 数据预处理
五 特征工程
六 建模
七 模型评估与选择
八 测试测试集
'''
# 一 导包
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score,recall_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

#设置显示
pd.set_option('display.max_columns',500)    #最大列数
pd.set_option('display.max_rows',50)
pd.set_option('display.width',120) #打印宽度
pd.set_option('display.max_colwidth',None)
# 二 读入数据
train = pd.read_csv(r'C:\Users\jack\.kaggle\train.csv')
test = pd.read_csv(r'C:\Users\jack\.kaggle\test.csv')
gender_submission = pd.read_csv(r'C:\Users\jack\.kaggle\gender_submission.csv')
# 数据预观察
# print(train.describe(include='all'),'\n','-'*55)
# print(test.describe(include='all'))
'''通过count知道Age有缺失170个，可以用中位数补齐；Cabin缺失690个，缺失太多，无法使用，建议删掉'''
'''其中PID对分析没作用，也删掉，船票编号对分析没作用，删掉'''
# print(train.head(20))
'''查看前20行，知道名字中Mr,Miss,Mrs,Master可能重要，考虑提取出来'''
'''因为矩阵运算必须是数字，所以考虑将object对象转换成数字类型'''
#因为后面还要调用原csv文件中数值，因此复制一个副本，用作计算
train1 = train.copy()
test1 = test.copy()

# 三 EDA
# 一个个feature EDA
# EDA函数用图和数据，显示
def plot_distribution(feature,color):
    plt.figure(dpi=125)
    sns.set_style('whitegrid')
    sns.distplot(train1[feature],label=feature,color=color)
    plt.show()
    print('{} max value is {}\n min value is {}\nmean value is {}\nmedian value is {}\nstd value is {}'.format(feature,train1[feature].max(),train1[feature].min(),train1[feature].mean(),train1[feature].median(),train1[feature].std()))
#Age EDA
train1['Age'].fillna(train1['Age'].mean(),inplace=True)
# plot_distribution('Age','green')
'''已知，titanic优先营救小孩和女人，
根据图和计算知道，最大年龄是80，最小年龄只有几个月
年龄平均值是29.69911764705882，年龄中位数是28.0
而年龄标准差是14.526497332334042，'''

# Sex EDA
# plt.figure(dpi=125)
# sns.countplot(x=train1['Sex'],hue=train1['Survived'],data=train1)
# plt.show()
'''根据图得知，女性相比男性，生存率高得多，因此性别是一个影响力很大得因素'''

# Pclass EDA
# plt.figure(dpi=125)
# sns.countplot(x=train1['Pclass'],hue=train1['Survived'],data=train1)
#plt.show()
'''根据图得知，第3类生存率最低，第1类生存率最高，因此pclass是一个很有影响力得因素'''

#Embarked EDA
# plt.figure(dpi=125)
# sns.countplot(x=train1['Embarked'],hue=train1['Survived'],data=train1)
# plt.show()
'''根据图，在C上船生存率最高'''
train1['Embarked'].fillna(train1['Embarked'].mode()[0],inplace=True)


#考虑同辈亲人SibSp和子女父母Parch，同属亲人，因此建立新标签family
train1['Family'] = train1['SibSp'] + train1['Parch'] + 1

#Family EDA
# plt.figure(dpi=125)
# sns.countplot(x=train1['Family'],hue=train1['Survived'],data=train1)
# sns.countplot(x=train1['Family'],hue=train1['Age'],data=train1)
# plt.show()
'''根据图，2，3，4人口得家庭生存率更高'''

#根据是否单身，EDA
for i in range(len(train1['Family'])):
    if train1['Family'][i]>1:
        train1['Family'][i] = 1
    else:
        train1['Family'][i] = 0

# plt.figure(dpi=125)
# sns.countplot(x=train1['Family'],hue=train1['Survived'],data=train1)
# plt.show()
'''根据图，有家庭的人生存率更高，是否有家庭是一个影响因素'''

# Cabin EDA
#因Cabin缺失太多，所以先补全
train1['Cabin'].fillna('S',inplace=True)
# 稍微处理一下
for i in range(len(train1['Survived'])):
    train1['Cabin'][i] = train1['Cabin'][i][0]     #把每一个Cabin拿出来，再把每一个Cabin的第一个字符那出来
#看一下
# plt.figure(dpi=125)
# sns.countplot(x=train1['Cabin'],hue=train1['Survived'],data=train1)
# plt.show()
'''在C,E,D,B,F中生存率更大'''

# Fare EDA
# plt.figure(dpi=125)
# plot_distribution('Fare','orange')
'''根据图，知道最小值是0，有人免费上船，大部分集中在0-100'''
'''根据计算得知，Fare最大值是512，平均值是32，中位数是14，标准差是49'''
#根据Fare分成2类，1或0
train1['Fare_val'] = 0  #初始化为0，
for i in range(len(train1['Survived'])):
    if train1['Fare'][i] > 32:
        train1['Fare_val'][i] = 1

# plt.figure(dpi=125)
# sns.countplot(x=train1['Fare_val'],hue=train1['Survived'],data=train1)
# plt.show()
# '''根据图，票价大于32的时候，生存率更高'''

# 对测试集做出一些修改
#test1['Age']
test1['Age'].fillna(test1['Age'].mean(),inplace=True)
#test1['Family']
test1['Family'] = test1['SibSp'] + test1['Parch'] + 1
#根据是否单身，
for i in range(len(test1['Family'])):
    if test1['Family'][i]>1:
        test1['Family'][i] = 1
    else:
        test1['Family'][i] = 0
#test1['Cabin']
test1['Cabin'].fillna('S',inplace=True)
for i in range(len(test1['Cabin'])):
    test1['Cabin'][i] = test1['Cabin'][i][0]     #把每一个Cabin拿出来，再把每一个Cabin的第一个字符那出来
#test1['Fare_val']
test1['Fare_val'] = 0  #初始化为0，
for i in range(len(test1['Fare_val'])):
    if test1['Fare'][i] > 32:
        test1['Fare_val'][i] = 1

# 四 数据预处理

#把所有特征列出来
features = [#'PassengerId',
            'Pclass',
            #'Name'	,
            'Sex',
            'Age',
            #'SibSp',
            #'Parch',
            'Family',    #derived from Parch & Parch
            #'Ticket',
            #'Fare',
            'Fare_val',  #derived from Fare
            #'Cabin',
            'Embarked'
            ]

target = 'Survived'    #结果

# print(train1[features].isnull().sum())
# print(test1[features].isnull().sum())

# print(train1[features].head(10))
'''Pclass     Sex        Age  Family  Fare_val Cabin Embarked
0       3    male  22.000000       1         0     S        S
1       1  female  38.000000       1         1     C        C
2       3  female  26.000000       0         0     S        S'''
#特征数值化：二元用LabelEncoder，三元或多元可用One-Hot Encoding
#将二元对象转换成1，0，
lbl = LabelEncoder()
train1['Sex'] = lbl.fit_transform(train1['Sex'].values.ravel())
test1['Sex'] = lbl.fit_transform(test1['Sex'].values.ravel())

# print(train1[features].head(10))
''' Pclass  Sex        Age  Family  Fare_val Cabin Embarked
0       3    1  22.000000       1         0     S        S
1       1    0  38.000000       1         1     C        C
2       3    0  26.000000       0         0     S        S'''
train_ds = pd.get_dummies(train1[features],columns=['Pclass','Embarked'],drop_first=True)
test_ds = pd.get_dummies(test1[features],columns=['Pclass','Embarked'],drop_first=True)
# print(train_ds.head(10),'-'*50)
# print(train_ds.head(10))

#创建一个XGBoost 模型
y_train = train1[target]        #y的训练集，即survived得取值
X_train, X_valid,y_train,y_valid = train_test_split(train_ds,y_train,test_size=0.33)

clf1 = XGBClassifier()
#params是XGB里得参数,部分需要调参
params = {'n_estimatores':range(80,800,50),   #模型参数
          'early_stopping_rounds':range(100,400,50),
          'max_depth':range(2,15,1),
          'min_child_weight':range(1,9,1),
          'subsample':np.linspace(0.3,0.9,20),
          'colsample_bytree':np.linspace(0.5,0.98,20),
          'learning_rate':np.linspace(0.01,2,20),   #学习任务参数
          'objective':['binary:logistic'],
          'eval_metric':['logloss'],
          'gamma':np.linspace(0.5,5,20)
          }
clf1_cv = RandomizedSearchCV(clf1,params,cv=10,n_iter=50,n_jobs=-1)
clf1_cv.fit(X_train,y_train)
# print(clf1_cv.best_params_)
'''{'subsample': 0.8368421052631578,
     'objective': 'binary:logistic',
     'n_estimatores': 700,
      'min_child_weight': 3, 
      'max_depth': 6, 
       'learning_rate': 0.11473684210526315,
       'gamma': 1.2105263157894737, 
      'eval_metric': 'logloss', 
       'early_stopping_rounds': 200,
       'colsample_bytree': 0.5}'''
best_model = clf1_cv.best_estimator_
# print(best_model)
# print(clf1_cv.best_score_)
'''best_score:0.8186723163841808
0.8288983050847458'''
clf1_pred = best_model.predict(X_valid)
print('accuracy:',accuracy_score(y_valid,clf1_pred))
print('recall:',recall_score(y_valid,clf1_pred))
'''accuracy: 0.7966101694915254
recall: 0.6326530612244898'''

# 保存模型
filename = r'C:\Users\jack\.kaggle\Titanic_model.sav'
pickle.dump(best_model,open(filename, 'wb'))

#下载模型
# load_model = pickle.load(open(filename,'rb'))
#
# result = load_model.score(X_valid, y_valid)
# print(result)

passId = test['PassengerId'].values

final_pred = best_model.predict(test_ds)
# print(final_pred)

#我的提交
sub = {'PassengerId':passId.ravel(),'Survived':final_pred}

submission_csv = pd.DataFrame(sub)

#保存
submission_csv.to_csv(r'C:\Users\jack\.kaggle\submission.csv',index=False)





