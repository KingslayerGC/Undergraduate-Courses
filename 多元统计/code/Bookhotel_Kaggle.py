#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Notebook显示设置
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'
import warnings
warnings.filterwarnings("ignore")


# In[1]:


# 读取数据
import pandas as pd
file_path = r"C:\Users\Mac\Desktop\过程\临时文件\hotel_bookings.csv\hotel_bookings.csv"
data = pd.read_csv(file_path, sep=',', header=0)
# 备份数据
data_backup = data.copy()


# In[3]:


# 查看数据类型和缺失情况
data.info()
# 观察标签数据的分布情况
data['is_canceled'].value_counts()


# In[8]:


# 绘制条形图1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
mapping = {month:i+1 for i, month in enumerate(months)}
data['month'] = data['arrival_date_month'].map(mapping)
data['count'] = 1
fig, ax = plt.subplots(1, figsize=[15, 10])
data.pivot_table(index='month', values='count', columns='is_canceled', aggfunc=np.sum).sort_index().plot(kind='bar', stacked=True, ax=ax)
ax.set_xticklabels(months, fontsize=15, rotation=30)
ax.legend(["Not Canceled", "Canceled"], loc='upper left', fontsize=20)
ax.set_ylabel("Hotel Book Number", fontsize=17)
ax.set_xlabel("Month of Arrival", fontsize=17)


# In[16]:


# 绘制条形图2
fig, ax = plt.subplots(1, figsize=(15,8))
data.pivot_table(index='market_segment', values='count', columns='is_canceled', aggfunc=np.sum).sort_index().plot(kind='bar', stacked=True, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=30)
ax.legend(["Not Canceled", "Canceled"], loc='upper left', fontsize=20)
ax.set_ylabel("Hotel Book Number", fontsize=17)
ax.set_xlabel("Market Segment", fontsize=17)


# In[15]:


# 绘制条形图3
fig, axes = plt.subplots(1, 2, figsize=(10,7))
i = 0
for col in ['previous_cancellations', 'lead_time']:
    sns.barplot(x='is_canceled', y=col, data=data, ax=axes[i], color="salmon")
    axes[i].set_xticklabels(["Not Canceled", "Canceled"], fontsize=15, rotation=30)
    axes[i].set_xlabel("")
    axes[i].set_ylabel(col, fontsize=17)
    sns.despine()
    i += 1


# In[3]:


# 初步筛选掉一些明显不相关的变量
data = data_backup
deleted_list=['arrival_date_year', 'reservation_status','reservation_status_date']
data.drop(columns=deleted_list, axis=1, inplace=True)


# In[4]:


# 填补缺失值
data['children'].fillna(int(data['children'].mode()), inplace=True)
data['country'].fillna(str(data['country'].mode()), inplace=True)
data['agent'].fillna("none", inplace=True)
data['company'].fillna("none", inplace=True)


# In[5]:


# 对分类变量进行单热编码
cat_list = ['hotel','arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
            'meal', 'country', 'market_segment', 'distribution_channel', 'is_repeated_guest',
            'reserved_room_type','assigned_room_type', 'deposit_type', 'agent', 'company',
            'customer_type', 'required_car_parking_spaces']
data = pd.get_dummies(data, columns=cat_list, prefix=dict(zip(cat_list, cat_list)), drop_first=True)


# In[6]:


# 分割训练集和测试集
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
for train_index, test_index in split.split(data, data['is_canceled']):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
X_train = train_data.drop(columns=['is_canceled'], axis=1)
y_train = train_data['is_canceled']
X_test = test_data.drop(columns=['is_canceled'], axis=1)
y_test = test_data['is_canceled']


# In[7]:


# 随机森林筛选变量
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(n_estimators=500,n_jobs=-1)
clf_forest.fit(X_train, y_train)
importances = pd.DataFrame(clf_forest.feature_importances_, columns=["importance"], index=X_train.columns).sort_values(by='importance', ascending=False)
X_train = X_train.loc[:, importances.index[:100]]
X_test = X_test.loc[:, importances.index[:100]]


# In[8]:


# 标准化
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)


# In[10]:


from sklearn.linear_model import LogisticRegression
logit_clf = LogisticRegression()
logit_clf.fit(X_train, y_train)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_clf = LinearDiscriminantAnalysis(solver='svd', n_components=2)
lda_clf.fit(X_train, y_train)

from sklearn.svm import LinearSVC
linearsvm_clf = LinearSVC()
linearsvm_clf.fit(X_train, y_train)


# In[11]:


# 查看几个线性分类器的准确率
from sklearn.metrics import accuracy_score
def accuracy(clf_name, kind):
    clf = globals()[clf_name]
    if kind == 'train':
        y_pred = clf.predict(X_train)
        print("训练集：", accuracy_score(y_train, y_pred))
    if kind == 'test':
        y_pred = clf.predict(X_test)
        print("测试集：", accuracy_score(y_test, y_pred))

clf_list = ['logit_clf', 'lda_clf', 'linearsvm_clf']
for clf_name in clf_list:
    print("\n", clf_name, "的准确率如下")
    accuracy(clf_name, 'train')
    accuracy(clf_name, 'test')


# In[55]:


# 绘制ROC曲线
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
def plot_roc_curve(clf_name, linetype):
    clf = globals()[clf_name]
    try:
        scores = clf.decision_function(X_train)
    except:
        scores = clf.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    auc = roc_auc_score(y_train, scores)
    label = clf_name+"  (AUC=" + str(round(auc,4)) + ")"
    plt.plot(fpr, tpr, linetype, linewidth=1, label=label)

plt.figure(figsize=(10, 10))
for clf_name, linetype in zip(clf_list, ['r', 'y', 'g']):
    plot_roc_curve(clf_name, linetype)

plt.plot([0, 1], [0, 1], 'b')
plt.axis([-0.01, 1, 0, 1.01])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.legend(loc='bottomright', fontsize=20)
plt.show()


# In[59]:


# 高斯核svm（已减少数据量）
import random
index = random.sample(range(len(X_train)), 5000)
X = X_train[index, :]
y = y_train.iloc[index]

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV

fold = KFold(n_splits=5, random_state=42)
param_range = {'C': [1, 10, 50, 100, 500, 1000], 'gamma': np.logspace(-5, -1, 5)}

svm_clf = SVC(kernel='rbf') 
grid= GridSearchCV(svm_clf, param_grid=param_range)
grid.fit(X, y)

print("高斯核SVM最佳参数组合为：", grid.best_params_)
print("最佳高斯核svm的准确率如下")
y_pred = grid.predict(X)
print("训练集：", accuracy_score(y, y_pred))
y_pred = grid.predict(X_test)
print("测试集：", accuracy_score(y_test, y_pred))


# In[17]:


# 决策树剪枝图
from sklearn.tree import DecisionTreeClassifier
path = DecisionTreeClassifier(criterion='gini').cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots(1, figsize=(12, 7))
ax.plot(ccp_alphas[:-1], impurities[:-1])
ax.set_xlabel("effective alpha", fontsize=15)
ax.set_ylabel("total impurity of leaves", fontsize=15)


# In[25]:


# 决策树
tree_clf = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.0001)
tree_clf.fit(X_train, y_train)
print('tree_clf', "的准确率如下")
accuracy('tree_clf', 'train')
accuracy('tree_clf', 'test')


# In[12]:


# 随机森林
forest_clf = RandomForestClassifier(n_estimators=500, max_depth=30, n_jobs=-1, oob_score=True, random_state=42)
forest_clf.fit(X_train, y_train)
print('forest_clf', "的准确率如下")
accuracy('forest_clf', 'train')
accuracy('forest_clf', 'test')
print("OOB-SCORE：", forest_clf.oob_score_)


# In[15]:


# Adaboost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=200, learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
print('ada_clf', "的准确率如下")
accuracy('ada_clf', 'train')
accuracy('ada_clf', 'test')


# In[ ]:




