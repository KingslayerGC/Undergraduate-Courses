# %%
## 读取数据
import pandas as pd
import math

file_path = r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\多元统计\LDA\data\wdbc.txt"
data = pd.read_table(file_path, sep=',', header=None)

data.replace(to_replace=0, value=0.001, inplace=True)
data.iloc[:, 2:] = data.iloc[:, 2:].applymap(lambda x: math.log(x))

X = data.drop(columns=[0, 1]).values
y = data.loc[:, 1].values

# %%
## 绘图函数
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 密度 & 直方图
def histplot(X, bins, xrange, hist=True, kde=False, xlabel=None):
    sns.distplot(X, bins=bins, hist=hist, kde=kde, norm_hist=True)
    sns.despine()
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.xticks(xrange)

# %%
##　LDA及结果展示
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver='svd', n_components=1)
X_lda = lda.fit_transform(X, y)

# 一维降维结果密度直方图
grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.4)

plt.subplot(grid[0, 0])
histplot(X_lda[y=='B'], bins=40, xrange=np.arange(-4,8,2), xlabel="Group B")

plt.subplot(grid[1, 0])
histplot(X_lda[y=='M'], bins=40, xrange=np.arange(-4,8,2), xlabel="Group M")

plt.subplot(grid[:, 1])
histplot(X_lda[y=='M'], bins=40, xrange=np.arange(-4,8,2),
         hist=False, kde=True, xlabel="Group B and M")
histplot(X_lda[y=='B'], bins=40, xrange=np.arange(-4,8,2),
         hist=False, kde=True)

# 估计参数
lda.scalings_

# Leave-One-Out混淆矩阵
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
y_pred = cross_val_predict(lda, X, y, cv=loo)
confusion_matrix(y, y_pred)
