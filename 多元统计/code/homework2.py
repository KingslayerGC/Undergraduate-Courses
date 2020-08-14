# %%
## 8.2

# 读取数据
import pandas as pd
file_path = r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\多元统计\LDA\data\wine.train.txt"
data = pd.read_table(file_path, header=None)
X = data.drop(columns=13).values
y = data.iloc[:, -1].values

# LDA及其结果展示
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver='svd', n_components=2)
X_lda = lda.fit_transform(X, y)

# 绘至LDA得分散点图
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.style.use('default')
plt.scatter(X_lda[:,0], X_lda[:,1], c=y)
plt.xlabel("1st LDA score")
plt.ylabel("2nd LDA score")

# %%
## 8.6
# 读取数据（这个数据集有150个样本）
import seaborn as sns
data = sns.load_dataset("iris")

X = data.drop(columns='species')
y = data['species']

# 数据转化（相除，取对数）
X['sepal_shape'] = X['sepal_length'] / X['sepal_width']
X['petal_shape'] = X['petal_length'] / X['petal_width']
import math
X_tr = X[['sepal_shape', 'petal_shape']].applymap(lambda x: math.log(x))

# 绘制初始变量散点图
sns.set()
sns.scatterplot(X['sepal_shape'], X['petal_shape'], hue=y)

# LDA & QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score
loo = LeaveOneOut()

# 输出LDA的cv误分类率
lda = LinearDiscriminantAnalysis(solver='svd', n_components=2)
y_lpred = cross_val_predict(lda, X, y, cv=loo)
print("LDA的cv误分类率为：", 1 - accuracy_score(y, y_lpred))

# 输出QDA的cv误分类率
qda = QuadraticDiscriminantAnalysis()
y_qpred = cross_val_predict(qda, X, y, cv=loo)
print("QDA的cv误分类率为：", 1 - accuracy_score(y, y_qpred))
