## 7.2
## 读取数据
import pandas as pd
file_path = r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\多元统计\LDA\data\wdbc.txt"
data = pd.read_table(file_path, sep=',', header=None)

X = data.drop(columns=[0, 1]).values
y = data.loc[:, 1].values


## 网格搜索SVM的最佳参数，并得到对应的cv误判率
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV

# 设置cv折数，定义参数表格
fold = KFold(n_splits=10, random_state=42)
param_range = {'C': [1, 10, 50, 100, 500, 1000],
               'gamma': np.logspace(-5, -1, 5)}

# 进行网格搜索
svm_clf = SVC(kernel='rbf') 
grid_cv = GridSearchCV(svm_clf, cv=fold,
                       param_grid=param_range)
grid_cv.fit(X, y)

# 得到所有组合的cv误判和最佳参数组合
svm_mcr = pd.DataFrame(
    (1 - grid_cv.cv_results_['mean_test_score']).reshape(6, -1),
    columns=param_range['gamma'], index=param_range['C'])
print("高斯核SVM各参数组合的cv误判率表，行为C值，列为gamma值\n", svm_mcr)
print("高斯核SVM最佳参数组合为：", grid_cv.best_params_,
      "；对应的cv误判率为：", svm_mcr.min().min())


## 得到LDA和TREE分类器的cv误判率
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
fold = KFold(n_splits=10, random_state=42)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_clf = LinearDiscriminantAnalysis(solver='svd', n_components=2)
print("LDA的cv误判率为：", 1 - cross_val_score(lda_clf, X, y, cv=fold).mean())

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5)
print("决策树的cv误判率为：", 1 - cross_val_score(tree_clf, X, y, cv=fold).mean())
