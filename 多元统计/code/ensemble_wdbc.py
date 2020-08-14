# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

## 读取数据
file = r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\多元统计\Committe\data\wdbc.txt"
data = pd.read_csv(file, header=None)
X = data.iloc[:, 2:]
y = data.iloc[:, 1]

# %%
## 基分类器数目和树深度对随机森林影响
error_rate1 = pd.DataFrame(index=[10]+list(range(25, 225, 25)),
                          columns=['max_depth_'+str(i) for i in range(1,4)])

for num in error_rate.index:
    for depth in range(1, 4):
        forest_clf = RandomForestClassifier(
            n_estimators=num, max_depth=depth,
            max_features=None, oob_score=True, random_state=42)
        forest_clf.fit(X, y)
        error_rate1.loc[num, 'max_depth_'+str(depth)] = 1 - forest_clf.oob_score_

error_rate1.plot()
plt.xlabel("Number of Bootstrap Samples")
plt.ylabel("Average OOB Misclassification Rate")

# %%
## 基分类器数目对随机森林影响
error_rate2 = pd.DataFrame(
    columns=[10]+list(range(25, 125, 25)), index=range(20))

for num in error_rate2.columns:
    for i in range(20):
        forest_clf = RandomForestClassifier(
            n_estimators=num,
            max_features=None, oob_score=True)
        forest_clf.fit(X, y)
        error_rate2.loc[i, num] = 1 - forest_clf.oob_score_

error_rate2.plot(kind='box')
plt.xlabel("Number of Bootstrap Samples")
plt.ylabel("Average OOB Misclassification Rate")

# %%
## 划分训练集和测试集
split = StratifiedShuffleSplit(test_size=0.3)
for train_index, test_index in split.split(X, y):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

## 基分类器数目对Adaboost影响
def error_rate_compute(clf):
    return 1 - accuracy_score(clf.predict(X_train), y_train),\
        1 - accuracy_score(clf.predict(X_test), y_test)

error_rate3 = pd.DataFrame(index=[1]+list(range(10, 50, 5)),
                           columns=['Train', 'Test'])

for i in range(len(error_rate3)):
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                n_estimators=error_rate3.index[i],
                                random_state=42)
    ada_clf.fit(X_train, y_train)
    error_rate3.loc[error_rate3.index[i], 'Train'],\
        error_rate3.loc[error_rate3.index[i], 'Test'] = \
            error_rate_compute(ada_clf)
    
error_rate3.plot()
plt.xlabel("Number of Base Estimators")
plt.ylabel("Adaboost Misclassification Rate")

# %%
## 特征重要度
forest_clf = RandomForestClassifier(n_estimators=500, random_state=42)
forest_clf.fit(X, y)
var_im = pd.Series(forest_clf.feature_importances_,
                   index=['V'+str(i+1) for i in range(X.shape[1])])\
    .sort_values(ascending=False).iloc[:10]

var_im.plot()
plt.xlabel("Variable")
plt.ylabel("Gini Feature Importance")