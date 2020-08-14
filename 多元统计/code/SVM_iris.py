## 读取数据（这个数据集有150个样本）
import seaborn as sns
data = sns.load_dataset("iris")
X = data.drop(columns='species')
y = data['species']

## 网格搜索最佳参数，并绘制gamma-cv图
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV

fold = KFold(n_splits=10, random_state=42)
svm_clf = SVC(C=500, kernel='rbf')
gamma_range = np.arange(1,201) / 1000

grid_cv = GridSearchCV(svm_clf, cv=fold,
                       param_grid={'gamma':gamma_range})
grid_cv.fit(X, y)

import matplotlib.pyplot as plt
cv_score = grid_cv.cv_results_['mean_test_score']
plt.plot(gamma_range, 1-cv_score)
plt.xlabel("gamma")
plt.ylabel("misclassification rate")


