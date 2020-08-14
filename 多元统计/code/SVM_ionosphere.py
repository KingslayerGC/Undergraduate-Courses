# %%
## 读取数据
import pandas as pd
file_path = r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\多元统计\SVM\data\ionosphere.txt"
data = pd.read_table(file_path, header=None, sep=',')
X = data.drop(columns=[1, 34])
y = data[34]

## ROC曲线函数
from sklearn.metrics import roc_curve
def plot_roc_curve(clf, X, y, label, ax, linetype='b'):
    clf.fit(X,y)
    scores = clf.decision_function(X)
    fpr, tpr, _ = roc_curve(y, scores, pos_label='g')
    ax.plot(fpr, tpr, linetype, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1],'r')
    ax.axis([0, 1, 0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='best')

## 绘制ROC曲线
from sklearn.svm import SVC
svm_clf = SVC(C=0.5, kernel='rbf', gamma=0.06)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_clf = LinearDiscriminantAnalysis(solver='svd', n_components=2)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
plot_roc_curve(svm_clf, X, y, label="SVM", ax=ax)
plot_roc_curve(lda_clf, X, y, label="LDA", ax=ax, linetype='y')

