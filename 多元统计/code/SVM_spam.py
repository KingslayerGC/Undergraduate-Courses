## 读取数据
import pandas as pd
file_path=r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\多元统计\SVM\data\spambase.txt"
data = pd.read_table(file_path)

## 处理数据
data.drop(columns='classdigit', inplace=True)
X = data.drop(columns='class')
y = data['class']

## kernelSVM的cv得分
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
fold = KFold(n_splits=10, random_state=42)
svm_clf = SVC(C=500, kernel='rbf', gamma=0.002)
cv_score = cross_val_score(svm_clf, X, y, cv=fold).mean()
