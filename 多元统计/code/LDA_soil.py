# %%
## 读取数据
import pandas as pd
file_path = r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\多元统计\LDA\data\soil.txt"
data = pd.read_table(file_path)

X = data.drop(columns='Group.no.').values
y = data['Group.no.'].values

# %%
## LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver='svd', n_components=2)
X_lda = lda.fit_transform(X, y)

# %%
## 绘图
import matplotlib.pyplot as plt
plt.scatter(X_lda[:,0], X_lda[:,1], c=y)
plt.xlabel("1st LDA score")
plt.ylabel("2nd LDA score")
