# %%
## 一些函数

# 计算协方差矩阵的特征值和特征向量
def eig_pca(data, method='cov'):
    if method == 'corr':
        matrix = np.corrcoef(data.T)
    else:
        matrix = np.cov(data.T)
    lam, v = np.linalg.eig(matrix)
    sorted_indices = np.argsort(-lam)
    lam = lam[sorted_indices]
    v = v[:, sorted_indices]
    return lam, v

# 拐点图
import matplotlib.pyplot as plt
def screeplot(subloc, variance, xlabels, title):
    plt.subplot(subloc)
    plt.plot(variance, marker='o', markerfacecolor='w')
    plt.title(title)
    plt.xticks(range(3), labels=xlabels)
    plt.ylabel('Variance')

# 配对散点图
import seaborn as sns
def pairplot(data, diag_kind='kde', hue=None):
    sns.pairplot(data, diag_kind='kde', hue=hue)

# %%
## 7.1

# 生成数据
import numpy as np
np.random.seed(42)
data = np.random.multivariate_normal(
    [0, 0, 0], [[1,1,1],[1,4,1],[1,1,100]], 100)

# 分别计算特征值和特征向量
lam1, v1 = eig_pca(data)
lam2, v2 = eig_pca(data, 'corr')
print(lam1, '\n\n', v1, '\n\n', lam2, '\n\n', v2)

# 计算 PC-scores
import pandas as pd
data_pca1 = pd.DataFrame(data.dot(v1), columns=['Comp1', 'Comp2', 'Comp3'])
data_pca2 = pd.DataFrame(data.dot(v2), columns=['Comp1', 'Comp2', 'Comp3'])

# 绘制screeplot
screeplot(121, lam1, data_pca1.columns, "cov_based")
screeplot(122, lam2, data_pca1.columns, "corr_based")

# 主成分的配对散点图
pairplot(data_pca1)
pairplot(data_pca2)


# %%
## 7.4

# 读取数据
file_path = r"C:\Users\Mac\Desktop\LDR\pendigits.txt"
digit = pd.read_table(file_path, header=0, sep=' ').values
X = digit[:, :16]

# 计算变量方差
feature_var = np.var(X, axis=0)
print(feature_var)

# PCA
lam, v = eig_pca(X)

# 选取合适的主成分数目
var_cumratio = (lam / lam.sum()).cumsum()
print("取前" + str(np.argmin(var_cumratio < 0.8) + 1) + "个成分可以达到80%的方差贡献")
print("取前" + str(np.argmin(var_cumratio < 0.9) + 1) + "个成分可以达到90%的方差贡献")

# 主成分的配对散点图
X_pca = pd.DataFrame(X.dot(v), columns=['Comp'+str(i+1) for i in range(16)])
pairplot(X_pca.iloc[:, :3])

# %%
## 7.9

# 读取数据（这个数据集有150个样本）
iris = sns.load_dataset("iris")
X = iris.drop(columns='species').values

# PCA
lam, v = eig_pca(X)

# 计算 PC-scores
data = pd.DataFrame(X.dot(v), columns=['Comp'+str(i+1) for i in range(4)])

# 主成分的配对散点图
data = pd.concat([data, iris['species']], axis=1)
pairplot(data, hue='species')


