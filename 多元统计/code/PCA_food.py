# %%
##　读取数据
import pandas as pd
file_path = r"C:\Users\Mac\Desktop\LDR\food.data.txt"
food = pd.read_table(file_path, header=0, sep=' ').iloc[:, 8:].values

# %%
## PCA过程
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scale = StandardScaler()
pca = PCA(svd_solver='full', random_state=42)
food_pca = pd.DataFrame(
    pca.fit_transform(scale.fit_transform(food)),
    columns=['Comp'+str(i) for i in range(1,7)])

# %%
## 方差贡献分析

# 各主成分方差贡献率展示
var_ratio = pca.explained_variance_ratio_
from prettytable import PrettyTable
table = PrettyTable([' '] + ['Comp'+str(i) for i in range(1,7)])
table.add_row(['Proportion Var'] + list(var_ratio.round(3)))
table.add_row(['Cumulative Var'] + list(var_ratio.cumsum().round(3)))
print(table)

# 方差拐点图
import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_, marker='o', markerfacecolor='w')
plt.xticks(range(6), labels=list(food_pca.columns))
plt.ylabel('Variance')

# %% 
## 绘图

# 第一第二主成分的散点图
fig, ax = plt.subplots(1, 1)
ax.plot(food_pca['Comp1'], food_pca['Comp2'], 'o',
             markerfacecolor='w', markersize=4)
ax.set_xlabel('Comp1 Score')
ax.set_ylabel('Comp2 Score')

# 所有主成分的配对散点图
import seaborn as sns
sns.pairplot(food_pca, diag_kind='kde')
