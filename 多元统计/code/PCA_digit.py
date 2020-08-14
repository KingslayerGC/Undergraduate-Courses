# %%
## 读取数据
import pandas as pd
file_path = r"C:\Users\Mac\Desktop\LDR\pendigits.txt"
digit = pd.read_table(file_path, header=0, sep=' ').values

X = digit[:, :16]
y = digit[:, -1]

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scale = StandardScaler()
pca = PCA(n_components=3)
X_pca = pca.fit_transform(scale.fit_transform(X))

# %%
## PCA分类效果展示
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))

ax = plt.subplot(221, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], s=0.3, c=y, cmap=plt.cm.jet)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("Comp1")
ax.set_ylabel("Comp2")
ax.set_zlabel("Comp3")

plt.subplot(222)
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=0.3, c=y, cmap=plt.cm.jet)
plt.xlabel("Comp1")
plt.ylabel("Comp2")

plt.subplot(223)
plt.scatter(X_pca[:, 0], X_pca[:, 2], s=0.3, c=y, cmap=plt.cm.jet)
plt.xlabel("Comp1")
plt.ylabel("Comp3")

plt.subplot(224)
plt.scatter(X_pca[:, 1], X_pca[:, 2], s=0.3, c=y, cmap=plt.cm.jet)
plt.xlabel("Comp2")
plt.ylabel("Comp3")

plt.subplots_adjust(wspace=0.2, hspace=0.2)