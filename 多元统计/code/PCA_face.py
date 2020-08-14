# %%
## 读取图片

import os
import numpy as np
import matplotlib.pyplot as plt

# 设置图片路径
dir_path = r"C:\Users\Mac\Desktop\LDR\face recognition"
image_list = []

# 读取所有图片
for parents, _, filename in os.walk(dir_path):
    for file in filename:
        file_path = os.path.join(parents, file)
        image = plt.imread(file_path)[:,:,0]
        image_shape = image.shape
        image_list.append(image.reshape(1, -1))
X = np.vstack(image_list)        

# %%
## 图片PCA效果展示

def image_show(x):
    plt.imshow(x.reshape(image_shape), cmap='Greys_r')
    plt.axis('off')

# 展示原图
p2 = image_list[3]
image_show(p2)

# 展示只取若干个主成分的图片
from sklearn.decomposition import PCA
plt.figure(figsize=(10,8))
for i in range(9):
    pca = PCA(n_components=i)
    pca.fit(X)
    X_pca = pca.inverse_transform(pca.transform(X))
    p2_pca = X_pca[3, :]
    plt.subplot(331 + i)
    image_show(p2_pca)
plt.subplots_adjust(wspace=0, hspace=0)




