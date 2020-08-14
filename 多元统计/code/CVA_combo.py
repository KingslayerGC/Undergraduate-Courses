# %%
## 读取数据
import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\Mac\Desktop\LDR\combo17_wo_missing.csv", index_col=0)

X = data.iloc[:, :23].values
Y = data.iloc[:, 23:].values

# %%
## CVA过程
from sklearn.cross_decomposition import CCA
cva = CCA(n_components=6, scale=False)
X_cva, Y_cva = cva.fit_transform(X, Y)

# %%
## 展示CVA组合的相关程度
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(11, 6))
for i in range(6):
    corr = round(np.corrcoef(X_cva[:, i], Y_cva[:, i])[0,1], 3)
    plt.subplot(231 + i)
    plt.plot(X_cva[:, i], Y_cva[:, i], 'ro',
             markerfacecolor='w', markersize=4)
    plt.xlabel(chr(958) + str(i+1))
    plt.ylabel(chr(969) + str(i+1))
    plt.yticks(rotation=60)
    plt.title("rho=" + str(corr))
plt.subplots_adjust(wspace=0.3, hspace=0.4)