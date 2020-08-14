# %%
## 加载数据
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"].astype(int)
# 处理数据
X_not5 = X[y!=5]
y_not5 = y[y!=5]
import numpy as np
import random
ind = random.sample(range(X_not5.shape[0]),6000)
X_not5 = X_not5[ind]; y_not5 = y_not5[ind]
X = np.vstack([X[y==5], X_not5])
y = np.hstack([y[y==5], y_not5])

# %%
## 查看数据
import matplotlib.pyplot as plt
import matplotlib as mpl
# 显示两张图片
image1 = X[0].reshape(28, 28)
image2 = X[-1].reshape(28, 28)
fig, axes = plt.subplots(1,2)
axes[0].imshow(image1, cmap = mpl.cm.binary)
axes[1].imshow(image2, cmap = mpl.cm.binary)
axes[0].axis("off"); axes[0].text(2, 3, y[0], fontsize=15)
axes[1].axis("off"); axes[1].text(2, 3, y[-1], fontsize=15)

# %%
## 分割训练集和验证集
y[y!=5] = 0; y[y==5] = 1
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=63)
for train_ind, test_ind in split.split(X, y):
    X_train = X[train_ind]
    y_train = y[train_ind]
    X_test = X[test_ind]
    y_test = y[test_ind]

# %%
## PCA降维
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(X_train)
X_train = std.transform(X_train)
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
# 绘制方差累计贡献186
var_ratio = pca.explained_variance_ratio_.cumsum()
plt.plot(range(273,279), var_ratio[272:278], 'b-')
plt.plot([273,276], [var_ratio[275],var_ratio[275]], 'r--')
plt.plot([276,276], [var_ratio[272],var_ratio[275]], 'r--')
plt.text(274.5, 0.9501, "(276, "+str(round(var_ratio[275],6))+")", fontsize=12)
plt.xlabel("Number of Components")
plt.ylabel("Cumsum Variance Ratio")
# 实现降维
X_train = pca.transform(X_train)[:,:276]
X_test = pca.transform(std.transform(X_test))[:,:276]

# %%
## KNN分类
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_range = {'n_neighbors':[3,5,7,9]}
knn_clf = KNeighborsClassifier(weights='distance')
grid = GridSearchCV(knn_clf, param_grid=param_range, cv=3)
grid.fit(X_train, y_train)
plt.bar(['3','5','7','9'], grid.cv_results_['mean_test_score'])
plt.ylim([0.95,0.97])
plt.xlabel("Parameter K")
plt.ylabel("Three Folds Test Accuracy")
knn_clf = grid.best_estimator_

# %%
## 逻辑回归
from sklearn.linear_model import LogisticRegression
param_range = {'penalty':['l2'], 'C':[0.1,1,10]}
logit_clf = LogisticRegression()
grid2 = GridSearchCV(logit_clf, param_grid=param_range, cv=3)
grid2.fit(X_train, y_train)
plt.bar(['10^-1','10^0','10^1'], grid2.cv_results_['mean_test_score'])
plt.ylim([0.92, 0.93])
plt.xlabel("Parameter C")
plt.ylabel("Three Folds Test Accuracy")
logit_clf = grid2.best_estimator_

# %%
## 高斯核SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_range = {'C': [1,10], 'gamma': np.logspace(-5, -3, 3)}
svm_clf = SVC(kernel='rbf') 
grid3 = GridSearchCV(svm_clf, param_grid=param_range)
grid3.fit(X_train, y_train)
plt.bar(['1|10^-5','1|10^-4','1|10^-3','10|10^-5','10|10^-4','10|10^-3'],
        grid3.cv_results_['mean_test_score'])
plt.ylim([0.89,1])
plt.xlabel("Parameter C|gamma")
plt.ylabel("Three Folds Test Accuracy")
svm_clf = grid3.best_estimator_

# %%
## 分类器效果比较
import pandas as pd
accuracy = pd.DataFrame({'Classifier':['KNN','KNN','Logistic','Logistic','SVM','SVM'],
                         'Set':['Train Set','Test Set','Train Set','Test Set','Train Set','Test Set'],
                         'Accuracy':[knn_clf.score(X_train, y_train),knn_clf.score(X_test, y_test),
                                     logit_clf.score(X_train, y_train),logit_clf.score(X_test, y_test),
                                     svm_clf.score(X_train, y_train),svm_clf.score(X_test, y_test)]})
import seaborn as sns
sns.barplot(x='Classifier', y='Accuracy', hue='Set', data=accuracy)
sns.despine()
plt.ylim([0.9,1])

# %%
## 随机森林
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
forest_clf.fit(X_train, y_train)
plt.bar(['Train Set','Test Set','OOB'],
        [forest_clf.score(X_train,y_train),
         forest_clf.score(X_test,y_test),
         forest_clf.oob_score_])
plt.ylim([0.9,1])
plt.xlabel("Different Set")
plt.ylabel("Accuracy")


# %%
## 卷积神经网络
# 重加载数据
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"].astype(int)
X_not5 = X[y!=5]
y_not5 = y[y!=5]
ind = random.sample(range(X_not5.shape[0]),6000)
X_not5 = X_not5[ind]; y_not5 = y_not5[ind]
X = np.vstack([X[y==5], X_not5])
y = np.hstack([y[y==5], y_not5])
y[y!=5] = 0; y[y==5] = 1
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=63)
for train_ind, test_ind in split.split(X, y):
    X_train = X[train_ind]
    y_train = y[train_ind]
    X_test = X[test_ind]
    y_test = y[test_ind]

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 定义初始化函数
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
# 定义超参数和所有节点
height = 28; width = 28; channels = 1
n_inputs = height * width
conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"
conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"
pool3_fmaps = conv2_fmaps
n_fc1 = 64
n_outputs = 2
reset_graph()
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
n_epochs = 10
batch_size = 100
# 开始训练
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)

