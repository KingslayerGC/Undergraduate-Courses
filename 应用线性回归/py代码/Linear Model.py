#!/usr/bin/env python
# coding: utf-8

# In[179]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'

import warnings
warnings.filterwarnings("ignore")


# In[180]:


## 获取数据

import numpy as np
import pandas as pd
data1 = pd.read_table(r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\应用线性回归\Data_5e\CH06FI05.txt", header=None, sep='  ')
data1.columns = ['x1', 'x2', 'y']

data2 = pd.read_table(r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\应用线性回归\Data_5e\CH07TA01.txt", header=None, sep='  ')
data2.columns = ['x1', 'x2', 'x3', 'y']

data3 =  pd.read_table(r"C:\Users\Mac\Desktop\过程\学业\本科\专业课\应用线性回归\Data_5e\CH11TA01.txt", header=None, sep='  ')
data3.columns = ['x', 'y']


# -----------------------------------多元线性回归---------------------------------

# In[181]:


## sklearn

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression(fit_intercept=True)
linear_reg.fit(data1[['x1', 'x2']], data1['y'])

# 输出系数
linear_reg.coef_
linear_reg.intercept_


# In[182]:


## statsmodel

import statsmodels.formula.api as smf
result = smf.ols('y ~x1 + x2', data=data1).fit()

# 输出回归结果概述
print(result.summary())

# 输出ANOVA表
import statsmodels.api as sm
table = sm.stats.anova_lm(result, typ=1)
table

# 预测期望的置信区间和区间估计
pred_df = pd.DataFrame([[65.4, 17.6]], columns=['x1', 'x2'])
predictions = result.get_prediction(pred_df)
predictions.summary_frame(alpha=0.05)


# 模型属性文档：https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html?highlight=olsresult

# --------------------------------模型诊断---------------------------------------

# In[194]:


## 模型诊断（初步）

# 得到相关矩阵
print('相关矩阵：')
data1.corr()

# 绘制散点图
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data1, diag_kind='kde', plot_kws={'alpha': 0.5})

# qq图
fig, ax = plt.subplots(1,1)
from scipy import stats
stats.probplot(data1['x1'], dist='norm', plot=ax)
plt.title('QQ Plot')


# In[71]:


## 偏相关系数

fit1 = smf.ols('y ~x1', data=data2).fit()
fit2 = smf.ols('y ~x2', data=data2).fit()
fit12 = smf.ols('y ~x1 + x2', data=data2).fit()
fit = smf.ols('y ~x1 + x2 + x3', data=data2).fit()

# 计算偏相关系数（方法一）
sse1 = fit1.ssr
sse2 = fit2.ssr
sse12 = fit12.ssr
sse = fit.ssr
ssr1_2 = sse2 - sse12
R1_2 = ssr1_2 / sse2
R1_2

# 计算偏相关系数（方法二）
e1 = fit2.resid
e2 = smf.ols('x1 ~x2', data=data2).fit().resid
np.corrcoef(e1, e2)[0, 1] ** 2


# In[ ]:


## 模型选择
### python没有向前向后算法，以下地址给出一个可能的手写结果


# https://planspace.org/20150423-forward_selection_with_statsmodels/

# In[175]:


## 模型诊断

# outlier of y
import scipy.stats as ss
p = 3
n = data2.shape[0]
a = 0.05
influence = fit12.get_influence().summary_frame()
influence['residual'] = fit12.resid
influence['t'] = influence['residual'] * ((n-p-1) / (fit12.ssr * (1 - influence['hat_diag'])- influence['residual']**2)) ** 0.5
print('Outlier of Y：')
print('thresh: ', ss.t.ppf(1 - a/2, n-p-1))
print('max of abs(t): ', abs(influence['t']).max(), '\n')

# outlier of x
print('Outlier of X：')
print('thresh: ', 2 * p /n)
print('max of h: ', influence['hat_diag'].max(), '\n')

# influential cases（诊断异常值）
print('Influence''s Cases：')
print(influence[['dfb_Intercept', 'dfb_x1', 'dfb_x2', 'cooks_d']], '\n')

# VIF（诊断共线性问题）
print('对角线为VIF：')
print(np.linalg.inv(data2[['x1', 'x2', 'x3']].corr()), '\n\n')

# Add Variable Plot
import seaborn as sns
print('Add Variable Plot：')
sns.regplot(e2, e1, ci=None)


# Influence分析文档：https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.html#statsmodels.stats.outliers_influence.OLSInfluence

# -----------------------------模型改进-------------------------------

# In[201]:


## Weighted Least Squares（针对 Unequal Error Variance）

# 权重回归
data3['abs_resid'] = abs(smf.ols('y ~x', data=data3).fit().resid)
weights = smf.ols('abs_resid ~x', data=data3).fit().fittedvalues ** -2
print(smf.wls('y ~x', data=data3, weights=weights).fit().summary())


# In[230]:


## Penalty Regression（针对共线性问题）

# Ridge Regression
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.1)


# In[235]:


## Robust Regression（针对异常值问题）

# Robust Regression,此处使用何种robust并不清楚
smf.rlm('y ~x', data=data3).fit()


# In[ ]:




