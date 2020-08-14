## 生成数据
X <- matrix(c(1,3,2,4,1,5,5,5,5,7,4,9,2,8,3,10), nrow=8, byrow=T)

## 展示点分布
plot(X[,1], X[,2], col='red', pch=16, xlab="X1", ylab="X2", ylim=c(2,11))
text(X[,1]+0.1, X[,2], 1:8)

## Agnes分层聚类
library(cluster)
#out <- agnes(X, method='s'); plot(out)
#out <- agnes(X, method='c'); plot(out)
out <- agnes(X, method='a'); plot(out)
# 结果概述
summary(out)

## Diana分层聚类
out <- diana(X); plot(out)

## Kmeans快速聚类
out <- kmeans(X,3)
# 结果展示
out$cluster; out$centers; out$withinss

## PAM快速聚类
out <- pam(X,3); plot(out)
# 结果展示
out$clustering; out$id.med; out$medoids

## Fanny聚类
out <- fanny(X,3)
# 结果展示
out$membership; out$clustering
