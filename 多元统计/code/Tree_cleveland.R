#读取数据
setwd("C:\\Users\\Mac\\Desktop\\过程\\学业\\本科\\专业课\\多元统计\\CART\\data")
CHD<-read.table("cleveland.txt", header=T)
CHD<-CHD[, -15]

library(rpart)
library(rpart.plot)
out.rpart<-rpart(diag~.,CHD)

#绘出分类树
rpart.plot(out.rpart, type=1, extra=1)
text(out.rpart, use.n=T, all=T)

CHD2<-cbind(CHD$age, CHD$diag)
Age<-sort(unique(CHD2[,1]))
T<-length(Age)

n<-matrix(nrow=2, ncol=2)
tau_L<-numeric(T)
tau_R<-numeric(T)
delta<-numeric(T)

for (i in 1:T)
{
  #当前分法下的左右节点
  CHD2_left<-matrix(CHD2[CHD2[,1]<=Age[i], ],ncol=2)
  CHD2_right<-matrix(CHD2[CHD2[,1]>Age[i], ],ncol=2)
  
  #左右节点和分类列联表
  n[1, 1]<-sum(CHD2_left[, 2]==1)
  n[1, 2]<-sum(CHD2_left[, 2]==2)
  n[2, 1]<-sum(CHD2_right[, 2]==1)
  n[2, 2]<-sum(CHD2_right[, 2]==2)
  
  n_row<-rowSums(n)
  n_col<-colSums(n)
  n_sum<-sum(n)
  
  #左右节点的熵
  tau_L[i]<--n[1,1]/n_row[1]*log(n[1,1]/n_row[1])-n[1,2]/n_row[1]*log(n[1,2]/n_row[1])
  tau_R[i]<--n[2,1]/n_row[2]*log(n[2,1]/n_row[2])-n[2,2]/n_row[2]*log(n[2,2]/n_row[2])
  
  #当前分法的减少熵
  delta[i]<-0.6899-n_row[1]/n_sum*tau_L[i]-n_row[2]/n_sum*tau_R[i]
}

par(mfrow=c(1,2))

#左右节点熵
plot(Age, tau_R, type='l', col='red', xlab="Age at Split", ylab="i(tau)")
points(Age, tau_L, type='l', col="blue")
legend("bottom", c("left", "right"), col=c('blue', 'red'), lty=c(1,1), bty='n')

#减少熵
plot(Age, delta, type='l', col='red', xlab="Age at Split", ylab="Goodness of Split")
