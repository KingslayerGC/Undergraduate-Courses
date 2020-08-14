##9.2
#读取数据
setwd("C:\\Users\\Mac\\Desktop\\过程\\学业\\本科\\专业课\\多元统计\\CART\\data")
CHD<-read.table("cleveland.txt", header=T)
CHD<-CHD[, -15]

CHD2<-cbind(CHD$age, CHD$diag)
Age<-sort(unique(CHD2[,1]))
T<-length(Age)

n<-matrix(nrow=2, ncol=2)
tau_L<-numeric(T)
tau_R<-numeric(T)
delta<-numeric(T)
delta_max<-0
threshold_best<-0

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
  
  #根节点和左右节点的基尼系数
  root <- 1-(n_col[1]/n_sum)^2-(n_col[2]/n_sum)^2
  tau_L[i] <- 1-(n[1,1]/n_row[1])^2-(n[1,2]/n_row[1])^2
  tau_R[i] <- 1-(n[2,1]/n_row[2])^2-(n[2,2]/n_row[2])^2
  
  #当前分法的减少基尼系数
  delta[i] <- root-n_row[1]/n_sum*tau_L[i]-n_row[2]/n_sum*tau_R[i]
  
  #找出最好的split
  if (is.na(delta[i])) {delta[i]<-0}
  if (delta[i]> delta_max)
  {
    delta_max<-delta[i]
    split_best<-n
    threshold_best<-Age[i]
  }
}

#最佳的split阈值及其效果
split_best
threshold_best

par(mfrow=c(1,2))

#左右节点基尼系数图
plot(Age, tau_R, type='l', col='red', xlab="Age at Split", ylab="i(tau)")
points(Age, tau_L, type='l', col="blue")
legend("bottom", c("left", "right"), col=c('blue', 'red'), lty=c(1,1), bty='n')

#减少基尼系数图
plot(Age, delta, type='l', col='red', xlab="Age at Split", ylab="Goodness of Split")

#定义一个计算最佳阈值的函数
tree <-function(column)
{
  xy<-cbind(CHD[[column]], CHD$diag)
  x_unique<-sort(unique(xy[,1]))
  T<-length(x_unique)
  
  delta_max<-0
  
  for (i in 1:T)
  {
    #当前分法下的左右节点
    xy_left<-matrix(xy[xy[,1]<=x_unique[i], ],ncol=2)
    xy_right<-matrix(xy[xy[,1]>x_unique[i], ],ncol=2)
    
    #左右节点和分类列联表
    n[1, 1]<-sum(xy_left[, 2]==1)
    n[1, 2]<-sum(xy_left[, 2]==2)
    n[2, 1]<-sum(xy_right[, 2]==1)
    n[2, 2]<-sum(xy_right[, 2]==2)
    
    n_row<-rowSums(n)
    n_col<-colSums(n)
    n_sum<-sum(n)
    
    #根节点和左右节点的基尼系数
    root <- 1-(n_col[1]/n_sum)^2-(n_col[2]/n_sum)^2
    tau_L <- 1-(n[1,1]/n_row[1])^2-(n[1,2]/n_row[1])^2
    tau_R <- 1-(n[2,1]/n_row[2])^2-(n[2,2]/n_row[2])^2
    
    #当前分法的减少基尼系数
    delta <- root-n_row[1]/n_sum*tau_L-n_row[2]/n_sum*tau_R
    
    #找出最好的split
    if (is.na(delta)) {delta <- 0}
    if (delta > delta_max) {delta_max<-delta}
  }
  
  print(column)
  print(delta_max)
}

#打印变量名及最佳阈值
for (col in colnames(CHD)) {if (col != 'diag') {tree(col)}}

##9.8
#读取数据
setwd("C:\\Users\\Mac\\Desktop\\过程\\学业\\本科\\专业课\\多元统计\\CART\\data")
vehicle<-read.table("vehicle3.txt", header=T)
vehicle$class<-factor(vehicle$class)

#绘制阿尔法&树大小-误判率图
library(rpart)
out<-rpart(class~.-pam, vehicle, cp=1e-10)
plotcp(out)

#阿尔法取值表
out$cptable

#剪枝并绘出新分类树
out.prune<-prune(out, cp=out$cptable[12,1])
library(rpart.plot)
rpart.plot(out.prune, type=1, extra=1)