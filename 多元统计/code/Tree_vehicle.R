#读取数据
setwd("C:\\Users\\Mac\\Desktop\\过程\\学业\\本科\\专业课\\多元统计\\CART\\data")
vehicle<-read.table("vehicle3.txt", header=T)
vehicle$class<-factor(vehicle$class)

library(rpart)
library(rpart.plot)

#绘制阿尔法&树大小-误判率图
out<-rpart(class~.-pam, vehicle, cp=1e-10)
plotcp(out)

#阿尔法取值表
out$cptable

#剪枝并绘出新分类树
out.prune<-prune(out, cp=out$cptable[12,1])
rpart.plot(out.prune, type=1, extra=1)