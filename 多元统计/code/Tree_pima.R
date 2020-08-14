#读取数据
setwd("C:\\Users\\Mac\\Desktop\\过程\\学业\\本科\\专业课\\多元统计\\CART\\data")
Pima<-read.table("pima.txt", header=T)
Pima$class<-factor(Pima$class)

library(rpart)
library(rpart.plot)

out<-rpart(class~.,Pima)

#绘出分类树
rpart.plot(out, type=1, extra=1)

#阿尔法取值表
out$cptable

#剪枝并绘出新分类树
out.prune<-prune(out, cp=out$cptable[5,1])
rpart.plot(out.prune, type=1, extra=1)