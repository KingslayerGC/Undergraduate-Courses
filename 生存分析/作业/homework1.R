## 读取数据
data<-read.csv("ACTG175(speff2trial).txt")
library(survival)
Y<-Surv(data$days, data$cens==1)

## 生存曲线
kmfit <- survfit(Y ~data$arms)
plot(kmfit, xlab="survival time in days", ylab="survival probabilities",
     col=c("red","blue","green","orange"))
legend("bottomleft", c("treatment 0","treatment 1","treatment 2","treatment 3"),
       lty="solid", col=c("red","blue","green","orange"))

## 生存函数点估计及区间估计
summary(kmfit, times=365)

summary(kmfit, times=730)