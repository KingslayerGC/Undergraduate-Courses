## 读取数据
data<-read.csv("ACTG175(speff2trial).txt")
library(survival)
Y<-Surv(data$days, data$cens==1)

## log rank test
survdiff(Y~data$arms)

## weighted log rank test
survdiff(Y~data$arms, rho=0.5)

## 读取数据
data<-read.csv("wcgsdata.csv")
library(survival)
Y<-Surv(data$Time169, data$Chd69==1)

## 生存曲线
kmfit<-survfit(Y~data$Dibpat0)
plot(kmfit, xlab="survival time in days",
     ylab="no disease probabilities", col=c("red","blue"))
legend("bottomleft", c("behaviour type A","behaviour type B"),
       lty="solid", col=c("red","blue"))

## log rank test
survdiff(Y~data$Dibpat0)

## stratified log rank test
data$Weight0<-data$Weight0 * 0.45359
data$Height0<-data$Height0 * 0.0254
data$BMI<-data$Weight0 / (data$Height0)^2


data$BMI<-cut(data$BMI, breaks=c(-Inf,18.5,25.0,30.0,Inf),
              labels=c("underweight","healthy weight","overweight","obese"),
              include.lowest=TRUE, right=FALSE)

data$Ncigs0[data$Ncigs0!=0]<-"smoker"
data$Ncigs0[data$Ncigs0==0]<-"not smoker"

survdiff(Y ~ data$Dibpat0 + strata(data$BMI,data$Ncigs0))
