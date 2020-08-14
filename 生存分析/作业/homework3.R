## 读取数据
data<-read.csv("wcgsdata.csv")
library(survival)
Y<-Surv(data$Time169, data$Chd69==1)

## 处理BMI,Ncigs0,Chol0,Dibpat0数据
data$Weight0<-data$Weight0 * 0.45359
data$Height0<-data$Height0 * 0.0254
data$BMI<-data$Weight0 / (data$Height0)^2

data$Ncigs0[data$Ncigs0!=0]<-"smoker"
data$Ncigs0[data$Ncigs0==0]<-"not smoker"

med <- median(as.numeric(data$Chol0[data$Chol0!='.']))
data$Chol0[data$Chol0=='.']<-med
data$Chol0<-as.numeric(data$Chol0)

## Cox Model
mod1<-coxph(Y ~ Age0 + BMI + Sbp0 + Dbp0 + Chol0 + Ncigs0 + Dibpat0, data=data)
summary(mod1)

## Test of PH Assumption
cox.zph(mod1,transform=rank)

## Log-Log Plot
for (col in list("Age0", "BMI", "Sbp0", "Dbp0", "Chol0"))
{
v<-as.matrix(data[col])
med <- median(v, na.rm=T)
v[v<=med]<-1
v[v>med]<-2
kmfit<-survfit(Y ~ v)
plot(kmfit, fun='cloglog',
     xlab="time in days using logarithmicscale", ylab="log-log survival",
     main=c("log-log curves by ",col))
}

for (col in list("Ncigs0", "Dibpat0"))
{
  v<-as.matrix(data[col])
  kmfit<-survfit(Y ~ v)
  plot(kmfit, fun='cloglog',
       xlab="time in days using logarithmicscale", ylab="log-log survival",
       main=c("log-log curves by ",col))
}

## Stratified Cox Model
mod2<-coxph(Y ~ Age0 + BMI + Sbp0 + Dbp0 + Chol0 + Dibpat0 + strata(Ncigs0), data=data)
summary(mod2)

## Extended Cox Model
data.cp <- survSplit(data,cut=data$Time169[data$Chd69==1],
                     end='Time169', event='Chd69', start='start', id='id')

data.cp$tDibpat0 <- data.cp$Dibpat0 * data.cp$Time169

coxph(Surv(data.cp$start, data.cp$Time169, data.cp$Chd69)
      ~Age0 + BMI + Sbp0 + Dbp0 + Chol0 + Ncigs0 + Dibpat0 + tDibpat0,
      data=data.cp)

## Log Normal Survival Model
mod3<-survreg(Y ~ Age0 + BMI + Sbp0 + Dbp0 + Chol0 + Ncigs0 + Dibpat0, data=data, dist='lognormal')
summary(mod3)

## Predict
pattern1<-data.frame(Age0=55,BMI=28.5,Sbp0=138,Dbp0=90,Chol0=280,Ncigs0='smoker',Dibpat0=1)
pattern2<-data.frame(Age0=42,BMI=22,Sbp0=120,Dbp0=80,Chol0=180,Ncigs0='not smoker',Dibpat0=1)

summary(survfit(mod1, newdata=pattern1), times=365*5)
summary(survfit(mod1, newdata=pattern2), times=365*5)
summary(survfit(mod1, newdata=pattern1), times=365*8)
summary(survfit(mod1, newdata=pattern2), times=365*8)
predict(mod3, newdata=pattern1, type='quantile', p=245:350/1000)
predict(mod3, newdata=pattern2, type='quantile', p=105:220/10000)
