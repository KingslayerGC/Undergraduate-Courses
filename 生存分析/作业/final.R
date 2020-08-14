## 读取数据
data<-read.csv("handouts_fhs.csv")
library(survival)

## 处理缺失值
# 查看数据缺失情况
data[data=='']<-NA
apply(data, 2, function(x){sum(is.na(x))})
# 用中位数或众数填补异常的缺失值
getmode<-function(v)
{uniqv<-unique(v);uniqv[which.max(tabulate(match(v, uniqv)))]}
fillna<-function(v, method='median', period=1)
{if(method=='mode') {x<-getmode(v)} else {x<-median(v, na.rm = TRUE)}
if(period==2) {v[is.na(v)&!(is.na(data['age2']))]<-x}
else if(period==3) {v[is.na(v)&!(is.na(data['age3']))]<-x}
else {v[is.na(v)&!(is.na(data['age1']))]<-x}
v}
for (col in list('totchol1','cigpday1','bmi1','bpmeds1','heartrte1','glucose1'))
{v<-as.matrix(data[col])
if(col=='bpmeds1') {data[col]<-fillna(v,method='mode')}
else {data[col]<-fillna(v)}}
for (col in list('totchol2','cigpday2','bmi2','bpmeds2','heartrte2','glucose2','bmidiff'))
{v<-as.matrix(data[col])
if(col=='bpmeds2') {data[col]<-fillna(v,method='mode',period=2)}
else {data[col]<-fillna(v,period=2)}}
for (col in list('totchol3','cigpday3','bmi3','bpmeds3','heartrte3','glucose3','hdlc3','ldlc3'))
{v<-as.matrix(data[col])
if(col=='bpmeds3') {data[col]<-fillna(v,method='mode',period=3)}
else {data[col]<-fillna(v,period=3)}}
# 查看处理后的数据缺失情况
apply(data, 2, function(x){sum(is.na(x))})

## CHD病因研究
# 构造数据集
data1<-data.frame(matrix(NA,4434,21))
colnames(data1)<-c('anychd','timechd','sex','totchol','age','sysbp','diabp','cursmoke','cigpday','bmi','diabetes',
                   'bpmeds','heartrte','glucose','prevchd','prevap','prevmi','prevstrk','prevhyp','hdlc','ldlc')
rownames(data1)<-rownames(data)
data1$anychd<-data$anychd;data1$timechd<-data$timechd
row3<-!is.na(data$prevchd3)&data$prevchd3=='Yes'
row2<-!is.na(data$prevchd2)&data$prevchd2=='Yes'
row1<-data$prevchd1=='Yes'
row4<-!(row3|row2|row1)
data1[row3,3:21]<-data[row3,56:74]
data1[row2,3:21]<-data[row2,37:55]
data1[row1,3:21]<-data[row1,18:36]
data1[row4,3:21]<-data[row4,18:36]
data1[!is.na(data$prevchd2)&row4,3:21]<-data[!is.na(data$prevchd2)&row4,37:55]
data1[!is.na(data$prevchd3)&row4,3:21]<-data[!is.na(data$prevchd3)&row4,56:74]
# KM Plot
Y<-Surv(data1$timechd, data1$anychd=='Yes')
data1$bmi_cat<-data$bmi
data1$bmi_cat[data1$bmi<25]<-'Under/Normal Weight'
data1$bmi_cat[data1$bmi>=25]<-'Over Weight'
par(mfrow=c(1,2))
kmfit<-survfit(Y~data1$sex)
plot(kmfit,col=c('blue','red'),xlab="Survival Time",ylab="No Disease Probabilities")
legend('bottom', c("Female","Male"),
       lty="solid", col=c('blue','red'), bty='n')
kmfit<-survfit(Y~data1$bmi_cat)
plot(kmfit,col=c('red','blue'),xlab="Survival Time",ylab="No Disease Probabilities")
legend('bottom', c("Under/Normal Weight (BMI<25)","Over Weight (BMI>25)"),
       lty="solid", col=c('blue','red'), bty='n')
# Log Rank Test
survdiff(Y~data1$bpmeds)
survdiff(Y~data1$bpmeds+strata(data1$bmi_cat,data1$diabetes))
# Log-Log Plot
par(mfrow=c(1,2))
kmfit<-survfit(Y~data1$diabetes)
plot(kmfit, fun='cloglog',col=c('blue','red'),
     xlab="time in days using logarithmicscale",ylab="log-log survival")
legend('top',c("No Diabetes","Diabetes"),lty="solid",col=c('blue','red'),bty='n')
kmfit<-survfit(Y~data1$sex)
plot(kmfit, fun='cloglog',col=c('blue','red'),
     xlab="time in days using logarithmicscale",ylab="log-log survival")
legend('top',c("Female","Male"),lty="solid",col=c('blue','red'),bty='n')
# Test of PH Assumption
mod1<-coxph(Y~sex+totchol+age+cursmoke+bmi+diabetes+bpmeds, data=data1)
cox.zph(mod1,transform=rank)
# Schoenfeld残差图
par(mfrow=c(1,2))
plot(cox.zph(mod1,transform=rank),var='age')
plot(cox.zph(mod1,transform=rank),var='bmi')
# Stratified Cox Model
data1$age_cat<-cut(data1$age,breaks=c(-Inf,50,60,70,Inf))
data1$totchol_cat<-cut(data1$totchol,breaks=c(-Inf,200,330,Inf),labels=c("lowq","normalq","highq"))
mod2<-coxph(Y~sex+cursmoke+bmi+bpmeds+strata(diabetes,age_cat,totchol_cat), data=data1)
# AFT Model
data1$timechd2<-data1$timechd
data1$timechd2[data1$timechd2==0]<-0.0001
Y2<-Surv(data1$timechd2,data1$anychd=='Yes')
mod3<-survreg(Y2~sex+cursmoke+bmi+bpmeds+diabetes+age+totchol, data=data1, dist='lognormal')
summary(mod3)
# Survival Curve
pattern1<-data1[2000,];pattern2<-data1[3000,]
i<-0;par(mfrow=c(1,2))
for (pattern in list(pattern1,pattern2))
{
i<-i+1
plot(survfit(mod1, newdata=pattern),conf.int=F,
     xlim=c(0.7,20),ylim=c(1,0.6),col='blue',
     xlab="Survival Time",ylab="No Disease Probabilities")
pct=0:20/100
days=predict(mod3,newdata=pattern,type='quantile', p=pct)
survival=1-pct
lines(days,survival,col='red')
legend('bottom',c("Cox Model","AFT model"),col=c('blue','red'),lty='solid',bty='n')
title(paste("Sample",as.character(i)))
}

## 死亡原因研究
# survSplit
data2<-subset(data,select=c('death','mi_fchd','stroke','timemifc','timestrk','timedth'))
data2<-data2[data2$mi_fchd=='Yes'|data2$stroke=='Yes',]
data2$death[data2$death=='Yes']<-1;data2$death[data2$death=='No']<-0
data2$death<-as.numeric(data2$death)
data2<-survSplit(data=data2,cut=data2$timedth[data2$death==1],
                 end='timedth', event='death', start='start', id='id')
data2$timemifc<-data2$timedth-data2$timemifc
data2$timemifc[data2$timemifc<0]<-NA
data2$timestrk<-data2$timedth-data2$timestrk
data2$timestrk[data2$timestrk<0]<-NA
data2<-data2[!is.na(data2$timemifc)|!is.na(data2$timestrk),]
data2[data2$id==409,][110:120,]
# Extend Model
Y3<-Surv(data2$timedth,data2$death)
mod4<-coxph(Y3~timemifc+timestrk,data=data2)
cox.zph(mod4,transform=rank)
par(mfrow=c(1,2))
plot(cox.zph(mod4,transform=rank),var='timemifc')
plot(cox.zph(mod4,transform=rank),var='timestrk')
mod5<-survreg(Y3~timemifc+timestrk,data=data2,dist='weibull')
summary(mod5)


