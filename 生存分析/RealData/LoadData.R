
#1.
library(titanic)
data("titanic_train")
data("titanic_test")
titanic_train

#2.
library(survival)
ovarian
bladder
cgd
colon
flchain
mgus
pbc
lung
heart
veteran

#3.
library(epitools)
data("wcgs")

#4.
library(LocalControl)
framingham

#5.
#set your Working directory
setwd("E:/myfile/学习/lecture/生存分析/助教工作/RealData")
addicts <- read.table("addicts.dat",
                      col.names=c('iD','clinic','status','survt','prison',
                                  'dose'),skip=19)

#6.
anderson <- read.table("anderson.dat",col.names=c('Surv','Relapse','Sex','log WBC',
                                                  'Rx'))

#7.
library(tpr)
data(dnase)

#
rhDNase <- read.table("rhDNase.dat",col.names=c("id","trt","fev","fev2","time1","time2","status","etype","enum","enum1","enum2"))
rhDNase$fevc    <- rhDNase$fev - mean(rhDNase$fev[rhDNase$enum == 1])
rhDNase$gtime   <- rhDNase$time2 - rhDNase$time1
rhDNase$status1 <- ifelse(rhDNase$etype == 1, rhDNase$status, 0)
rhDNase[1:10, c("id","enum","etype","time1","time2","gtime",
                "status","status1","trt","fev","fevc")]

#8.
library(speff2trial)
data(ACTG175)

#9.
library(etm)
data(abortion)

#10.
library(mvna)
data(sir.adm)

#11.
library(mvna)
data(sir.cont)

#12.
library(kmi)
data(icu.pneu)






