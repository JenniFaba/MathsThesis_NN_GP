## Set working directory to current script's location.
fileloc <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(fileloc)
rm(fileloc)
library(e1071) ##for svm
library(neuralnet) #for nnet multilayer, act.fct="logistic","tanh" 
library(nnet) ##other lib for Nnet, only 1-layer, only act.fct="logistic"
library(kernlab) ##for GP
#library(caret) ##for some data handling functions
library(MASS) ##for mvnorm
#library(Metrics)##Measures of prediction error:mse, mae
library(xts) ## handling time series
library(plyr)
library(reshape2)
library(ggplot2)
library(Rcpp)
library(bench)
library(fBasics)



sourceCpp("C-codes/kernelsC.cpp")
set.seed(1996)

### I.Exploring kernels: Drawing some priors

##Define a kernel(single pairs)
# MyKer <- function(x,y) {
#  2*exp(((sin(pi*(x-y)/3))^2)/(-2*0.5^2))
# }

### Hermite2k coef 0
#her2k0 <- function(k){(-1)^k*prod(seq(1,2*k,2))}

###The Hermite based recursive Kernel.
## Activation function Relu
##KHC11 is KH in C++, loaded from kernelsC.cpp
K<-KHC11
  
class(K) <- 'kernel'


# Cov matrix for kernel:
SigmaK <- function(X1,X2) {
  Sigma <- matrix(0, length(X1),length(X2))
  
  for (i in 1:length(X1)) {
    for (j in 1:length(X2)) {
      Sigma[i,j] <- K(as.vector(X1[i]),as.vector(X2[j]),3)  
    }
  }
  return(Sigma)
}




###II: Forecasting with GP
##Data SP500
sp500 = as.xts(read.zoo(file='data/SP500_shiller.csv',sep=',', header=T, format='%Y-%m-%d'))
data=sp500['1900/2012']
plot(data$SP500)
##frequenia de  sampling
tau=1 #data is monthly. Try tau=12 (year), tau=3 (quarterly)

##1. Target as price
target <- data$SP500
sp500PE = na.omit(data$P.E10) ##feature:P/E MA(10)


##2. Target as  Returns
sp500PE = diff(log(sp500PE),diff=tau)
plot(sp500PE)
target=diff(log(data$SP500),diff=tau)  ##compute tau-period returns
target=na.omit(target-mean(na.omit(target)))
plot(target)


#MODEL1 FULL
#Features: lags 1,2,3,4,5 y PE
feat1 = merge(na.omit(lag(target,1)),na.omit(lag(target,2)),na.omit(lag(target,3)),
             na.omit(lag(target,4)),na.omit(lag(target,5)),
             sp500PE, all=FALSE)

#all makes the columns equal

##add TARGET. We want to predict PRICE or RETURN
dataset1 = merge(feat1,target,all=FALSE)
colnames(dataset1) = c("lag.1",  "lag.2", "lag.3", "lag.4", "lag.5",
                      "PE10", "TARGET")

#6features


##  training (75%) and testing (25%). 
T<-nrow(dataset1)
p1=0.75
T_trn1 <- round(p1*T)
trainindex1 <- 1:T_trn1

## data frames
training1 = as.data.frame(dataset1[trainindex1,])
testing1 = as.data.frame(dataset1[-trainindex1,])


#training =  scale(as.data.frame(dataset[trainindex,]))  no creo que valga la pena escalar
#testing = scale(as.data.frame(dataset[-trainindex,]))

rownames(training1) = NULL
rownames(testing1) = NULL



##The kernel (hermites)
#K<-KH

system.time({
  gpfitred1 = gausspr(TARGET~., data=training1[,-c(6)],
                      type="regression",
                      #kernel= "tanhdot",#"rbfdot", #"vanilladot",
                      kernel=K,# MyKer, #
                      error=TRUE,
                      #kernel= Ker4Nnet,  
                      #kpar = list(l=2,s1 = 0.4,s2=1), #list of kernel hyper-parameters for rbf
                      kpar = list(l=3),
                      ## if you make it constant value then does not make mle estimation of sigma
                      #kpar=list(scale=2,offset=2), ##for tanh
                      var = 0.003 # the initial noise variance: 0.001 default min value
                      
  )
})



## predictor 
GPpredict <- predict(gpfitfull1,testing)
gppredict <- predict(gpfitfull1,testing)



saveRDS(gpfitfull1, file = "trained_modelfull1.rds")
model_full1 <- loadRDS("trained_modelfull1.rds")



#MODEL2 FULL
#Features: lags 1,2,3,4,5 y PE y lagsPE 1,2,3
feat = merge(na.omit(lag(target,1)),na.omit(lag(target,2)),na.omit(lag(target,3)),
             na.omit(lag(target,4)),na.omit(lag(target,5)),
             sp500PE,na.omit(lag(sp500PE,1)),na.omit(lag(sp500PE,2)),
             na.omit(lag(sp500PE,3)),na.omit(lag(sp500PE,4)),all=FALSE)



##add TARGET. We want to predict PRICE or RETURN
dataset = merge(feat,target,all=FALSE)
colnames(dataset) = c("lag.1",  "lag.2", "lag.3", "lag.4", "lag.5",
                      "PE10","PE10.1","PE10.2","PE10.3","PE10.4",
                      "TARGET")

#10features


##  training (75%) and testing (25%). 
T<-nrow(dataset)
p=0.75
T_trn <- round(p*T)
trainindex <- 1:T_trn

T-T_trn


## data frames
training = as.data.frame(dataset[trainindex,])
testing = as.data.frame(dataset[-trainindex,])


#training =  scale(as.data.frame(dataset[trainindex,]))  no creo que valga la pena escalar
#testing = scale(as.data.frame(dataset[-trainindex,]))

rownames(training) = NULL
rownames(testing) = NULL

##The kernel (hermites)
#K<-KH

system.time({
  gpfitfull2 = gausspr(TARGET~., data=training,
                      type="regression",
                      #kernel= "tanhdot",#"rbfdot", #"vanilladot",
                      kernel=K,# MyKer, #
                      #kernel= Ker4Nnet,  
                      #kpar = list(l=2,s1 = 0.4,s2=1), #list of kernel hyper-parameters for rbf
                      kpar = list(l=3),
                      ## if you make it constant value then does not make mle estimation of sigma
                      #kpar=list(scale=2,offset=2), ##for tanh
                      var = 0.003 # the initial noise variance: 0.001 default min value
                      
  )
})



## predictor 
GPpredict <- predict(gpfitfull2,testing)
error(gpfitfull2)

saveRDS(gpfitfull2, file = "trained_modelfull2.rds")
modelfull2 <- readRDS("trained_modelfull2.rds")



#MODEL3 FULL
#Features: lags 1,2,3,4,5 y PE, lags PE y todos los indicadores
feat = merge(na.omit(lag(target,1)),na.omit(lag(target,2)),na.omit(lag(target,3)),
             na.omit(lag(target,4)),na.omit(lag(target,5)),
             sp500PE,na.omit(lag(sp500PE,1)),na.omit(lag(sp500PE,2)),
             na.omit(lag(sp500PE,3)),na.omit(lag(sp500PE,4)),
             data$Dividend,data$Consumer.Price.Index, data$Real.Price, data$Earnings,
             data$Real.Dividend, data$Long.Interest.Rate, data$Real.Earnings,
             all=FALSE)



##add TARGET. We want to predict PRICE or RETURN
dataset = merge(feat,target,all=FALSE)
colnames(dataset) = c("lag.1",  "lag.2", "lag.3", "lag.4", "lag.5",
                      "PE10","PE10.1","PE10.2","PE10.3","PE10.4",
                      "Div", "ConsInd", "RealP", "Earn", "RealDiv", "LIR", "RealEarn",#names of other features,
                      "TARGET")
#17 features

##  training (75%) and testing (25%). 
T<-nrow(dataset)
p=0.75
T_trn <- round(p*T)
trainindex <- 1:T_trn

## data frames
training = as.data.frame(dataset[trainindex,])
testing = as.data.frame(dataset[-trainindex,])


#training =  scale(as.data.frame(dataset[trainindex,]))  no creo que valga la pena escalar
#testing = scale(as.data.frame(dataset[-trainindex,]))

rownames(training) = NULL
rownames(testing) = NULL

##The kernel (hermites)
#K<-KH

system.time({
  gpfitred3 = gausspr(TARGET~., data=training[, c(6:10, 18)],
                       type="regression",
                       #kernel= "tanhdot",#"rbfdot", #"vanilladot",
                       kernel=K,# MyKer, #
                       #kernel= Ker4Nnet,  
                       #kpar = list(l=2,s1 = 0.4,s2=1), #list of kernel hyper-parameters for rbf
                       kpar = list(l=3),
                       ## if you make it constant value then does not make mle estimation of sigma
                       #kpar=list(scale=2,offset=2), ##for tanh
                       var = 0.003 # the initial noise variance: 0.001 default min value
                       
  )
})

## predictor 
GPpredict <- predict(gpfitfull3,testing)

saveRDS(gpfitfull3, file = "trained_modelfull3.rds")
modelfull3 <- readRDS("trained_modelfull3.rds")


#MODEL4 FULL
#Features: PE, y todos los indicadores
feat = merge(sp500PE,data$Dividend,data$Consumer.Price.Index, data$Real.Price, data$Earnings,
             data$Real.Dividend, data$Long.Interest.Rate, data$Real.Earnings,
             all=FALSE)



##add TARGET. We want to predict PRICE or RETURN
dataset = merge(feat,target,all=FALSE)
colnames(dataset) = c("PE10","Div", "ConsInd", "RealP", "Earn", "RealDiv", "LIR", "RealEarn",#names of other features,
                      "TARGET")
#17 features

##  training (75%) and testing (25%). 
T<-nrow(dataset)
p=0.75
T_trn <- round(p*T)
trainindex <- 1:T_trn

## data frames
training = as.data.frame(dataset[trainindex,])
testing = as.data.frame(dataset[-trainindex,])


#training =  scale(as.data.frame(dataset[trainindex,]))  no creo que valga la pena escalar
#testing = scale(as.data.frame(dataset[-trainindex,]))

rownames(training) = NULL
rownames(testing) = NULL

##The kernel (hermites)
#K<-KH

system.time({
  gpfitred4 = gausspr(TARGET~., data=training[,c(1,9)],
                      type="regression",
                      #kernel= "tanhdot",#"rbfdot", #"vanilladot",
                      kernel=K,# MyKer, #
                      #kernel= Ker4Nnet,  
                      #kpar = list(l=2,s1 = 0.4,s2=1), #list of kernel hyper-parameters for rbf
                      kpar = list(l=3),
                      ## if you make it constant value then does not make mle estimation of sigma
                      #kpar=list(scale=2,offset=2), ##for tanh
                      var = 0.003 # the initial noise variance: 0.001 default min value
                      
  )
})






##BOOTSTRAP
require(bootstrap)
require(boot)
library(boot)
modelfull1<- readRDS("trained_modelfull1.rds")
modelred1 <- readRDS("trained_modelred1.rds")
modelfull2 <- readRDS("trained_modelfull2.rds")
modelfull3 <- readRDS("trained_modelfull3.rds")
modelred3 <- readRDS("trained_modelred3.rds")
n_test<-round((1-p1)*T)
n_test

#Estadísticos a bootstrapear

#MODELO1
e1=testing1[,"TARGET"]-predict(gpfitfull1, testing1)
e2=testing1[,"TARGET"]-predict(gpfitred1, testing1[,-c(6)])
par(mfrow = c(1, 2))
plot(e1)
lines(e1,type="l", col=2)

plot(e2)
lines(e2,type="l", col=2)


aov.boot=function(data,ind)
{
  return(var(data[ind,2])/var(data[ind,1]))
}

boot.aov=boot(data=cbind(e1,e2),statistic=aov.boot,R=999)

boot.ci(boot.aov)




#MODELO2
e2=testing1[,"TARGET"]-predict(gpfitfull1, testing1)
e1=testing[,"TARGET"]-predict(gpfitfull2, testing)
par(mfrow = c(1, 2))
#corresponde a 'una observación', en el bootstrap, tenemos 999 más de estos
plot(e1)
lines(e1,type="l", col=2)

plot(e2)
lines(e2,type="l", col=2)


aov.boot=function(data,ind)
{
  return(var(data[ind,2])/var(data[ind,1]))
}

boot.aov=boot(data=cbind(e1,e2),statistic=aov.boot,R=999, )

boot.ci(boot.aov, conf=0.5)


#MODELO3
e2=testing[,"TARGET"]-predict(gpfitred3, testing[,c(6:10, 18)])
e1=testing[,"TARGET"]-predict(gpfitfull3, testing)
par(mfrow = c(1, 2))
#corresponde a 'una observación', en el bootstrap, tenemos 999 más de estos
plot(e1)
lines(e1,type="l", col=2)

plot(e2)
lines(e2,type="l", col=2)


aov.boot=function(data,ind)
{
  return(var(data[ind,2])/var(data[ind,1]))
}

boot.aov=boot(data=cbind(e1,e2),statistic=aov.boot,R=999, )

boot.ci(boot.aov, conf=0.95)


#MODELO4
e2=testing[,"TARGET"]-predict(gpfitred4, testing[,c(1, 9)])
e1=testing[,"TARGET"]-predict(gpfitfull4, testing)
par(mfrow = c(1, 2))
#corresponde a 'una observación', en el bootstrap, tenemos 999 más de estos
plot(e1)
lines(e1,type="l", col=2)

plot(e2)
lines(e2,type="l", col=2)


aov.boot=function(data,ind)
{
  return(var(data[ind,2])/var(data[ind,1]))
}

boot.aov=boot(data=cbind(e1,e2),statistic=aov.boot,R=999, )

boot.ci(boot.aov, conf=0.95)

