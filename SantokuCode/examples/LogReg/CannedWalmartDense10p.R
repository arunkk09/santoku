#Copyright 2015 Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

source("../../src/Classes.R")
source("../../src/LogReg/LogRegDense.R")
library(data.table)
library(Matrix)
options(width=190)
options(scipen=999)

wtr = read.csv("../../../SantokuData/Walmart/DWTtraintestSing10p.csv")
wts = read.csv("../../../SantokuData/Walmart/DWTholdSing10p.csv")
Str = read.csv("../../../SantokuData/Walmart/DWTtraintestMultS10p.csv") #entity table alone
Sts = read.csv("../../../SantokuData/Walmart/DWTholdMultS10p.csv")
R1 = read.csv("../../../SantokuData/Walmart/DWTMultR1.csv")
R2 = read.csv("../../../SantokuData/Walmart/DWTMultR210p.csv")


#factorized learning and scoring
kfkds = list(KFKD(EntCol="storefk", AttCol="storefk",UseFK=TRUE), KFKD(EntCol="purchaseidfk",AttCol="purchaseidfk",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)

pt = proc.time(); mylogregmult = LogRegDenseMultLearn(multtrain, lambda = 6000, alpha = 0.0002, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes

pt = proc.time(); predmult = LogRegDenseMultScore(mylogregmult, multtest); print(proc.time() - pt) #takes 
tabpred = table(data.matrix(multtest@Target), predmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives % accuracy

pt = proc.time(); tpredmult = LogRegDenseMultScore(mylogregmult, multtrain); print(proc.time() - pt) #takes 
tabpred = table(data.matrix(multtrain@Target), tpredmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives % accuracy


#single table learning and scoring
singtrain = SingData(Target=as.data.frame(wtr[,1]), TargetName="weekly_sales", SingTable=wtr[,setdiff(names(wtr),c("weekly_sales","storefk","purchaseidfk"))], FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(wts[,1]), TargetName="weekly_sales", SingTable=wts[,setdiff(names(wtr),c("weekly_sales","storefk","purchaseidfk"))], FDs=list())

pt = proc.time(); mylogregsing = LogRegDenseSingLearn(singtrain, lambda = 6000, alpha = 0.0002, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 

pt = proc.time(); predsing = LogRegDenseSingScore(mylogregsing, singtest); print(proc.time() - pt) #takes 
tabpred = table(data.matrix(singtest@Target), predsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives % accuracy

pt = proc.time(); tpredsing = LogRegDenseSingScore(mylogregsing, singtrain); print(proc.time() - pt) #takes 
tabpred = table(data.matrix(singtrain@Target), tpredsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives % accuracy


#verify equivalence of learned models from sing and factorized
print(mylogregmult@LossVal)
print(mylogregsing@LossVal)
print(unlist(mylogregmult@Coefs))
print(unlist(mylogregsing@Coefs))
