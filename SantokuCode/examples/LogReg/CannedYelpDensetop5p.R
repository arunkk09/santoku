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

ytr = read.csv("../../../SantokuData/Yelp/DYRtraintestSingtop5p.csv")
yts = read.csv("../../../SantokuData/Yelp/DYRholdSingtop5p.csv")
Str = read.csv("../../../SantokuData/Yelp/DYRtraintestMultStop5p.csv") #entity table alone
Sts = read.csv("../../../SantokuData/Yelp/DYRholdMultStop5p.csv")
R1 = read.csv("../../../SantokuData/Yelp/DYRMultR1top5p.csv")
R2 = read.csv("../../../SantokuData/Yelp/DYRMultR2top5p.csv")


#factorized learning and scoring
kfkds = list(KFKD(EntCol="useridfk", AttCol="useridfk",UseFK=TRUE), KFKD(EntCol="businessidfk",AttCol="businessidfk",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)

pt = proc.time(); mylogregmult = LogRegDenseMultLearn(multtrain, lambda = 1000, alpha = 0.0003, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 0.8s

pt = proc.time(); predmult = LogRegDenseMultScore(mylogregmult, multtest); print(proc.time() - pt) #takes 0.03s
tabpred = table(data.matrix(multtest@Target), predmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives 72.22% accuracy

pt = proc.time(); tpredmult = LogRegDenseMultScore(mylogregmult, multtrain); print(proc.time() - pt) #takes 0.05s
tabpred = table(data.matrix(multtrain@Target), tpredmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives 72.00% accuracy


#single table learning and scoring
singtrain = SingData(Target=as.data.frame(ytr[,1]), TargetName="stars", SingTable=ytr[,setdiff(names(ytr),c("stars","useridfk","businessidfk"))], FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(yts[,1]), TargetName="stars", SingTable=yts[,setdiff(names(ytr),c("stars","useridfk","businessidfk"))], FDs=list())

pt = proc.time(); mylogregsing = LogRegDenseSingLearn(singtrain, lambda = 1000, alpha = 0.0003, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 2.6s

pt = proc.time(); predsing = LogRegDenseSingScore(mylogregsing, singtest); print(proc.time() - pt) #takes 0.07s
tabpred = table(data.matrix(singtest@Target), predsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives 72.22% accuracy

pt = proc.time(); tpredsing = LogRegDenseSingScore(mylogregsing, singtrain); print(proc.time() - pt) #takes 0.18s
tabpred = table(data.matrix(singtrain@Target), tpredsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives % accuracy


#verify equivalence of learned models from sing and factorized
print(mylogregmult@LossVal)
print(mylogregsing@LossVal) #identical! 17077.43
print(unlist(mylogregmult@Coefs))
print(unlist(mylogregsing@Coefs)) #identical upto 10dec


#single table learning and scoring; nor1r2
singtr = SingData(Target=as.data.frame(ytr[,1]), TargetName="stars", SingTable=as.data.frame(Str[,setdiff(names(Str),c("stars","useridfk","businessidfk"))]), FDs=list()) #fds ignored for now
singts = SingData(Target=as.data.frame(yts[,1]), TargetName="stars", SingTable=as.data.frame(Sts[,setdiff(names(Sts),c("stars","useridfk","businessidfk"))]), FDs=list())

pt = proc.time(); mylogregsing2 = LogRegDenseSingLearn(singtr, lambda = 1000, alpha = 0.0003, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 2.6s

pt = proc.time(); predsing = LogRegDenseSingScore(mylogregsing2, singts); print(proc.time() - pt) #takes 0.07s
tabpred = table(data.matrix(singts@Target), predsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives 72.22% accuracy

pt = proc.time(); tpredsing = LogRegDenseSingScore(mylogregsing2, singtr); print(proc.time() - pt) #takes 0.07s
tabpred = table(data.matrix(singtr@Target), tpredsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives 72.22% accuracy

