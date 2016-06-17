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

ftr = read.csv("../../../SantokuData/Flights/DOFtraintestnewSingtop5p.csv")
fts = read.csv("../../../SantokuData/Flights/DOFholdnewSingtop5p.csv")
Str = read.csv("../../../SantokuData/Flights/DOFtraintestnewMultStop5p.csv") #entity table alone
Sts = read.csv("../../../SantokuData/Flights/DOFholdnewMultStop5p.csv")
R1 = read.csv("../../../SantokuData/Flights/DOFMultR1top5p.csv")
R2 = read.csv("../../../SantokuData/Flights/DOFMultR2top5p.csv")
R3 = read.csv("../../../SantokuData/Flights/DOFMultR3top5p.csv")


#factorized learning and scoring
kfkds = list(KFKD(EntCol="airlineid", AttCol="airlineid",UseFK=TRUE), KFKD(EntCol="sairportid",AttCol="sairportid",UseFK=TRUE), KFKD(EntCol="dairportid",AttCol="dairportid",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2, R3), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2, R3), KFKDs=kfkds)

pt = proc.time(); mylogregmult = LogRegDenseMultLearn(multtrain, lambda = 100, alpha = 0.001, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 2.3s

pt = proc.time(); predmult = LogRegDenseMultScore(mylogregmult, multtest); print(proc.time() - pt) #takes 0.05s
tabpred = table(data.matrix(multtest@Target), predmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives 78.60% accuracy

pt = proc.time(); tpredmult = LogRegDenseMultScore(mylogregmult, multtrain); print(proc.time() - pt) #takes 0.07s
tabpred = table(data.matrix(multtrain@Target), tpredmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives 78.99% accuracy


#single table learning and scoring
singtrain = SingData(Target=as.data.frame(ftr[,1]), TargetName="codeshare", SingTable=ftr[,setdiff(names(ftr),c("codeshare","airlineid","sairportid","dairportid"))], FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(fts[,1]), TargetName="codeshare", SingTable=fts[,setdiff(names(ftr),c("codeshare","airlineid","sairportid","dairportid"))], FDs=list())

pt = proc.time(); mylogregsing = LogRegDenseSingLearn(singtrain, lambda = 100, alpha = 0.001, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 7.4s

pt = proc.time(); predsing = LogRegDenseSingScore(mylogregsing, singtest); print(proc.time() - pt) #takes 0.15s
tabpred = table(data.matrix(singtest@Target), predsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives 78.60% accuracy

pt = proc.time(); tpredsing = LogRegDenseSingScore(mylogregsing, singtrain); print(proc.time() - pt) #takes 0.40s
tabpred = table(data.matrix(singtrain@Target), tpredsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives 78.99% accuracy


#verify equivalence of learned models from sing and factorized
print(mylogregmult@LossVal)
print(mylogregsing@LossVal) #identical! 8113.9
print(unlist(mylogregmult@Coefs))
print(unlist(mylogregsing@Coefs)) #order mixed up?
