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

etr = read.csv("../../../SantokuData/Expedia/DEHtraintest10reSingtop5p.csv")
ets = read.csv("../../../SantokuData/Expedia/DEHhold10reSingtop5p.csv")
Str = read.csv("../../../SantokuData/Expedia/DEHtraintest10reMultStop5p.csv") #entity table alone
Sts = read.csv("../../../SantokuData/Expedia/DEHhold10reMultStop5p.csv")
R1 = read.csv("../../../SantokuData/Expedia/DEHMultR1top5p.csv")
R2 = read.csv("../../../SantokuData/Expedia/DEHMultR2top5p.csv")


#factorized learning and scoring
kfkds = list(KFKD(EntCol="prop_idfk", AttCol="prop_idfk",UseFK=TRUE), KFKD(EntCol="srch_idfk",AttCol="srch_idfk",UseFK=FALSE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)

pt = proc.time(); mylogregmult = LogRegDenseMultLearn(multtrain, lambda = 1000, alpha = 0.0002, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 2.7s

pt = proc.time(); predmult = LogRegDenseMultScore(mylogregmult, multtest); print(proc.time() - pt) #takes 0.04s
tabpred = table(data.matrix(multtest@Target), predmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives 73.94% accuracy

pt = proc.time(); tpredmult = LogRegDenseMultScore(mylogregmult, multtrain); print(proc.time() - pt) #takes 0.07s
tabpred = table(data.matrix(multtrain@Target), tpredmult)
print(sum(diag(tabpred))/sum(tabpred)); #gives 73.05% accuracy


#single table learning and scoring
singtrain = SingData(Target=as.data.frame(etr[,1]), TargetName="position", SingTable=etr[,setdiff(names(etr),c("position","prop_idfk","srch_idfk"))], FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(ets[,1]), TargetName="position", SingTable=ets[,setdiff(names(etr),c("position","prop_idfk","srch_idfk"))], FDs=list())

pt = proc.time(); mylogregsing = LogRegDenseSingLearn(singtrain, lambda = 1000, alpha = 0.0002, eps = 0.001, maxiters = 20); print(proc.time() - pt) #takes 3.8s

pt = proc.time(); predsing = LogRegDenseSingScore(mylogregsing, singtest); print(proc.time() - pt) #takes 0.8s
tabpred = table(data.matrix(singtest@Target), predsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives 73.83% accuracy

pt = proc.time(); tpredsing = LogRegDenseSingScore(mylogregsing, singtrain); print(proc.time() - pt) #takes 0.24s
tabpred = table(data.matrix(singtrain@Target), tpredsing)
print(sum(diag(tabpred))/sum(tabpred)); #gives 73.10% accuracy


#verify equivalence of learned models from sing and factorized
print(mylogregmult@LossVal)
print(mylogregsing@LossVal) #different!
print(unlist(mylogregmult@Coefs))
print(unlist(mylogregsing@Coefs)) #very different in some coefs!
