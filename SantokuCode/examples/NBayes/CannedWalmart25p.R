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
source("../../src/NBayes/NBayes.R")
options(width=190)

wtr = read.csv("../../../SantokuData/Walmart/WTtraintest25p.csv")
wts = read.csv("../../../SantokuData/Walmart/WThold25p.csv")

Str = read.csv("../../../SantokuData/Walmart/WTtraintest25pMultS.csv")
Sts = read.csv("../../../SantokuData/Walmart/WThold25pMultS.csv")
R1 = read.csv("../../../SantokuData/Walmart/stores_disc.csv")
R2 = read.csv("../../../SantokuData/Walmart/features_disc25p.csv")

#single table learning and scoring
singtrain = SingData(Target=as.data.frame(wtr[,1]), TargetName="weekly_sales", SingTable=wtr[,-1],FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(wts[,1]), TargetName="weekly_sales", SingTable=wts[,-1], FDs=list())

pt = proc.time(); mynbsing = NBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.32s

pt = proc.time(); predsing = NBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~5.3s
pt = proc.time(); tpredsing = NBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~5.3s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="storefk",AttCol="storefk",UseFK=TRUE), KFKD(EntCol="purchaseidfk", AttCol="purchaseidfk",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
pt = proc.time(); mynbmult = NBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.16s
pt = proc.time(); predmult = NBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~1.1s
pt = proc.time(); tpredmult = NBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~3.3s

multtrainnor1 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtestnor1 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtrainnor2 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1), KFKDs=kfkds[1])
multtestnor2 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1), KFKDs=kfkds[1])

pt = proc.time(); mynbmult = NBayesMultLearn(multtrain);
predmult = NBayesMultScore(mynbmult, multtest); 
tpredmult = NBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~4.5s

pt = proc.time(); mynbmultnor1 = NBayesMultLearn(multtrainnor1);
predmultnor1 = NBayesMultScore(mynbmultnor1, multtestnor1); 
tpredmultnor1 = NBayesMultScore(mynbmultnor1, multtrainnor1); print(proc.time() - pt) #takes ~5.8s

pt = proc.time(); mynbmultnor2 = NBayesMultLearn(multtrainnor2);
predmultnor2 = NBayesMultScore(mynbmultnor2, multtestnor2); 
tpredmultnor2 = NBayesMultScore(mynbmultnor2, multtrainnor2); print(proc.time() - pt) #takes ~5.3s

#verify equivalence of learned model CPTs from sing and factorized
print(mynbsing@LogCPTs$size)
print(mynbmult@LogCPTs$size)
print(mynbsing@LogYPT)
print(mynbmult@LogYPT)

#verify equivalence of predictions from sing and factorized
tabtss = table(wts[,1], predsing)
tabtrs = table(wtr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #86.93%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #86.72%

tabtsm = table(wts[,1], predmult)
tabtrm = table(wtr[,1], tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #86.93% identical
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #86.72% identical

tabtsmnor1 = table(wts[,1], predmultnor1)
tabtrmnor1 = table(wtr[,1], tpredmultnor1)
print(tabtsmnor1)
print((sum(diag(tabtsmnor1))/sum(tabtsmnor1))) #88.77%
print(tabtrmnor1)
print((sum(diag(tabtrmnor1))/sum(tabtrmnor1))) #88.72%

tabtsmnor2 = table(wts[,1], predmultnor2)
tabtrmnor2 = table(wtr[,1], tpredmultnor2)
print(tabtsmnor2)
print((sum(diag(tabtsmnor2))/sum(tabtsmnor2))) #87.44%
print(tabtrmnor2)
print((sum(diag(tabtrmnor2))/sum(tabtrmnor2))) #87.39%
