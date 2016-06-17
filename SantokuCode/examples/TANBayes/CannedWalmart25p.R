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
source("../../src/TANBayes/TANBayes.R")
options(width=190)

wtr = read.csv("../../../SantokuData/Walmart/WTtraintest25p.csv")
wts = read.csv("../../../SantokuData/Walmart/WThold25p.csv")

Str = read.csv("../../../SantokuData/Walmart/WTtraintest25pMultS.csv")
Sts = read.csv("../../../SantokuData/Walmart/WThold25pMultS.csv")
#for TAN, the fks have to be made factors
Sall = rbind(Str,Sts)
Sall$storefk = factor(Sall$storefk)
Sall$purchaseidfk = factor(Sall$purchaseidfk)
Str$storefk = factor(Str$storefk, levels=levels(Sall$storefk))
Sts$storefk = factor(Sts$storefk, levels=levels(Sall$storefk))
Str$purchaseidfk = factor(Str$purchaseidfk, levels=levels(Sall$purchaseidfk))
Sts$purchaseidfk = factor(Sts$purchaseidfk, levels=levels(Sall$purchaseidfk))

R1 = read.csv("../../../SantokuData/Walmart/stores_disc.csv")
R2 = read.csv("../../../SantokuData/Walmart/features_disc25p.csv")

#single table learning and scoring
singtrain = SingData(Target=as.data.frame(wtr[,1]), TargetName="weekly_sales", SingTable=wtr[,-1],FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(wts[,1]), TargetName="weekly_sales", SingTable=wts[,-1], FDs=list())

pt = proc.time(); mynbsing = TANBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.19s
pt = proc.time(); predsing = TANBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~0.34s
pt = proc.time(); tpredsing = TANBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~0.75s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="storefk",AttCol="storefk",UseFK=TRUE), KFKD(EntCol="purchaseidfk", AttCol="purchaseidfk",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
pt = proc.time(); mynbmult = TANBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.05s
pt = proc.time(); predmult = TANBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~0.0s
pt = proc.time(); tpredmult = TANBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~0.197s

multtrainnor1 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtestnor1 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtrainnor2 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1), KFKDs=kfkds[1])
multtestnor2 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1), KFKDs=kfkds[1])

pt = proc.time(); mynbmult = TANBayesMultLearn(multtrain);
predmult = TANBayesMultScore(mynbmult, multtest); 
tpredmult = TANBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~0.35s

pt = proc.time(); mynbmultnor1 = TANBayesMultLearn(multtrainnor1);
predmultnor1 = TANBayesMultScore(mynbmultnor1, multtestnor1); 
tpredmultnor1 = TANBayesMultScore(mynbmultnor1, multtrainnor1); print(proc.time() - pt) #takes ~0.34s

pt = proc.time(); mynbmultnor2 = TANBayesMultLearn(multtrainnor2);
predmultnor2 = TANBayesMultScore(mynbmultnor2, multtestnor2); 
tpredmultnor2 = TANBayesMultScore(mynbmultnor2, multtrainnor2); print(proc.time() - pt) #takes ~0.35s

#compare predictions from sing and factorized
tabtss = table(wts[,1], predsing)
tabtrs = table(wtr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #91.97%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #91.47%

tabtsm = table(wts[,1], predmult)
tabtrm = table(wtr[,1], tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #92.02%
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #91.55%

tabtsmnor1 = table(wts[,1], predmultnor1)
tabtrmnor1 = table(wtr[,1], tpredmultnor1)
print(tabtsmnor1)
print((sum(diag(tabtsmnor1))/sum(tabtsmnor1))) #92.11%
print(tabtrmnor1)
print((sum(diag(tabtrmnor1))/sum(tabtrmnor1))) #91.46%

tabtsmnor2 = table(wts[,1], predmultnor2)
tabtrmnor2 = table(wtr[,1], tpredmultnor2)
print(tabtsmnor2)
print((sum(diag(tabtsmnor2))/sum(tabtsmnor2))) #92.09%
print(tabtrmnor2)
print((sum(diag(tabtrmnor2))/sum(tabtrmnor2))) #91.59%
