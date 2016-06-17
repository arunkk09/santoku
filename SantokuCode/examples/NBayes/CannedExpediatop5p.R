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

ytr = read.csv("../../../SantokuData/Expedia/EHtraintest10retop5p.csv")
yts = read.csv("../../../SantokuData/Expedia/EHhold10retop5p.csv")
Str = read.csv("../../../SantokuData/Expedia/EHtraintest10retop5pMultS.csv")
Sts = read.csv("../../../SantokuData/Expedia/EHhold10retop5pMultS.csv")

R1 = read.csv("../../../SantokuData/Expedia/hotels_disc10retop5p.csv")
R2 = read.csv("../../../SantokuData/Expedia/searches_disc10retop5p.csv")

ytr$prop_idfk = factor(ytr$prop_idfk, levels=levels(R1$prop_idfk))
ytr$srch_idfk = factor(ytr$srch_idfk, levels=levels(R2$srch_idfk))
ytr$srch_destination_id = factor(ytr$srch_destination_id, levels=levels(R2$srch_destination_id))
ytr$visitor_location_country_id = factor(ytr$visitor_location_country_id, levels=levels(R2$visitor_location_country_id))

yts$prop_idfk = factor(yts$prop_idfk, levels=levels(R1$prop_idfk))
yts$srch_idfk = factor(yts$srch_idfk, levels=levels(R2$srch_idfk))
yts$srch_destination_id = factor(yts$srch_destination_id, levels=levels(R2$srch_destination_id))
yts$visitor_location_country_id = factor(yts$visitor_location_country_id, levels=levels(R2$visitor_location_country_id))

#single table learning and scoring
ytr = ytr[,-3] #srch_idfk is not closed domain
yts = yts[,-3]
singtrain = SingData(Target=as.data.frame(ytr[,1]), TargetName="position", SingTable=ytr[,-1],FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(yts[,1]), TargetName="position", SingTable=yts[,-1], FDs=list())

pt = proc.time(); mynbsing = NBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.3s
pt = proc.time(); predsing = NBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~4.2s
pt = proc.time(); tpredsing = NBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~12.8s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="prop_idfk",AttCol="prop_idfk",UseFK=TRUE), KFKD(EntCol="srch_idfk", AttCol="srch_idfk",UseFK=FALSE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
pt = proc.time(); mynbmult = NBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.6s
pt = proc.time(); predmult = NBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~1.8s
pt = proc.time(); tpredmult = NBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~4.3s

multtrainnor1 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtestnor1 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtrainnor2 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1), KFKDs=kfkds[1])
multtestnor2 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1), KFKDs=kfkds[1])

pt = proc.time(); mynbmult = NBayesMultLearn(multtrain);
predmult = NBayesMultScore(mynbmult, multtest); 
tpredmult = NBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~6.5s

pt = proc.time(); mynbmultnor1 = NBayesMultLearn(multtrainnor1);
predmultnor1 = NBayesMultScore(mynbmultnor1, multtestnor1); 
tpredmultnor1 = NBayesMultScore(mynbmultnor1, multtrainnor1); print(proc.time() - pt) #takes ~6.7s

pt = proc.time(); mynbmultnor2 = NBayesMultLearn(multtrainnor2);
predmultnor2 = NBayesMultScore(mynbmultnor2, multtestnor2); 
tpredmultnor2 = NBayesMultScore(mynbmultnor2, multtrainnor2); print(proc.time() - pt) #takes ~5.6s

#verify equivalence of learned model CPTs from sing and factorized
print(mynbsing@LogCPTs$time)
print(mynbmult@LogCPTs$time)
print(mynbsing@LogYPT)
print(mynbmult@LogYPT)

#verify equivalence of predictions from sing and factorized
tabtss = table(yts[,1], predsing)
tabtrs = table(ytr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #74.84%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #75.34%

tabtsm = table(yts[,1], predmult)
tabtrm = table(ytr[,1], tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #74.84%
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #75.34% identical!

tabtsmnor1 = table(yts[,1], predmultnor1)
tabtrmnor1 = table(ytr[,1], tpredmultnor1)
print(tabtsmnor1)
print((sum(diag(tabtsmnor1))/sum(tabtsmnor1))) #78.65%
print(tabtrmnor1)
print((sum(diag(tabtrmnor1))/sum(tabtrmnor1))) #79.02%

tabtsmnor2 = table(yts[,1], predmultnor2)
tabtrmnor2 = table(ytr[,1], tpredmultnor2)
print(tabtsmnor2)
print((sum(diag(tabtsmnor2))/sum(tabtsmnor2))) #74.56%
print(tabtrmnor2)
print((sum(diag(tabtrmnor2))/sum(tabtrmnor2))) #75.98%
