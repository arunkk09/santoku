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

ytr = read.csv("../../../../SantokuData/Expedia/EHtraintest10retop5p.csv")
yts = read.csv("../../../../SantokuData/Expedia/EHhold10retop5p.csv")
Str = read.csv("../../../../SantokuData/Expedia/EHtraintest10retop5pMultS.csv")
Sts = read.csv("../../../../SantokuData/Expedia/EHhold10retop5pMultS.csv")
Sall = rbind(Str,Sts)
Sall$prop_idfk = factor(Sall$prop_idfk)
Sall$srch_idfk = factor(Sall$srch_idfk)
Str$prop_idfk = factor(Str$prop_idfk, levels=levels(Sall$prop_idfk))
Str$srch_idfk = factor(Str$srch_idfk, levels=levels(Sall$srch_idfk))
Sts$prop_idfk = factor(Sts$prop_idfk, levels=levels(Sall$prop_idfk))
Sts$srch_idfk = factor(Sts$srch_idfk, levels=levels(Sall$srch_idfk))

R1 = read.csv("../../../../SantokuData/Expedia/hotels_disc10retop5p.csv")
R2 = read.csv("../../../../SantokuData/Expedia/searches_disc10retop5p.csv")

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

pt = proc.time(); mynbsing = TANBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.21s
pt = proc.time(); predsing = TANBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~0.43s
pt = proc.time(); tpredsing = TANBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~0.78s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="prop_idfk",AttCol="prop_idfk",UseFK=TRUE), KFKD(EntCol="srch_idfk", AttCol="srch_idfk",UseFK=FALSE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
pt = proc.time(); mynbmult = TANBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.04s
pt = proc.time(); predmult = TANBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~0.05s
pt = proc.time(); tpredmult = TANBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~0.05s

multtrainnor1 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtestnor1 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtrainnor2 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1), KFKDs=kfkds[1])
multtestnor2 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1), KFKDs=kfkds[1])

pt = proc.time(); mynbmult = TANBayesMultLearn(multtrain);
predmult = TANBayesMultScore(mynbmult, multtest); 
tpredmult = TANBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~0.16s

pt = proc.time(); mynbmultnor1 = TANBayesMultLearn(multtrainnor1);
predmultnor1 = TANBayesMultScore(mynbmultnor1, multtestnor1); 
tpredmultnor1 = TANBayesMultScore(mynbmultnor1, multtrainnor1); print(proc.time() - pt) #takes ~0.15s

pt = proc.time(); mynbmultnor2 = TANBayesMultLearn(multtrainnor2);
predmultnor2 = TANBayesMultScore(mynbmultnor2, multtestnor2); 
tpredmultnor2 = TANBayesMultScore(mynbmultnor2, multtrainnor2); print(proc.time() - pt) #takes ~1.3s

#compare predictions from sing and factorized
tabtss = table(yts[,1], predsing)
tabtrs = table(ytr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #84.05%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #86.70%

tabtsm = table(Sts[,1], predmult)
tabtrm = table(Str[,1], tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #92.02% much higher!
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #91.54%

tabtsmnor1 = table(yts[,1], predmultnor1)
tabtrmnor1 = table(ytr[,1], tpredmultnor1)
print(tabtsmnor1)
print((sum(diag(tabtsmnor1))/sum(tabtsmnor1))) #92.12%
print(tabtrmnor1)
print((sum(diag(tabtrmnor1))/sum(tabtrmnor1))) #91.46%

tabtsmnor2 = table(yts[,1], predmultnor2)
tabtrmnor2 = table(ytr[,1], tpredmultnor2)
print(tabtsmnor2)
print((sum(diag(tabtsmnor2))/sum(tabtsmnor2))) #88.22%
print(tabtrmnor2)
print((sum(diag(tabtrmnor2))/sum(tabtrmnor2))) #97.61%
