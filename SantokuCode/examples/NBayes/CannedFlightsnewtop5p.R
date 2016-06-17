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

ytr = read.csv("../../../SantokuData/Flights/OFtraintestnewtop5p.csv")
yts = read.csv("../../../SantokuData/Flights/OFholdnewtop5p.csv")
Str = read.csv("../../../SantokuData/Flights/OFtraintestnewtop5pMultS.csv")
Sts = read.csv("../../../SantokuData/Flights/OFholdnewtop5pMultS.csv")

R1 = read.csv("../../../SantokuData/Flights/airlinesc_discnewtop5p.csv")
R2 = read.csv("../../../SantokuData/Flights/sairportsc_discnewtop5p.csv")
R3 = read.csv("../../../SantokuData/Flights/dairportsc_discnewtop5p.csv")

ytr$airlineid = factor(ytr$airlineid, levels=levels(R1$airlineid))
ytr$acountry = factor(ytr$acountry, levels=levels(R1$acountry))
ytr$sairportid = factor(ytr$sairportid, levels=levels(R2$sairportid))
ytr$scity = factor(ytr$scity, levels=levels(R2$scity))
ytr$scountry = factor(ytr$scountry, levels=levels(R2$scountry))
ytr$dairportid = factor(ytr$dairportid, levels=levels(R3$dairportid))
ytr$dcity = factor(ytr$dcity, levels=levels(R3$dcity))
ytr$dcountry = factor(ytr$dcountry, levels=levels(R3$dcountry))

yts$airlineid = factor(yts$airlineid, levels=levels(R1$airlineid))
yts$acountry = factor(yts$acountry, levels=levels(R1$acountry))
yts$sairportid = factor(yts$sairportid, levels=levels(R2$sairportid))
yts$scity = factor(yts$scity, levels=levels(R2$scity))
yts$scountry = factor(yts$scountry, levels=levels(R2$scountry))
yts$dairportid = factor(yts$dairportid, levels=levels(R3$dairportid))
yts$dcity = factor(yts$dcity, levels=levels(R3$dcity))
yts$dcountry = factor(yts$dcountry, levels=levels(R3$dcountry))

#single table learning and scoring
singtrain = SingData(Target=as.data.frame(ytr[,1]), TargetName="codeshare", SingTable=ytr[,-1],FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(yts[,1]), TargetName="codeshare", SingTable=yts[,-1], FDs=list())

pt = proc.time(); mynbsing = NBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.1s
pt = proc.time(); predsing = NBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~2.8s
pt = proc.time(); tpredsing = NBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~8.3s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="airlineid",AttCol="airlineid",UseFK=TRUE), KFKD(EntCol="sairportid", AttCol="sairportid",UseFK=TRUE), KFKD(EntCol="dairportid", AttCol="dairportid",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2, R3), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2, R3), KFKDs=kfkds)
pt = proc.time(); mynbmult = NBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.25s
pt = proc.time(); predmult = NBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~1.8s
pt = proc.time(); tpredmult = NBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~4.9s

FStrnor1 = Str
FStsnor1 = Sts
FStrnor1$airlineid = factor(FStrnor1$airlineid)
FStsnor1$airlineid = factor(FStsnor1$airlineid)
trnor1 = MultData(Target=as.data.frame(FStrnor1[,1]), EntTable=FStrnor1[,-1], AttTables=list(R2, R3), KFKDs=kfkds[-1])
tsnor1 = MultData(Target=as.data.frame(FStsnor1[,1]), EntTable=FStsnor1[,-1], AttTables=list(R2, R3), KFKDs=kfkds[-1])
pt = proc.time(); mynbmult = NBayesMultLearn(trnor1);
predmult = NBayesMultScore(mynbmult, tsnor1);
tpredmult = NBayesMultScore(mynbmult, trnor1); print(proc.time() - pt)

FStrnor1r2 = Str
FStsnor1r2 = Sts
FStrnor1r2$airlineid = factor(FStrnor1r2$airlineid)
FStsnor1r2$airlineid = factor(FStsnor1r2$airlineid)
FStrnor1r2$sairportid = factor(FStrnor1r2$sairportid)
FStsnor1r2$sairportid = factor(FStsnor1r2$sairportid)
trnor1r2 = MultData(Target=as.data.frame(FStrnor1r2[,1]), EntTable=FStrnor1r2[,-1], AttTables=list(R3), KFKDs=kfkds[3])
tsnor1r2 = MultData(Target=as.data.frame(FStsnor1r2[,1]), EntTable=FStsnor1r2[,-1], AttTables=list(R3), KFKDs=kfkds[3])
pt = proc.time(); mynbmult = NBayesMultLearn(trnor1r2);
predmult = NBayesMultScore(mynbmult, tsnor1r2);
tpredmult = NBayesMultScore(mynbmult, trnor1r2); print(proc.time() - pt)

#verify equivalence of learned model CPTs from sing and factorized
print(mynbsing@LogCPTs$stimezone)
print(mynbmult@LogCPTs$stimezone)
print(mynbsing@LogYPT)
print(mynbmult@LogYPT)

#verify equivalence of predictions from sing and factorized
tabtss = table(yts[,1], predsing)
tabtrs = table(ytr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #70.28%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #70.20%

tabtsm = table(Sts[,1], predmult)
tabtrm = table(Str[,1], tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #70.28%
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #70.20% identical!


