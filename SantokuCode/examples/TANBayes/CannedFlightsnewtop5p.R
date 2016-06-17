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

ytr = read.csv("../../../SantokuData/Flights/OFtraintestnewtop5p.csv")
yts = read.csv("../../../SantokuData/Flights/OFholdnewtop5p.csv")
Str = read.csv("../../../SantokuData/Flights/OFtraintestnewtop5pMultS.csv")
Sts = read.csv("../../../SantokuData/Flights/OFholdnewtop5pMultS.csv")
Sall = rbind(Str,Sts)
Sall$airlineid = factor(Sall$airlineid)
Sall$sairportid = factor(Sall$sairportid)
Sall$dairportid = factor(Sall$dairportid)
Str$airlineid = factor(Str$airlineid, levels=levels(Sall$airlineid))
Str$sairportid = factor(Str$sairportid, levels=levels(Sall$sairportid))
Str$dairportid = factor(Str$dairportid, levels=levels(Sall$dairportid))
Sts$airlineid = factor(Sts$airlineid, levels=levels(Sall$airlineid))
Sts$sairportid = factor(Sts$sairportid, levels=levels(Sall$sairportid))
Sts$dairportid = factor(Sts$dairportid, levels=levels(Sall$dairportid))

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

pt = proc.time(); mynbsing = TANBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.17s
pt = proc.time(); predsing = TANBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~0.34s
pt = proc.time(); tpredsing = TANBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~0.56s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="airlineid",AttCol="airlineid",UseFK=TRUE), KFKD(EntCol="sairportid", AttCol="sairportid",UseFK=TRUE), KFKD(EntCol="dairportid", AttCol="dairportid",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2, R3), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2, R3), KFKDs=kfkds)
pt = proc.time(); mynbmult = TANBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.06s
pt = proc.time(); predmult = TANBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~0.19s
pt = proc.time(); tpredmult = TANBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~0.3s

#compare predictions from sing and factorized
tabtss = table(yts[,1], predsing)
tabtrs = table(ytr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #83.47%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #85.82%

tabtsm = table(Sts[,1], predmult)
tabtrm = table(Str[,1], tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #84.09%
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #86.16%
