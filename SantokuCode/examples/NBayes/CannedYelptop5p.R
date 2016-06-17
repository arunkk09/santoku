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

ytr = read.csv("../../../SantokuData/Yelp/YRtraintesttop5p.csv")
yts = read.csv("../../../SantokuData/Yelp/YRholdtop5p.csv")
Str = read.csv("../../../SantokuData/Yelp/YRtraintesttop5pMultS.csv")
Sts = read.csv("../../../SantokuData/Yelp/YRholdtop5pMultS.csv")

R1 = read.csv("../../../SantokuData/Yelp/user_disc_gendtop5p.csv")
ytr$useridfk = factor(ytr$useridfk, levels=levels(R1$useridfk))
yts$useridfk = factor(yts$useridfk, levels=levels(R1$useridfk))
R2 = read.csv("../../../SantokuData/Yelp/business_checkin_disctop5p.csv")

Ytr = read.csv("../../../SantokuData/Yelp/YRtraintesttop5p.csv")
Yts = read.csv("../../../SantokuData/Yelp/YRholdtop5p.csv")
Ytr = Ytr[,c("stars","useridfk","businessidfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender","bstars","wday5","wday3","wday4","wend5","wend3","wend4",
             "wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501","cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
			 Yts = Yts[,c("stars","useridfk","businessidfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender","bstars","wday5","wday3","wday4","wend5","wend3","wend4",
			              "wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501","cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
						  YR1 = read.csv("../../../SantokuData/Yelp/user_disc_gendtop5p.csv")
						  YR1 = YR1[,c("useridfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender")]
						  YR2 = read.csv("../../../SantokuData/Yelp/business_checkin_disctop5p.csv")
						  YR2 = YR2[,c("businessidfk","bstars","wday5","wday3","wday4","wend5","wend3","wend4","wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501",
						               "cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
									   Ytr$useridfk = factor(Ytr$useridfk, levels=levels(YR1$useridfk))
									   Yts$useridfk = factor(Yts$useridfk, levels=levels(YR1$useridfk))


#single table learning and scoring
singtrain = SingData(Target=as.data.frame(ytr[,1]), TargetName="stars", SingTable=ytr[,-1],FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(yts[,1]), TargetName="stars", SingTable=yts[,-1], FDs=list())

pt = proc.time(); mynbsing = NBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.4s
pt = proc.time(); predsing = NBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~6.0s
pt = proc.time(); tpredsing = NBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~18.9s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="useridfk",AttCol="useridfk",UseFK=TRUE), KFKD(EntCol="businessidfk", AttCol="businessidfk",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1, R2), KFKDs=kfkds)
pt = proc.time(); mynbmult = NBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.6s
pt = proc.time(); predmult = NBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~1.1s
pt = proc.time(); tpredmult = NBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~1.5s

multtrainnor1 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtestnor1 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R2), KFKDs=kfkds[2])
multtrainnor2 = MultData(Target=as.data.frame(Str[,1]), EntTable=Str[,-1], AttTables=list(R1), KFKDs=kfkds[1])
multtestnor2 = MultData(Target=as.data.frame(Sts[,1]), EntTable=Sts[,-1], AttTables=list(R1), KFKDs=kfkds[1])

remuid = setdiff(unique(Sts$useridfk), unique(Str$useridfk)) #to ensure all uids appear in tr
YSts1 = subset(Sts, is.element(Sts$useridfk, remuid))
YSts2 = subset(Sts, !is.element(Sts$useridfk, remuid))
TYStr = rbind(Str,YSts1)
TYSts = YSts2
TYSnames = names(TYStr)
TYStr = as.data.frame(cbind(sample(c('a','b'),nrow(TYStr),replace=TRUE), TYStr))
names(TYStr) = c("Temp", TYSnames)
TYSts = as.data.frame(cbind(sample(c('a','b'),nrow(TYSts),replace=TRUE), TYSts))
names(TYSts) = c("Temp", TYSnames)
TYSall = rbind(TYStr, TYSts)
TYSall$useridfk = factor(TYSall$useridfk)
TYSall$businessidfk = factor(TYSall$businessidfk)
TYStr$useridfk = factor(TYStr$useridfk, levels=levels(TYSall$useridfk))
TYStr$businessidfk = factor(TYStr$businessidfk, levels=levels(TYSall$businessidfk))
TYSts$useridfk = factor(TYSts$useridfk, levels=levels(TYSall$useridfk))
TYSts$businessidfk = factor(TYSts$businessidfk, levels=levels(TYSall$businessidfk))

multtrainnor1r2 = SingData(Target=as.data.frame(Ytr[,1]), TargetName="stars", SingTable=as.data.frame(Ytr[,c("useridfk", "businessidfk")]), FDs=list())
multtestnor1r2 = SingData(Target=as.data.frame(Yts[,1]), TargetName="stars", SingTable=as.data.frame(Yts[,c("useridfk", "businessidfk")]), FDs=list())

pt = proc.time(); mynbmult = NBayesMultLearn(multtrain);
predmult = NBayesMultScore(mynbmult, multtest); 
tpredmult = NBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~3.3s

pt = proc.time(); mynbmultnor1 = NBayesMultLearn(multtrainnor1);
predmultnor1 = NBayesMultScore(mynbmultnor1, multtestnor1); 
tpredmultnor1 = NBayesMultScore(mynbmultnor1, multtrainnor1); print(proc.time() - pt) #takes ~2.2s

pt = proc.time(); mynbmultnor2 = NBayesMultLearn(multtrainnor2);
predmultnor2 = NBayesMultScore(mynbmultnor2, multtestnor2); 
tpredmultnor2 = NBayesMultScore(mynbmultnor2, multtrainnor2); print(proc.time() - pt) #takes ~2.7s

pt = proc.time(); mynbmultnor1r2 = NBayesSingLearn(multtrainnor1r2);
predmultnor1r2 = NBayesSingScore(mynbmultnor1r2, multtestnor1r2); 
tpredmultnor1r2 = NBayesSingScore(mynbmultnor1r2, multtrainnor1r2); print(proc.time() - pt) #takes ~2.7s

#verify equivalence of learned model CPTs from sing and factorized
print(mynbsing@LogCPTs$vfunny)
print(mynbmult@LogCPTs$vfunny)
print(mynbsing@LogYPT)
print(mynbmult@LogYPT)

#verify equivalence of predictions from sing and factorized
tabtss = table(yts[,1], predsing)
tabtrs = table(ytr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #71.55%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #73.96%

tabtsm = table(yts[,1], predmult)
tabtrm = table(ytr[,1], tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #71.55%
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #73.96% identical!

tabtsmnor1 = table(yts[,1], predmultnor1)
tabtrmnor1 = table(ytr[,1], tpredmultnor1)
print(tabtsmnor1)
print((sum(diag(tabtsmnor1))/sum(tabtsmnor1))) #71.24%
print(tabtrmnor1)
print((sum(diag(tabtrmnor1))/sum(tabtrmnor1))) #73.51%

tabtsmnor2 = table(yts[,1], predmultnor2)
tabtrmnor2 = table(ytr[,1], tpredmultnor2)
print(tabtsmnor2)
print((sum(diag(tabtsmnor2))/sum(tabtsmnor2))) #72.16%
print(tabtrmnor2)
print((sum(diag(tabtrmnor2))/sum(tabtrmnor2))) #75.41%

tabtsmnor1r2 = table(yts[,1], predmultnor1r2)
tabtrmnor1r2 = table(ytr[,1], tpredmultnor1r2)
print(tabtsmnor1r2)
print((sum(diag(tabtsmnor1r2))/sum(tabtsmnor1r2))) #72.42%
print(tabtrmnor1r2)
print((sum(diag(tabtrmnor1r2))/sum(tabtrmnor1r2))) #75.91%

