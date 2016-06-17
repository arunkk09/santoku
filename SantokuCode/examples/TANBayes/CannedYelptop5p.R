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

ytr = read.csv("../../../SantokuData/Yelp/YRtraintesttop5p.csv")
yts = read.csv("../../../SantokuData/Yelp/YRholdtop5p.csv")
ytr = ytr[,c("stars","useridfk","businessidfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender","bstars","wday5","wday3","wday4","wend5","wend3","wend4","wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501","cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
yts = yts[,c("stars","useridfk","businessidfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender","bstars","wday5","wday3","wday4","wend5","wend3","wend4","wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501","cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
Str = read.csv("../../../SantokuData/Yelp/YRtraintesttop5pMultS.csv")
Sts = read.csv("../../../SantokuData/Yelp/YRholdtop5pMultS.csv")
remuid = setdiff(unique(Sts$useridfk), unique(Str$useridfk)) #to ensure all uids appear in tr
Sts1 = subset(Sts, is.element(Sts$useridfk, remuid))
Sts2 = subset(Sts, !is.element(Sts$useridfk, remuid))
Str = rbind(Str,Sts1)
Sts = Sts2
Snames = names(Str)
Str = as.data.frame(cbind(sample(c('a','b'),nrow(Str),replace=TRUE), Str))
names(Str) = c("Temp", Snames)
Sts = as.data.frame(cbind(sample(c('a','b'),nrow(Sts),replace=TRUE), Sts))
names(Sts) = c("Temp", Snames)
Sall = rbind(Str, Sts)
Sall$useridfk = factor(Sall$useridfk)
Sall$businessidfk = factor(Sall$businessidfk)
Str$useridfk = factor(Str$useridfk, levels=levels(Sall$useridfk))
Str$businessidfk = factor(Str$businessidfk, levels=levels(Sall$businessidfk))
Sts$useridfk = factor(Sts$useridfk, levels=levels(Sall$useridfk))
Sts$businessidfk = factor(Sts$businessidfk, levels=levels(Sall$businessidfk))

R1 = read.csv("../../../SantokuData/Yelp/user_disc_gendtop5p.csv")
ytr$useridfk = factor(ytr$useridfk, levels=levels(R1$useridfk))
yts$useridfk = factor(yts$useridfk, levels=levels(R1$useridfk))
R2 = read.csv("../../../SantokuData/Yelp/business_checkin_disctop5p.csv")
R2 = R2[, c("businessidfk","bstars","wday5","wday3","wday4","wend5","wend3","wend4","wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501","cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]

#single table learning and scoring
singtrain = SingData(Target=as.data.frame(ytr[,1]), TargetName="stars", SingTable=ytr[,-1],FDs=list()) #fds ignored for now
singtest = SingData(Target=as.data.frame(yts[,1]), TargetName="stars", SingTable=yts[,-1], FDs=list())

pt = proc.time(); mynbsing = TANBayesSingLearn(singtrain); print(proc.time() - pt) #takes ~0.5s
pt = proc.time(); predsing = TANBayesSingScore(mynbsing, singtest); print(proc.time() - pt) #takes ~1.2s
pt = proc.time(); tpredsing = TANBayesSingScore(mynbsing, singtrain); print(proc.time() - pt) #takes ~1.7s

#factorized learning and scoring
kfkds = list(KFKD(EntCol="useridfk",AttCol="useridfk",UseFK=TRUE), KFKD(EntCol="businessidfk", AttCol="businessidfk",UseFK=TRUE)) #att col and ent col names are identical for now
multtrain = MultData(Target=as.data.frame(Str[,2]), EntTable=Str[,-2], AttTables=list(R1, R2), KFKDs=kfkds)
multtest = MultData(Target=as.data.frame(Sts[,2]), EntTable=Sts[,-2], AttTables=list(R1, R2), KFKDs=kfkds)
pt = proc.time(); mynbmult = TANBayesMultLearn(multtrain); print(proc.time() - pt) #takes ~0.14s
pt = proc.time(); predmult = TANBayesMultScore(mynbmult, multtest); print(proc.time() - pt) #takes ~0.49s
pt = proc.time(); tpredmult = TANBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~0.67s

multtrainnor1 = MultData(Target=as.data.frame(Str[,2]), EntTable=Str[,-2], AttTables=list(R2), KFKDs=kfkds[2])
multtestnor1 = MultData(Target=as.data.frame(Sts[,2]), EntTable=Sts[,-2], AttTables=list(R2), KFKDs=kfkds[2])
multtrainnor2 = MultData(Target=as.data.frame(Str[,2]), EntTable=Str[,-2], AttTables=list(R1), KFKDs=kfkds[1])
multtestnor2 = MultData(Target=as.data.frame(Sts[,2]), EntTable=Sts[,-2], AttTables=list(R1), KFKDs=kfkds[1])

pt = proc.time(); mynbmult = TANBayesMultLearn(multtrain);
predmult = TANBayesMultScore(mynbmult, multtest); 
tpredmult = TANBayesMultScore(mynbmult, multtrain); print(proc.time() - pt) #takes ~1.16s

pt = proc.time(); mynbmultnor1 = TANBayesMultLearn(multtrainnor1);
predmultnor1 = TANBayesMultScore(mynbmultnor1, multtestnor1); 
tpredmultnor1 = TANBayesMultScore(mynbmultnor1, multtrainnor1); print(proc.time() - pt) #takes ~1.31s

pt = proc.time(); mynbmultnor2 = TANBayesMultLearn(multtrainnor2);
predmultnor2 = TANBayesMultScore(mynbmultnor2, multtestnor2); 
tpredmultnor2 = TANBayesMultScore(mynbmultnor2, multtrainnor2); print(proc.time() - pt) #takes ~1.42s

#compare predictions from sing and factorized
tabtss = table(yts[,1], predsing)
tabtrs = table(ytr[,1], tpredsing)
print(tabtss)
print((sum(diag(tabtss))/sum(tabtss))) #88.51%
print(tabtrs)
print((sum(diag(tabtrs))/sum(tabtrs))) #97.34%

tabtsm = table(Sts$stars, predmult)
tabtrm = table(Str$stars, tpredmult)
print(tabtsm)
print((sum(diag(tabtsm))/sum(tabtsm))) #89.50%
print(tabtrm)
print((sum(diag(tabtrm))/sum(tabtrm))) #97.38%

tabtsmnor1 = table(Sts$stars, predmultnor1)
tabtrmnor1 = table(Str$stars, tpredmultnor1)
print(tabtsmnor1)
print((sum(diag(tabtsmnor1))/sum(tabtsmnor1))) #89.78%
print(tabtrmnor1)
print((sum(diag(tabtrmnor1))/sum(tabtrmnor1))) #97.33%

tabtsmnor2 = table(Sts$stars, predmultnor2)
tabtrmnor2 = table(Str$stars, tpredmultnor2)
print(tabtsmnor2)
print((sum(diag(tabtsmnor2))/sum(tabtsmnor2))) #89.19%
print(tabtrmnor2)
print((sum(diag(tabtrmnor2))/sum(tabtrmnor2))) #97.42%
