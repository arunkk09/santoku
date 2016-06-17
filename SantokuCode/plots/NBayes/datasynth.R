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
library(data.table)
library(Matrix)
options(width=190)
options(scipen=999)

mns = 1e4 #5e5
mnr = 1e2 #1e4
mds = 4
mdr = 100 #400

#varyns = c(1e4, 5e4, 1e5, 5e5, 1e6, 5e6)
#varydr = c(2, 4, 20, 40, 200, 400)
varyns = c(1e2, 5e2, 1e3, 5e3, 1e4)
varydr = c(2, 4, 20, 40, 100)

np = 5
#params is a 2xnpx4 matrix of param values with cols being ns,nr,ds,dr; np combos each for varying ns and dr
params = rbind(cbind(varyns, rep(mnr, np), rep(mds, np), rep(mdr, np)), cbind(rep(mns, np), rep(mnr, np), rep(mds, np), varydr))

for (i in 1:nrow(params)) {

ns = params[i, 1]; nr = params[i, 2]; ds = params[i, 3]; dr = params[i, 4]; 
cat("GEN_DATA:", ns, nr, ds, dr, "\n")

#generate synth data R, and S incl Y and FK
tinyr = as.data.frame(matrix(sample(c("t","f"), (nr * dr), replace=TRUE), nrow=nr, ncol=dr))
tinys = as.data.frame(matrix(sample(c("t","f"), (ns * ds), replace=TRUE), nrow=ns, ncol=ds))
yvec = sample(c("t","f"), ns, replace=TRUE)
fkvec = sample(seq(1, nr), ns, replace=TRUE)
tinys = cbind(yvec, tinys, fkvec)
names(tinys) = c("y", "xs1", "xs2", "xs3", "xs4", "fk")
#names(tinys) = c("y", "xs1", "fk")

tinyr = cbind(seq(1, nr), tinyr)
names(tinyr)[1] = "fk"
tinyr$fk = factor(tinyr$fk)
tinys$fk = factor(tinys$fk, levels=levels(tinyr$fk))
pt = proc.time(); tinyt = merge(tinys, tinyr); cat("TIME_J:", proc.time() - pt, "\n")
tinyt$fk = factor(tinyt$fk, levels=levels(tinyr$fk))

singtrain = SingData(Target=as.data.frame(tinyt[,2]), TargetName="y", SingTable=tinyt[,-2], FDs=list())
kfkds = list(KFKD(EntCol="fk", AttCol="fk", UseFK=TRUE))
multtrain = MultData(Target=as.data.frame(tinys[,1]), EntTable=tinys[,-1], AttTables=list(tinyr), KFKDs=kfkds)

#time slss and flfs thrice
for (tt in 1:3) {
	pt = proc.time(); mynbsing = NBayesSingLearn(singtrain); cat("TIME_SL:", proc.time() - pt, "\n")
	pt = proc.time(); tpredsing = NBayesSingScore(mynbsing, singtrain); cat("TIME_SS:", proc.time() - pt, "\n")

	pt = proc.time(); mynbmult = NBayesMultLearn(multtrain); cat("TIME_FL:", proc.time() - pt, "\n")
	pt = proc.time(); tpredmult = NBayesMultScore(mynbmult, multtrain); cat("TIME_FS:", proc.time() - pt, "\n")
}

}
