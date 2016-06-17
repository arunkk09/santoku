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

mns = 5e5 #5e4
mnr = 1e4 #5e2
mds = 4
mdr = 40 #400

varyns = c(1e4, 5e4, 1e5, 5e5, 1e6, 5e6)
varydr = c(2, 4, 20, 40, 200, 400)

#params is a 2x6x4 matrix of param values with cols being ns,nr,ds,dr; 6 combos each for varying ns and dr
params = rbind(cbind(varyns, rep(mnr, 6), rep(mds, 6), rep(mdr, 6)), cbind(rep(mns, 6), rep(mnr, 6), rep(mds, 6), varydr))

for (i in 1:nrow(params)) {

ns = params[i, 1]; nr = params[i, 2]; ds = params[i, 3]; dr = params[i, 4]; 
cat("GEN_DATA:", ns, nr, ds, dr, "\n")

#generate synth data R, and S incl Y and FK
tinyr = as.data.frame(matrix(rep(1.1, nr * dr), nrow=nr, ncol=dr))
tinys = as.data.frame(matrix(rep(1.1, ns * ds), nrow=ns, ncol=ds))
yvec = sample(c(1,-1), ns, replace=TRUE)
fkvec = sample(seq(1, nr), ns, replace=TRUE)
tinys = cbind(yvec, tinys, fkvec)
names(tinys) = c("y", "xs1", "xs2", "xs3", "xs4", "fk")
#names(tinys) = c("y", "xs1", "fk")

tinyrp = cbind(seq(1, nr), tinyr)
names(tinyrp)[1] = "fk"
pt = proc.time(); tinyt = merge(tinys, tinyrp); tinyt = tinyt[,-1]; cat("TIME_J:", proc.time() - pt, "\n")

singtrain = SingData(Target=as.data.frame(tinyt[,1]), TargetName="y", SingTable=tinyt[,-1], FDs=list())
kfkds = list(KFKD(EntCol="fk", AttCol="fk"))
multtrain = MultData(Target=as.data.frame(tinys[,1]), EntTable=tinys[,-1], AttTables=list(tinyr), KFKDs=kfkds)

#time slss and flfs thrice
for (tt in 1:3) {
	pt = proc.time(); mylogregsing = LogRegDenseSingLearn(singtrain, lambda = 1000, alpha = 0.00001, eps = 0, maxiters = 20); cat("TIME_SL:", proc.time() - pt, "\n")
	pt = proc.time(); tpredsing = LogRegDenseSingScore(mylogregsing, singtrain); cat("TIME_SS:", proc.time() - pt, "\n")

	pt = proc.time(); mylogregmult = LogRegDenseMultLearn(multtrain, lambda = 1000, alpha = 0.00001, eps = 0, maxiters = 20); cat("TIME_FL:", proc.time() - pt, "\n")
	pt = proc.time(); tpredmult = LogRegDenseMultScore(mylogregmult, multtrain); cat("TIME_FS:", proc.time() - pt, "\n")
}

}
