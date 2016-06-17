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

library(data.table)
library(Matrix)

#coef splits are assumed to be in order: enttable, att tables; att table list order is assumed to be preserved across learning and scoring
LogRegDense <- setClass (
	"LogRegDense",
	slots = c (
		Coefs = "list",
		NumSplits = "numeric",
		LossVal = "numeric",
		Call = "call"
		)
	)

#asingdata is an instance of SingData; output is an instance of LogRegDense
#lambda is l2 regularizer; alpha is bgd fixed stepsize; eps is fractional drop in loss covergence tolerance; maxiters is max num of bgd iters
LogRegDenseSingLearn <- function (asingdata, lambda = 0.0, alpha = 0.1, eps = 0.01, maxiters = 100) {
	call <- match.call()
	yvec = data.matrix(asingdata@Target);
	xmat = data.matrix(asingdata@SingTable); #FKs are assumed to be dropped a priori
	#xmat = cbind(rep(1, nrow(x)), x); #for intercept 
	wvec = rep(0.0, ncol(xmat));
	#prevloss = 0.5 * lambda * sum(wvec * wvec) + sum(log(1 + exp(-yvec * (xmat %*% wvec))));
	prevloss = nrow(xmat) * log(2.0);
	it = 1;
	while(it <= maxiters) {
		tmpmat = yvec * (xmat %*% wvec);
		partsum = sum(log(1 + exp(-tmpmat)));
		#cat("new sum(log(1+e(-y*XW)))", partsum, "wpart",  -0.5 * lambda * sum(wvec**2), "\n")
		currloss = -0.5 * lambda * sum(wvec**2) + partsum;
		frac = (prevloss - currloss) / prevloss;
		cat("Iter:", it, "prev loss:", currloss, "frac:", frac, "\n")
		if((it > 1) & (frac > 0) & (frac < eps)) { #converged
			break;
		}
		prevloss = currloss;

		gvec = -lambda * wvec + t(xmat) %*% (-yvec / (1 + exp(tmpmat)));
		cat("\tIter:", it, "gnorm:", sum(gvec**2), "\n")
		#cat("new gvec", gvec, "\n")
		wvec = wvec - alpha * gvec / it
		#cat("new wvec", wvec, "\n")
		it = it + 1;
	}
	return (LogRegDense(Coefs = list(wvec), NumSplits = 1, LossVal = currloss, Call = call))
}

#thenb is an instance of LogRegDense; newdata is an instance of SingDataSparse; output is a vector of +-1
LogRegDenseSingScore <- function (thenb, newdata) {
	fullcoefs = unlist(thenb@Coefs)
	pred = data.matrix(newdata@SingTable) %*% fullcoefs;
	pred = sign(pred)
	pred[pred == 0] = 1
	#return (table(data.matrix(newdata@Target), data.matrix(pred)))
	return (data.matrix(pred))
}

#this is the factorized learning implementation
LogRegDenseMultLearn <- function (amultdata, lambda = 0.0, alpha = 0.1, eps = 0.01, maxiters = 100) {
	call <- match.call()
	yvec = data.matrix(amultdata@Target);
	
	numrs = length(amultdata@KFKDs);
	ns = nrow(amultdata@EntTable);
	listFKs = list(amultdata@KFKDs[[1]]@EntCol)	
	listJs <- list(sparseMatrix(seq(1,ns), amultdata@EntTable[,listFKs[[1]]], x=rep(1,ns), dims=c(ns, nrow(amultdata@AttTables[[1]])))) #get sparse matrices J of dim len(FK) x len(unique(FK)); rid are already assumed to be linear; dims added since not all rid might be in fk
	if(numrs > 1) {
		for (i in 2:numrs) {
			listFKs <- append(listFKs, amultdata@KFKDs[[i]]@EntCol)
			listJs <- append(listJs, sparseMatrix(seq(1,ns), amultdata@EntTable[,listFKs[[i]]], x=rep(1,ns), dims=c(ns, nrow(amultdata@AttTables[[i]]))))
		}
	}
	#need to use only XS
	attribsS <- setdiff(names(amultdata@EntTable), listFKs); #incl intercept feature; excl FKs
	#wvec = rep(0.0, ncol(xmat));
	wvecs = list(rep(0.0, length(attribsS)))
	for(i in 1:numrs) {
		wri = rep(0.0, ncol(amultdata@AttTables[[i]])) #rid is implicitly the row num; rid/fk usable as feature would have been spadensified as part of XRi
		wvecs = append(wvecs, list(wri))
	}
	prevloss = nrow(amultdata@EntTable) * log(2.0);
	#cat("Init loss: ", prevloss, "\n")
	it = 1;
	while(it <= maxiters) {
		gvecs = list(-lambda * wvecs[[1]])
		for(i in 2:(1+numrs)) {
			gvecs = append(gvecs, list(-lambda * wvecs[[i]]))
		}
		fips = data.matrix(amultdata@EntTable[,attribsS]) %*% wvecs[[1]]; #partial inner product portion from S; used for adding more; reordered per ri
		for(i in 1:numrs) { #get partial ips from each R
			pipi = data.matrix(amultdata@AttTables[[i]]) %*% wvecs[[(1 + i)]]
			fips = fips + listJs[[i]] %*% pipi; #join done implicitly using sparse matrix mult!
		}
		fips = data.matrix(fips)
		tmpmatyip = yvec * fips
		partsum = sum(log(1 + exp(-tmpmatyip)))
		currloss = -0.5 * lambda * sum(unlist(wvecs)**2) + partsum;
		frac = (prevloss - currloss) / prevloss;
		cat("Iter:", it, "prev loss:", currloss, "frac:", frac, "\n")
		if((it > 1) & (frac > 0) & (frac < eps)) { #converged
			break;
		}
		prevloss = currloss
		tmpmat = -yvec / (1 + exp(tmpmatyip))
		gvecs[[1]] = gvecs[[1]] + t(data.matrix(amultdata@EntTable[,attribsS])) %*% tmpmat;
		for(i in 1:numrs) {
			gipi = data.matrix(t(listJs[[i]]) %*% tmpmat); #join and group by sum done implicitly using transpose of sparse matrix mult!
			gvecs[[(1 + i)]] = gvecs[[(1 + i)]] + t(data.matrix(amultdata@AttTables[[i]])) %*% gipi;
		}
		#cat("new gvec", unlist(gvecs), "\n")
		cat("\tIter:", it, "gnorm:", sum(unlist(gvecs)**2), "\n")
		# wvec = wvec - alpha * gvec;
		for(i in 1:(1 + numrs)) {
			wvecs[[i]] = wvecs[[i]] - alpha * gvecs[[i]] / it
		}
		#cat("new wvec", unlist(wvecs), "\n");
		it = it + 1;
	}
	return (LogRegDense(Coefs = wvecs, NumSplits = (1 + numrs), LossVal = currloss, Call = call))	
}

#thenb is an instance of LogRegDense; newdata is an instance of MultDataSparse; output is a vector of +-1
#the att tables list, the cols of forkeys matrix are assumed to be aligned; same with coef vecs, starting with ent table's
#also assumed is that the forkey matrix columns encode the actual rownums in the att tables
LogRegDenseMultScore <- function(thenb, multnewdata) {
	numrs <- length(multnewdata@KFKDs);
	ns <- nrow(multnewdata@EntTable);
	listFKs <- list(multnewdata@KFKDs[[1]]@EntCol)
	listJs <- list(sparseMatrix(seq(1,ns), multnewdata@EntTable[,listFKs[[1]]], x=rep(1,ns), dims=c(ns, nrow(multnewdata@AttTables[[1]])))) #get sparse matrices J of dim len(FK) x len(unique(FK)); rid are already assumed to be linear
	if(numrs > 1) {
		for (i in 2:numrs) {
			listFKs <- append(listFKs, multnewdata@KFKDs[[i]]@EntCol)
			listJs <- append(listJs, sparseMatrix(seq(1,ns), multnewdata@EntTable[,listFKs[[i]]], x=rep(1,ns), dims=c(ns, nrow(multnewdata@AttTables[[i]]))))
		}
	}
	#need to use only XS
	attribsS <- setdiff(names(multnewdata@EntTable), listFKs); #incl intercept feature; excl FKs
	
	fips = data.matrix(multnewdata@EntTable[,attribsS]) %*% thenb@Coefs[[1]]; #partial inner product portion from S; used for adding more; reordered per ri
	for(i in 1:numrs) { #get partial ips from each R
		pipi = data.matrix(multnewdata@AttTables[[i]]) %*% thenb@Coefs[[(1 + i)]]
		fips = fips + listJs[[i]] %*% pipi; #join done implicitly using sparse matrix mult!
	}
	fips = sign(data.matrix(fips))
	fips[fips == 0] = 1
	#return (table(data.matrix(multnewdata@Target), fips))
	return (fips)
}
