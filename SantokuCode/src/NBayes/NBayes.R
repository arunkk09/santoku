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

NBayes <- setClass (
	"NBayes",
	slots = c (
		LogYPT = "table",
		LogCPTs = "list",
		Levels = "character",
		Call = "call"
		)
	)

#asingdata is an instance of SingData; output is an instance of NBayes
NBayesSingLearn <- function (asingdata, laplace = 1, threshold = 0.001, eps = 0) {
	call <- match.call()
	logest <- function(var) {
		tab <- table(asingdata@Target[,1], var)
		prob <- (tab + laplace) / (rowSums(tab) + laplace * nlevels(var))
		prob[prob <= eps] <- threshold
		return (log(prob))
	}
	logypt = log(table(asingdata@Target))
	logcpts = lapply(as.data.frame(asingdata@SingTable), logest)
	return (NBayes(LogYPT = logypt, LogCPTs = logcpts, Levels = levels(asingdata@Target[,1]), Call = call))
}

#thenb is an instance of NBayes; newdata is an instance of SingData; output is a factor
NBayesSingScore <- function (thenb, newdata, ...) {
	attribs = names(newdata@SingTable)
	newdatad = as.data.frame(newdata@SingTable);
	newdatat = data.matrix(newdatad);
	L <- matrix(rep(thenb@LogYPT, nrow(newdatad)), nrow=length(thenb@Levels), byrow=FALSE);
	for(v in 1:ncol(newdatat)) {
		for(co in 1:nrow(newdatat)) {
			nd <- newdatat[co,v];
			L[,co] <- L[,co] + thenb@LogCPTs[[v]][,nd];
		}
	}
	FL <- factor(thenb@Levels[apply(L, 2, which.max)], levels=thenb@Levels)
	return (FL)
}

#this is the factorized learning implementation, optimized version
#FK is rownum indicator into R similar to logreg datasets; PK is another feature in XR!
#if usefk is false, PK is not used as a feature
NBayesMultLearn <- function (amultdata, laplace = 1, threshold = 0.001, eps = 0) {
	call <- match.call()
	s <- as.data.frame(amultdata@EntTable)
	lfks <- length(amultdata@KFKDs);
	y <- amultdata@Target[,1]
	ndomy <- length(levels(y));
	logest <- function(var) {
		tab <- table(y, var)
		prob <- (tab + laplace) / (rowSums(tab) + laplace * nlevels(var))
		prob[prob <= eps] <- threshold
		return (log(prob))
	}
	logestR <- function(i) {
		pkRi <- (amultdata@KFKDs[[i]])@AttCol
		dt <- table(y, s[,pkRi]) #attcol and entcol have same names but different semantics; TODO: what if fk does not have some rids? out of order and maybe out of bounds for dt[,ti]!
		ndt <- as.numeric(colnames(dt))
		Ri <- amultdata@AttTables[[i]]
		mRi <- data.matrix(Ri)
		Rfeats <- names(Ri)
		if(amultdata@KFKDs[[i]]@UseFK == FALSE) {
			Rfeats <- setdiff(Rfeats, pkRi)
		}
		logestRfeat <- function(var) {
			varind <- match(var, names(Ri))
			tab <- matrix(rep(0, ndomy*length(levels(Ri[,var]))), nrow=ndomy)
			dimnames(tab) <- list(levels(y), levels(Ri[,var]))
			tii <- 1
			for (ti in ndt) {
				tab[,mRi[ti,varind]] <- tab[,mRi[ti,varind]] + dt[,tii]
				tii <- tii + 1
			}
			prob <- (tab + laplace) / (rowSums(tab) + laplace * ncol(tab))
			prob[prob <= eps] <- threshold
			return (log(prob))
		}
		logcptsRi <- lapply(Rfeats, logestRfeat)
		return(logcptsRi)
	}
	logypt <- log(table(y))
	listfks <- list()
	for (i in 1:lfks) {
		listfks <- append(listfks, amultdata@KFKDs[[i]]@EntCol)
	}
	attribsS <- setdiff(names(s), listfks)
	logcpts <- list()
	allnames <- list()
	if(length(attribsS) > 0) {
		logcpts <- lapply(as.data.frame(s[,attribsS]), logest)
		allnames <- attribsS
	}
	for (i in 1:lfks) {
		logcpts <- append(logcpts, logestR(i))
		Rfeats <- names(amultdata@AttTables[[i]])
		if(amultdata@KFKDs[[i]]@UseFK == FALSE) {
			Rfeats <- setdiff(Rfeats, (amultdata@KFKDs[[i]])@AttCol)
		}
		allnames <- append(allnames, Rfeats)
	}
	names(logcpts) <- allnames
	return (NBayes(LogYPT = logypt, LogCPTs = logcpts, Levels = levels(y), Call = call))
}


#factorized scoring with S, a list of FK attrib names in S, a list of Ri data frames in same order with same PK names as FKs
#this is the optimized version with FK being rownums and PK being another feature in XR
NBayesMultScore <- function(thenb, multnewdata, ...) {
	#for each Ri, we precompute the logsum probs component from features in Ri and store it in a 2-col data frame keyed by FK value
	#so we get sum(log(prob for all XRij and FKi)) and store it first as presumi (i=1:k for Ri tables)
	#the overall flow is as follows: logsum = logy + sum(logprob for all XSj only) + presum1 for FKi + ... + presumk for FK k
	#note that since logcpts already have the log probs, there is no need to invoke the log function here!
	#if usefk is false, do not include PK/FK as a feature in CPT; it is purely a physical link
	#pt = proc.time();
	lfks <- length(multnewdata@KFKDs);
	lfkinds <- 1:lfks;
	ndomy <- length(thenb@Levels)
	presumi <- function(i) {
		pkRi <- multnewdata@KFKDs[[i]]@AttCol;
		Ridf <- as.data.frame(multnewdata@AttTables[[i]]);
		attribsRi <- names(Ridf);
		newRi <- data.matrix(Ridf);
		getlevels <- function(v) {
			return(levels(Ridf[[v]]))
		}
		alllevels <- sapply(1:ncol(newRi), getlevels);
		pkRiind <- match(pkRi, attribsRi) #assuming all feature names are unique
		cptindxri <- rep(0,length(attribsRi)); #integer indices into thenb cpts for attribsRi features
		for(j in 1:length(cptindxri)) {
			cptindxri[j] <- match(c(attribsRi[j]),names(thenb@LogCPTs))
		}
		preRi <- matrix(rep(0, ndomy * nrow(Ridf)),nrow=length(thenb@Levels), byrow=TRUE);
		seqalong <- seq_along(attribsRi)
		if((multnewdata@KFKDs[[i]])@UseFK == FALSE) {
			seqalong <- setdiff(seqalong, pkRiind)
		}
		for (co in 1:nrow(newRi)) {
			preprobv <- function(v) {
				rd <- newRi[co,v];
				return(thenb@LogCPTs[[cptindxri[v]]][,alllevels[[v]][rd]])
			}
			#preRi[,co] <- apply(log(sapply(seq_along(attribsRi), preprobv)), 1, sum)
			#a major bug arose due to an implicit assumption that order of fks and order of levels as factors are the same! but they are not! it has been fixed now using pkRi as below:
			#preRi[,newRi[co,pkRiind]] <- apply(log(sapply(seq_along(attribsRi), preprobv)), 1, sum)
			preRi[,co] <- apply(sapply(seqalong, preprobv), 1, sum)
		}
		return (preRi)
	}
	presums <- list(presumi(1))
	listFKs <- list(multnewdata@KFKDs[[1]]@EntCol)
	if(lfks > 1) {
		for (i in 2:lfks) {
			presums <- append(presums, list(presumi(i)))
			listFKs <- append(listFKs, multnewdata@KFKDs[[i]]@EntCol)
		}
	}
	#need to use only XSj since logprob for FKi is present in presumi
	attribsS <- setdiff(names(multnewdata@EntTable), listFKs); 
	cptindx <- rep(0,length(attribsS)); #integer indices into thenb cpts for attribS features
	for(i in 1:length(cptindx)) {
		cptindx[i] <- match(c(attribsS[i]),names(thenb@LogCPTs))
	}
	ndindx <- rep(0,length(attribsS)); #integer indices into newdata columns for attribS features
	for(i in 1:length(ndindx)) {
		ndindx[i] <- match(c(attribsS[i]),names(multnewdata@EntTable))
	}
	newdata <- data.matrix(multnewdata@EntTable);
	L <- matrix(rep(thenb@LogYPT, nrow(newdata)),nrow=length(thenb@Levels), byrow=FALSE);
	if(length(attribsS) > 0) {
		for(v in 1:length(attribsS)) {
			for(co in 1:nrow(newdata)) {
				nd <- newdata[co,ndindx[v]];
				L[,co] <- L[,co] + thenb@LogCPTs[[cptindx[v]]][,nd]
			}
		}
	}
	for(co in 1:nrow(newdata)) {
		fksum <- rep(0.0, length(thenb@Levels))
		for (k in lfkinds) {
			tmp = newdata[co,listFKs[[k]]]
			fksum <- fksum + presums[[k]][,tmp]
		}
		L[,co] <- L[,co] + fksum
	}
	FL <- factor(thenb@Levels[apply(L, 2, which.max)], levels=thenb@Levels)
	return (FL)
}
