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

library(bnlearn)

TANBayesSingLearn <- function (asingdata) {
	s <- as.data.frame(cbind(asingdata@Target, asingdata@SingTable))
	names(s) <- c(asingdata@TargetName, names(asingdata@SingTable))
	mytan <- tree.bayes(s, asingdata@TargetName)
	return (mytan)
}

TANBayesSingScore <- function (thetan, newdata) {
	s <- as.data.frame(cbind(newdata@Target, newdata@SingTable))
	names(s) <- c(newdata@TargetName, names(newdata@SingTable))
	pred <- predict(thetan, s)
	return (pred)
}

#this is the factorized learning implementation
#assumed that the factor levels match in the training set and the base tables
#the foreign key in S is assumed to have the same name as the RID in corresp R
#owing to the fl-tan theorem, we can simply ignore all Rs and use S alone (ties can screw this up a bit)
#the only exception is when fk is not usable - then, merge that R and S into Snew
TANBayesMultLearn <- function (amultdata) {
	s <- as.data.frame(cbind(amultdata@Target, amultdata@EntTable))
	names(s) <- c("Y", names(amultdata@EntTable))
	lfks <- length(amultdata@KFKDs);
	for (i in 1:lfks) {
		if(amultdata@KFKDs[[i]]@UseFK == FALSE) { #merge r into snew, exclude fk; need to replace fk in r with seq int col
			r <- amultdata@AttTables[[i]]
			xrfeats <- setdiff(names(r), amultdata@KFKDs[[i]]@AttCol)
			r <- as.data.frame(cbind(seq(1,nrow(r)), r[,xrfeats]))
			names(r) <- c(amultdata@KFKDs[[i]]@AttCol, xrfeats)
			s <- merge(s, r)
			remfeats <- setdiff(names(s), amultdata@KFKDs[[i]]@EntCol)
			s <- as.data.frame(s[,remfeats])
		}
	}
	mytan <- tree.bayes(s, "Y")
	return (mytan)
}

TANBayesMultScore <- function(thetan, multnewdata) {
	s <- as.data.frame(cbind(multnewdata@Target, multnewdata@EntTable))
	names(s) <- c("Y", names(multnewdata@EntTable))
	lfks <- length(multnewdata@KFKDs);
	for (i in 1:lfks) {
		if(multnewdata@KFKDs[[i]]@UseFK == FALSE) { #merge r into snew, exclude fk
			r <- multnewdata@AttTables[[i]]
			xrfeats <- setdiff(names(r), multnewdata@KFKDs[[i]]@AttCol)
			r <- as.data.frame(cbind(seq(1,nrow(r)), r[,xrfeats]))
			names(r) <- c(multnewdata@KFKDs[[i]]@AttCol, xrfeats)
			s <- merge(s, r)
			remfeats <- setdiff(names(s), multnewdata@KFKDs[[i]]@EntCol)
			s <- as.data.frame(s[,remfeats])
		}
	}
	pred <- predict(thetan, s)
	return (pred)
}
