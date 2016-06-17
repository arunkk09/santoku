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

library(Matrix)

#class for a functional dependency (FD); lhs and rhs are lists of feature names (strings)
FD <- setClass (
	"FD", 
	slots = c (
		LHS = "character",
		RHS = "character",
		UseLHS = "logical"
		)
	)

SingData <- setClass (
	"SingData",
	slots = c (
		Target = "data.frame",
		TargetName = "character",
		SingTable = "data.frame",
		FDs = "list"
		)
	)

SingDataSparse <- setClass (
	"SingDataSparse",
	slots = c (
		Target = "data.frame",
		SingTable = "Matrix"
		)
	)

setGeneric (
	name = "AddFD",
	def = function (theObject, thefd) {
			standardGeneric ("AddFD")
		}
	)

setMethod (
	f = "AddFD",
	signature = "SingData",
	definition = function (theObject, thefd) {
				theObject@FDs <- c (theObject@FDs, thefd)
				validObject (theObject)
				return (theObject)
			}
	)

KFKD <- setClass (
	"KFKD", 
	slots = c (
		EntCol = "character",
		AttCol = "character",
		UseFK = "logical"
		)
	)

#attribute tables and attribute table names are assumed to be in order
MultData <- setClass (
	"MultData",
	slots = c (
		Target = "data.frame",
		EntTable = "data.frame",
		AttTables = "list",
		KFKDs = "list"
		)
	)

MultDataSparse <- setClass (
	"MultDataSparse",
	slots = c (
		Target = "data.frame",
		EntTable = "Matrix",
		AttTables = "list",
		ForKeys = "matrix"
		)
	)

setGeneric (
	name = "AddKFKD",
	def = function (theObject, thekfkd) {
			standardGeneric ("AddKFKD")
		}
	)

setMethod (
	f = "AddKFKD",
	signature = "MultData",
	definition = function (theObject, thekfkd) {
				theObject@KFKDs <- c (theObject@KFKDs, thekfkd)
				validObject (theObject)
				return (theObject)
			}
	)

