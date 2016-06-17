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

library(shiny)
library(caret)
library(bnlearn)
library(data.table)
library(Matrix)
source("../src/Classes.R")
source("../src/LogReg/LogRegDense.R")
source("../src/NBayes/NBayes.R")
source("../src/TANBayes/TANBayes.R")
options(scipen=999, digits=5)
set.seed(123)

#Walmart for lr
DWtr = read.csv("../../SantokuData/Walmart/DWTtraintestSing10p.csv")
DWts = read.csv("../../SantokuData/Walmart/DWTholdSing10p.csv")
DWStr = read.csv("../../SantokuData/Walmart/DWTtraintestMultS10p.csv") #entity table alone
DWSts = read.csv("../../SantokuData/Walmart/DWTholdMultS10p.csv")
DWR1 = read.csv("../../SantokuData/Walmart/DWTMultR1.csv")
DWR2 = read.csv("../../SantokuData/Walmart/DWTMultR210p.csv")
nDWS = names(DWStr);
nDWR1 = names(DWR1);
nDWR2 = names(DWR2);
nDWA = names(DWtr);

#Walmart for others
Wtr = read.csv("../../SantokuData/Walmart/WTtraintest25p.csv")
Wts = read.csv("../../SantokuData/Walmart/WThold25p.csv")
WStr = read.csv("../../SantokuData/Walmart/WTtraintest25pMultS.csv")
WSts = read.csv("../../SantokuData/Walmart/WThold25pMultS.csv") 
WR1 = read.csv("../../SantokuData/Walmart/stores_disc.csv")
WR2 = read.csv("../../SantokuData/Walmart/features_disc25p.csv")
WStrnor1 = WStr
WStsnor1 = WSts
WStrnor1$storefk = factor(WStrnor1$storefk)
WStsnor1$storefk = factor(WStsnor1$storefk)
WStrnor2 = WStr
WStsnor2 = WSts
WStrnor2$purchaseidfk = factor(WStrnor2$purchaseidfk)
WStsnor2$purchaseidfk = factor(WStsnor2$purchaseidfk)
nWS = names(WStr);
nWR1 = names(WR1);
nWR2 = names(WR2);
nWA = names(Wtr);
#for TAN, the fks have to be made factors
TWSall = rbind(WStr,WSts)
TWStr = WStr
TWSts = WSts
TWSall$storefk = factor(TWSall$storefk)
TWSall$purchaseidfk = factor(TWSall$purchaseidfk)
TWStr$storefk = factor(TWStr$storefk, levels=levels(TWSall$storefk))
TWSts$storefk = factor(TWSts$storefk, levels=levels(TWSall$storefk))
TWStr$purchaseidfk = factor(TWStr$purchaseidfk, levels=levels(TWSall$purchaseidfk))
TWSts$purchaseidfk = factor(TWSts$purchaseidfk, levels=levels(TWSall$purchaseidfk))

#Yelp for lr
DYtr = read.csv("../../SantokuData/Yelp/DYRtraintestSingtop5p.csv")
DYts = read.csv("../../SantokuData/Yelp/DYRholdSingtop5p.csv")
DYStr = read.csv("../../SantokuData/Yelp/DYRtraintestMultStop5p.csv") #entity table alone
DYSts = read.csv("../../SantokuData/Yelp/DYRholdMultStop5p.csv")
DYR1 = read.csv("../../SantokuData/Yelp/DYRMultR1top5p.csv")
DYR2 = read.csv("../../SantokuData/Yelp/DYRMultR2top5p.csv")
nDYS = names(DYStr);
nDYR1 = names(DYR1);
nDYR2 = names(DYR2);
nDYA = names(DYtr);

#Yelp for others
Ytr = read.csv("../../SantokuData/Yelp/YRtraintesttop5p.csv")
Yts = read.csv("../../SantokuData/Yelp/YRholdtop5p.csv")
Ytr = Ytr[,c("stars","useridfk","businessidfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender","bstars","wday5","wday3","wday4","wend5","wend3","wend4",
             "wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501","cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
Yts = Yts[,c("stars","useridfk","businessidfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender","bstars","wday5","wday3","wday4","wend5","wend3","wend4",
             "wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501","cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
YR1 = read.csv("../../SantokuData/Yelp/user_disc_gendtop5p.csv")
YR1 = YR1[,c("useridfk","ureviewcnt","ustars","vuseful","vfunny","vcool","gender")]
YR2 = read.csv("../../SantokuData/Yelp/business_checkin_disctop5p.csv")
YR2 = YR2[,c("businessidfk","bstars","wday5","wday3","wday4","wend5","wend3","wend4","wday2","wend2","wend1","wday1","cat109","cat344","cat33","city","cat501",
             "cat404","cat259","cat246","cat79","open","cat221","latitude","longitude")]
Ytr$useridfk = factor(Ytr$useridfk, levels=levels(YR1$useridfk))
Yts$useridfk = factor(Yts$useridfk, levels=levels(YR1$useridfk))
YStr = read.csv("../../SantokuData/Yelp/YRtraintesttop5pMultS.csv")
YSts = read.csv("../../SantokuData/Yelp/YRholdtop5pMultS.csv")
YStrnor1 = YStr
YStsnor1 = YSts
YStrnor1$useridfk = factor(YStrnor1$useridfk)
YStsnor1$useridfk = factor(YStsnor1$useridfk)
YStrnor2 = YStr
YStsnor2 = YSts
YStrnor2$businessidfk = factor(YStrnor2$businessidfk)
YStsnor2$businessidfk = factor(YStsnor2$businessidfk)
nYS = names(YStr);
nYR1 = names(YR1);
nYR2 = names(YR2);
nYA = names(Ytr);
remuid = setdiff(unique(YSts$useridfk), unique(YStr$useridfk)) #to ensure all uids appear in tr
YSts1 = subset(YSts, is.element(YSts$useridfk, remuid))
YSts2 = subset(YSts, !is.element(YSts$useridfk, remuid))
TYStr = rbind(YStr,YSts1)
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

#Expedia for lr
DEtr = read.csv("../../SantokuData/Expedia/DEHtraintest10reSingtop5p.csv")
DEts = read.csv("../../SantokuData/Expedia/DEHhold10reSingtop5p.csv")
DEStr = read.csv("../../SantokuData/Expedia/DEHtraintest10reMultStop5p.csv") #entity table alone
DESts = read.csv("../../SantokuData/Expedia/DEHhold10reMultStop5p.csv")
DER1 = read.csv("../../SantokuData/Expedia/DEHMultR1top5p.csv")
DER2 = read.csv("../../SantokuData/Expedia/DEHMultR2top5p.csv")
nDES = names(DEStr);
nDER1 = names(DER1);
nDER2 = names(DER2);
nDEA = names(DEtr);

#Expedia for others
Etr = read.csv("../../SantokuData/Expedia/EHtraintest10retop5p.csv")
Ets = read.csv("../../SantokuData/Expedia/EHhold10retop5p.csv")
ER1 = read.csv("../../SantokuData/Expedia/hotels_disc10retop5p.csv")
ER2 = read.csv("../../SantokuData/Expedia/searches_disc10retop5p.csv")
Etr$srch_destination_id = factor(Etr$srch_destination_id, levels=levels(ER2$srch_destination_id))
Ets$srch_destination_id = factor(Ets$srch_destination_id, levels=levels(ER2$srch_destination_id))
Etr$visitor_location_country_id = factor(Etr$visitor_location_country_id, levels=levels(ER2$visitor_location_country_id))
Ets$visitor_location_country_id = factor(Ets$visitor_location_country_id, levels=levels(ER2$visitor_location_country_id))
Etr$prop_idfk = factor(Etr$prop_idfk, levels=levels(ER1$prop_idfk))
Ets$prop_idfk = factor(Ets$prop_idfk, levels=levels(ER1$prop_idfk))
Etr$srch_idfk = factor(Etr$srch_idfk, levels=levels(ER2$srch_idfk))
Ets$srch_idfk = factor(Ets$srch_idfk, levels=levels(ER2$srch_idfk))
EStr = read.csv("../../SantokuData/Expedia/EHtraintest10retop5pMultS.csv")
ESts = read.csv("../../SantokuData/Expedia/EHhold10retop5pMultS.csv")
EStrnor1 = EStr
EStsnor1 = ESts
EStrnor1$prop_idfk = factor(EStrnor1$prop_idfk)
EStsnor1$prop_idfk = factor(EStsnor1$prop_idfk)
EStrnor2 = EStr
EStsnor2 = ESts
EStrnor2$srch_idfk = factor(EStrnor2$srch_idfk)
EStsnor2$srch_idfk = factor(EStsnor2$srch_idfk)
nES = names(EStr);
nER1 = names(ER1);
nER2 = names(ER2);
nEA = names(Etr);
#for TAN, the fks have to be made factors
TEStr = EStr
TESts = ESts
TESall = rbind(TEStr,TESts)
TESall$prop_idfk = factor(TESall$prop_idfk)
TESall$srch_idfk = factor(TESall$srch_idfk)
TEStr$prop_idfk = factor(TEStr$prop_idfk, levels=levels(TESall$prop_idfk))
TEStr$srch_idfk = factor(TEStr$srch_idfk, levels=levels(TESall$srch_idfk))
TESts$prop_idfk = factor(TESts$prop_idfk, levels=levels(TESall$prop_idfk))
TESts$srch_idfk = factor(TESts$srch_idfk, levels=levels(TESall$srch_idfk))

#Flights for lr
DFtr = read.csv("../../SantokuData/Flights/DOFtraintestnewSingtop5p.csv")
DFts = read.csv("../../SantokuData/Flights/DOFholdnewSingtop5p.csv")
DFStr = read.csv("../../SantokuData/Flights/DOFtraintestnewMultStop5p.csv") #entity table alone
DFSts = read.csv("../../SantokuData/Flights/DOFholdnewMultStop5p.csv")
DFR1 = read.csv("../../SantokuData/Flights/DOFMultR1top5p.csv")
DFR2 = read.csv("../../SantokuData/Flights/DOFMultR2top5p.csv")
DFR3 = read.csv("../../SantokuData/Flights/DOFMultR3top5p.csv")
nDFS = names(DFStr);
nDFR1 = names(DFR1);
nDFR2 = names(DFR2);
nDFR3 = names(DFR3);
nDFA = names(DFtr);

#Flights for others
Ftr = read.csv("../../SantokuData/Flights/OFtraintestnewtop5p.csv")
Fts = read.csv("../../SantokuData/Flights/OFholdnewtop5p.csv")
FR1 = read.csv("../../SantokuData/Flights/airlinesc_discnewtop5p.csv")
FR2 = read.csv("../../SantokuData/Flights/sairportsc_discnewtop5p.csv")
FR3 = read.csv("../../SantokuData/Flights/dairportsc_discnewtop5p.csv")
Ftr$airlineid = factor(Ftr$airlineid, levels=levels(FR1$airlineid))
Ftr$acountry = factor(Ftr$acountry, levels=levels(FR1$acountry))
Ftr$sairportid = factor(Ftr$sairportid, levels=levels(FR2$sairportid))
Ftr$scity = factor(Ftr$scity, levels=levels(FR2$scity))
Ftr$scountry = factor(Ftr$scountry, levels=levels(FR2$scountry))
Ftr$dairportid = factor(Ftr$dairportid, levels=levels(FR3$dairportid))
Ftr$dcity = factor(Ftr$dcity, levels=levels(FR3$dcity))
Ftr$dcountry = factor(Ftr$dcountry, levels=levels(FR3$dcountry))
Fts$airlineid = factor(Fts$airlineid, levels=levels(FR1$airlineid))
Fts$acountry = factor(Fts$acountry, levels=levels(FR1$acountry))
Fts$sairportid = factor(Fts$sairportid, levels=levels(FR2$sairportid))
Fts$scity = factor(Fts$scity, levels=levels(FR2$scity))
Fts$scountry = factor(Fts$scountry, levels=levels(FR2$scountry))
Fts$dairportid = factor(Fts$dairportid, levels=levels(FR3$dairportid))
Fts$dcity = factor(Fts$dcity, levels=levels(FR3$dcity))
Fts$dcountry = factor(Fts$dcountry, levels=levels(FR3$dcountry))
FStr = read.csv("../../SantokuData/Flights/OFtraintestnewtop5pMultS.csv")
FSts = read.csv("../../SantokuData/Flights/OFholdnewtop5pMultS.csv")
FStrnor1 = FStr
FStsnor1 = FSts
FStrnor1$airlineid = factor(FStrnor1$airlineid)
FStsnor1$airlineid = factor(FStsnor1$airlineid)
FStrnor2 = FStr
FStsnor2 = FSts
FStrnor2$sairportid = factor(FStrnor2$sairportid)
FStsnor2$sairportid = factor(FStsnor2$sairportid)
FStrnor3 = FStr
FStsnor3 = FSts
FStrnor3$dairportid = factor(FStrnor3$dairportid)
FStsnor3$dairportid = factor(FStsnor3$dairportid)
FStrnor1r2 = FStr
FStsnor1r2 = FSts
FStrnor1r2$airlineid = factor(FStrnor1r2$airlineid)
FStsnor1r2$airlineid = factor(FStsnor1r2$airlineid)
FStsnor1r2$sairportid = factor(FStsnor1r2$sairportid)
FStsnor1r2$sairportid = factor(FStsnor1r2$sairportid)
FStrnor1r3 = FStr
FStsnor1r3 = FSts
FStrnor1r3$airlineid = factor(FStrnor1r3$airlineid)
FStrnor1r3$airlineid = factor(FStrnor1r3$airlineid)
FStsnor1r3$dairportid = factor(FStsnor1r3$dairportid)
FStsnor1r3$dairportid = factor(FStsnor1r3$dairportid)
FStrnor2r3 = FStr
FStsnor2r3 = FSts
FStrnor2r3$sairportid = factor(FStrnor2r3$sairportid)
FStrnor2r3$sairportid = factor(FStrnor2r3$sairportid)
FStsnor2r3$dairportid = factor(FStsnor2r3$dairportid)
FStsnor2r3$dairportid = factor(FStsnor2r3$dairportid)
nFS = names(FStr);
nFR1 = names(FR1);
nFR2 = names(FR2);
nFR3 = names(FR3);
nFA = names(Ftr);
TFStr = FStr
TFSts = FSts
TFSall = rbind(TFStr,TFSts)
TFSall$airlineid = factor(TFSall$airlineid)
TFSall$sairportid = factor(TFSall$sairportid)
TFSall$dairportid = factor(TFSall$dairportid)
TFStr$airlineid = factor(TFStr$airlineid, levels=levels(TFSall$airlineid))
TFStr$sairportid = factor(TFStr$sairportid, levels=levels(TFSall$sairportid))
TFStr$dairportid = factor(TFStr$dairportid, levels=levels(TFSall$dairportid))
TFSts$airlineid = factor(TFSts$airlineid, levels=levels(TFSall$airlineid))
TFSts$sairportid = factor(TFSts$sairportid, levels=levels(TFSall$sairportid))
TFSts$dairportid = factor(TFSts$dairportid, levels=levels(TFSall$dairportid))

shinyServer(function(input, output) {
  output$uideps <- renderUI({
    if(is.null(input$dbterm))
      return()
    switch(input$dataset,
           "Walmart" = switch(input$dbterm,
                              "Single table" = fluidRow(column(12, 
                                                               tags$b("Functional Dependencies:"),
                                                               fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE), 
                                                                               checkboxInput("usefk2", "Include LHS", TRUE)),
                                                                        column(5, selectInput("lhs1", "LHS 1", nWA, "storefk"), 
                                                                               selectInput("lhs2", "LHS 2", nWA, "purchaseidfk")),
                                                                        column(5, selectInput("rhs1", "RHS 1", nWA, multiple = TRUE), 
                                                                               selectInput("rhs2", "RHS 2", nWA, multiple = TRUE))))),
                              "Multi table" = fluidRow(column(12, 
                                                             tags$b("Key-Foreign Key Dependencies:"),
                                                             fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                             checkboxInput("usefk2", "Include FK", TRUE)),
                                                                      column(5, selectInput("fk1", "Foreign Key 1", nWS, "storefk"), 
                                                                             selectInput("fk2", "Foreign Key 2", nWS, "purchaseidfk")),
                                                                      column(5, selectInput("pk1", "Primary Key 1", nWR1, "storefk"), 
                                                                             selectInput("pk2", "Primary Key 2", nWR2, "purchaseidfk")))))
           ),
           "Walmart (R)" = switch(input$dbterm,
                                  "Single table" = fluidRow(column(12, 
                                                                   tags$b("Functional Dependencies:"),
                                                                   fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE), 
                                                                                   checkboxInput("usefk2", "Include LHS", TRUE)),
                                                                            column(5, selectInput("lhs1", "LHS 1", nDWA, "storefk"), 
                                                                                   selectInput("lhs2", "LHS 2", nDWA, "purchaseidfk")),
                                                                            column(5, selectInput("rhs1", "RHS 1", nDWA, multiple = TRUE), 
                                                                                   selectInput("rhs2", "RHS 2", nDWA, multiple = TRUE))))),
                                  "Multi table" = fluidRow(column(12, 
                                                                 tags$b("Key-Foreign Key Dependencies:"),
                                                                 fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                                 checkboxInput("usefk2", "Include FK", TRUE)),
                                                                          column(5, selectInput("fk1", "Foreign Key 1", nDWS, "storefk"), 
                                                                                 selectInput("fk2", "Foreign Key 2", nDWS, "purchaseidfk")),
                                                                          column(5, selectInput("pk1", "Primary Key 1", c("storefk", nDWR1), "storefk"), 
                                                                                 selectInput("pk2", "Primary Key 2", c("purchaseidfk", nDWR2), "purchaseidfk")))))
           ),
           "Yelp" = switch(input$dbterm,
                              "Single table" = fluidRow(column(12, 
                                                               tags$b("Functional Dependencies:"),
                                                               fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE), 
                                                                               checkboxInput("usefk2", "Include LHS", TRUE)),
                                                                        column(5, selectInput("lhs1", "LHS 1", nYA, "useridfk"), 
                                                                               selectInput("lhs2", "LHS 2", nYA, "businessidfk")),
                                                                        column(5, selectInput("rhs1", "RHS 1", nYA, multiple = TRUE), 
                                                                               selectInput("rhs2", "RHS 2", nYA, multiple = TRUE))))),
                              "Multi table" = fluidRow(column(12, 
                                                             tags$b("Key-Foreign Key Dependencies:"),
                                                             fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                             checkboxInput("usefk2", "Include FK", TRUE)),
                                                                      column(5, selectInput("fk1", "Foreign Key 1", nYS, "useridfk"), 
                                                                             selectInput("fk2", "Foreign Key 2", nYS, "businessidfk")),
                                                                      column(5, selectInput("pk1", "Primary Key 1", nYR1, "useridfk"), 
                                                                             selectInput("pk2", "Primary Key 2", nYR2, "businessidfk")))))
           ),
           "Yelp (R)" = switch(input$dbterm,
                                  "Single table" = fluidRow(column(12, 
                                                                   tags$b("Functional Dependencies:"),
                                                                   fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE), 
                                                                                   checkboxInput("usefk2", "Include LHS", TRUE)),
                                                                            column(5, selectInput("lhs1", "LHS 1", nDYA, "useridfk"), 
                                                                                   selectInput("lhs2", "LHS 2", nDYA, "businessidfk")),
                                                                            column(5, selectInput("rhs1", "RHS 1", nDYA, multiple = TRUE), 
                                                                                   selectInput("rhs2", "RHS 2", nDYA, multiple = TRUE))))),
                                  "Multi table" = fluidRow(column(12, 
                                                                 tags$b("Key-Foreign Key Dependencies:"),
                                                                 fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                                 checkboxInput("usefk2", "Include FK", TRUE)),
                                                                          column(5, selectInput("fk1", "Foreign Key 1", nDYS, "useridfk"), 
                                                                                 selectInput("fk2", "Foreign Key 2", nDYS, "businessidfk")),
                                                                          column(5, selectInput("pk1", "Primary Key 1", c("useridfk", nDYR1), "useridfk"), 
                                                                                 selectInput("pk2", "Primary Key 2", c("businessidfk", nDYR2), "businessidfk")))))
           ),
           "Expedia" = switch(input$dbterm,
                              "Single table" = fluidRow(column(12, 
                                                               tags$b("Functional Dependencies:"),
                                                               fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE), 
                                                                               checkboxInput("usefk2", "Include LHS", FALSE)),
                                                                        column(5, selectInput("lhs1", "LHS 1", nEA, "prop_idfk"), 
                                                                               selectInput("lhs2", "LHS 2", nEA, "srch_idfk")),
                                                                        column(5, selectInput("rhs1", "RHS 1", nEA, multiple = TRUE), 
                                                                               selectInput("rhs2", "RHS 2", nEA, multiple = TRUE))))),
                              "Multi table" = fluidRow(column(12, 
                                                             tags$b("Key-Foreign Key Dependencies:"),
                                                             fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                             checkboxInput("usefk2", "Include FK", FALSE)),
                                                                      column(5, selectInput("fk1", "Foreign Key 1", nES, "prop_idfk"), 
                                                                             selectInput("fk2", "Foreign Key 2", nES, "srch_idfk")),
                                                                      column(5, selectInput("pk1", "Primary Key 1", nER1, "prop_idfk"), 
                                                                             selectInput("pk2", "Primary Key 2", nER2, "srch_idfk")))))
           ),
           "Expedia (R)" = switch(input$dbterm,
                               "Single table" = fluidRow(column(12, 
                                                                tags$b("Functional Dependencies:"),
                                                                fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE), 
                                                                                checkboxInput("usefk2", "Include LHS", FALSE)),
                                                                         column(5, selectInput("lhs1", "LHS 1", nDEA, "prop_idfk"), 
                                                                                selectInput("lhs2", "LHS 2", nDEA, "srch_idfk")),
                                                                         column(5, selectInput("rhs1", "RHS 1", nDEA, multiple = TRUE), 
                                                                                selectInput("rhs2", "RHS 2", nDEA, multiple = TRUE))))),
                               "Multi table" = fluidRow(column(12, 
                                                              tags$b("Key-Foreign Key Dependencies:"),
                                                              fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                              checkboxInput("usefk2", "Include FK", FALSE)),
                                                                       column(5, selectInput("fk1", "Foreign Key 1", nDES, "prop_idfk"), 
                                                                              selectInput("fk2", "Foreign Key 2", nDES, "srch_idfk")),
                                                                       column(5, selectInput("pk1", "Primary Key 1", c("prop_idfk", nDER1), "prop_idfk"), 
                                                                              selectInput("pk2", "Primary Key 2", c("srch_idfk", nDER2), "srch_idfk")))))
           ),
           "Flights" = switch(input$dbterm,
                              "Single table" = fluidRow(column(12, 
                                                               tags$b("Functional Dependencies:"),
                                                               fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE),
                                                                               checkboxInput("usefk2", "Include LHS", TRUE),
                                                                               checkboxInput("usefk3", "Include LHS", TRUE)),
                                                                        column(5, selectInput("lhs1", "LHS 1", nFA, "airlineid"), 
                                                                               selectInput("lhs2", "LHS 2", nFA, "sairportid"), 
                                                                               selectInput("lhs3", "LHS 3", nFA, "dairportid")),
                                                                        column(5, selectInput("rhs1", "RHS 1", nFA, multiple = TRUE), 
                                                                               selectInput("rhs2", "RHS 2", nFA, multiple = TRUE), 
                                                                               selectInput("rhs3", "RHS 3", nFA, multiple = TRUE))))),
                              "Multi table" = fluidRow(column(12, 
                                                             tags$b("Key-Foreign Key Dependencies:"),
                                                             fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                             checkboxInput("usefk2", "Include FK", TRUE), 
                                                                             checkboxInput("usefk3", "Include FK", TRUE)),
                                                                      column(5, selectInput("fk1", "Foreign Key 1", nFS, "airlineid"), 
                                                                             selectInput("fk2", "Foreign Key 2", nFS, "sairportid"), 
                                                                             selectInput("fk3", "Foreign Key 3", nFS, "dairportid")),
                                                                      column(5, selectInput("pk1", "Primary Key 1", nFR1, "airlineid"), 
                                                                             selectInput("pk2", "Primary Key 2", nFR2, "sairportid"), 
                                                                             selectInput("pk3", "Primary Key 3", nFR3, "dairportid")))))
           ),
           "Flights (R)" = switch(input$dbterm,
                               "Single table" = fluidRow(column(12, 
                                                                tags$b("Functional Dependencies:"),
                                                                fluidRow(column(2, checkboxInput("usefk1", "Include LHS", TRUE),
                                                                                checkboxInput("usefk2", "Include LHS", TRUE),
                                                                                checkboxInput("usefk3", "Include LHS", TRUE)),
                                                                         column(5, selectInput("lhs1", "LHS 1", nDFA, "airlineid"), 
                                                                                selectInput("lhs2", "LHS 2", nDFA, "sairportid"), 
                                                                                selectInput("lhs3", "LHS 3", nDFA, "dairportid")),
                                                                         column(5, selectInput("rhs1", "RHS 1", nDFA, multiple = TRUE), 
                                                                                selectInput("rhs2", "RHS 2", nDFA, multiple = TRUE), 
                                                                                selectInput("rhs3", "RHS 3", nDFA, multiple = TRUE))))),
                               "Multi table" = fluidRow(column(12, 
                                                               tags$b("Key-Foreign Key Dependencies:"),
                                                               fluidRow(column(2, checkboxInput("usefk1", "Include FK", TRUE), 
                                                                               checkboxInput("usefk2", "Include FK", TRUE), 
                                                                               checkboxInput("usefk3", "Include FK", TRUE)),
                                                                        column(5, selectInput("fk1", "Foreign Key 1", nDFS, "airlineid"), 
                                                                               selectInput("fk2", "Foreign Key 2", nDFS, "sairportid"), 
                                                                               selectInput("fk3", "Foreign Key 3", nDFS, "dairportid")),
                                                                        column(5, selectInput("pk1", "Primary Key 1", c("airlineid", nDFR1), "airlineid"), 
                                                                               selectInput("pk2", "Primary Key 2", c("sairportid", nDFR2), "sairportid"),
                                                                               selectInput("pk3", "Primary Key 3", c("dairportid", nDFR3), "dairportid")))))
           )
           )#end switch
    })#end uideps
  
  output$uimlpt <- renderUI({
    if(is.null(input$mlalgo))
      return()
    switch(input$mlalgo,
           "lr" = fluidRow(column(12, numericInput("lambdal2", "L2 Regularizer", value=6000), 
                                  numericInput("alpha", "Stepsize", value=0.0002), 
                                  numericInput("maxiters", "Max Iterations", value=20))),
           "nb" = fluidRow(column(12, numericInput("laplace", "Laplace Parameter", min=1, value=1))),
           "dt" = radioButtons("dtsplit", "Splitting Function", choices = c("Information Gain", "Information Gain Ratio"))
           #"fr"= radioButtons("frsplit", "Splitting Function", choices = c("Information Gain", "Information Gain Ratio"))
    )#end switch
  })#end uimlpt
  
  trainres <- eventReactive(input$buttontrain, {
    thisfds = list(); singtrain = NULL; singtest = NULL
    thiskfkds = list(); multtrain = NULL; multtest = NULL
    yvectr = NULL; yvects = NULL; predvectr = NULL; predvects = NULL
    tim = 0.0
    if(input$dbterm == "Multi table") {
      thiskfkds = append(thiskfkds, KFKD(EntCol=input$fk1, AttCol=input$pk1, UseFK=input$usefk1))
      if(!is.null(input$fk2)) {
        thiskfkds = append(thiskfkds, KFKD(EntCol=input$fk2, AttCol=input$pk2, UseFK=input$usefk2))
      }
      if(!is.null(input$fk3)) {
        thiskfkds = append(thiskfkds, KFKD(EntCol=input$fk3, AttCol=input$pk3, UseFK=input$usefk3))
      }
      cat("KFKDs:\n")
      print(thiskfkds)
      multtrain = switch(input$dataset,
                         "Walmart" = MultData(Target=as.data.frame(WStr[,1]), EntTable=WStr[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                         "Walmart (R)" = MultData(Target=as.data.frame(DWStr[,1]), EntTable=DWStr[,-1], AttTables=list(DWR1, DWR2), KFKDs=thiskfkds),
                         "Yelp" = MultData(Target=as.data.frame(YStr[,1]), EntTable=YStr[,-1], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                         "Yelp (R)" = MultData(Target=as.data.frame(DYStr[,1]), EntTable=DYStr[,-1], AttTables=list(DYR1, DYR2), KFKDs=thiskfkds),
                         "Expedia" = MultData(Target=as.data.frame(EStr[,1]), EntTable=EStr[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                         "Expedia (R)" = MultData(Target=as.data.frame(DEStr[,1]), EntTable=DEStr[,-1], AttTables=list(DER1, DER2), KFKDs=thiskfkds),
                         "Flights" = MultData(Target=as.data.frame(FStr[,1]), EntTable=FStr[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds),
                         "Flights (R)" = MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,-1], AttTables=list(DFR1, DFR2, DFR3), KFKDs=thiskfkds)
      )
      multtest = switch(input$dataset,
                        "Walmart" = MultData(Target=as.data.frame(WSts[,1]), EntTable=WSts[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                        "Walmart (R)" = MultData(Target=as.data.frame(DWSts[,1]), EntTable=DWSts[,-1], AttTables=list(DWR1, DWR2), KFKDs=thiskfkds),
                        "Yelp" = MultData(Target=as.data.frame(YSts[,1]), EntTable=YSts[,-1], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                        "Yelp (R)" = MultData(Target=as.data.frame(DYSts[,1]), EntTable=DYSts[,-1], AttTables=list(DYR1, DYR2), KFKDs=thiskfkds),
                        "Expedia" = MultData(Target=as.data.frame(EStr[,1]), EntTable=EStr[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                        "Expedia (R)" = MultData(Target=as.data.frame(DEStr[,1]), EntTable=DEStr[,-1], AttTables=list(DER1, DER2), KFKDs=thiskfkds),
                        "Flights" = MultData(Target=as.data.frame(FSts[,1]), EntTable=FSts[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds),
                        "Flights (R)" = MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,-1], AttTables=list(DFR1, DFR2, DFR3), KFKDs=thiskfkds)
      )
      pt = proc.time();
      if(input$mlalgo == "lr") {
        mylr = LogRegDenseMultLearn(multtrain, lambda = input$lambdal2, alpha = input$alpha, eps = 0.001, maxiters = input$maxiters)
        predvectr = LogRegDenseMultScore(mylr, multtrain)
        if(input$checkcv) {
          predvects = LogRegDenseMultScore(mylr, multtest);
        }
      }
      else if(input$mlalgo == "nb") {
        mynb = NBayesMultLearn(multtrain)
        predvectr = NBayesMultScore(mynb, multtrain)
        if(input$checkcv) {
          predvects = NBayesMultScore(mynb, multtest);
        }
      }
      else if(input$mlalgo == "tan") {
        #TAN datasets are different since fks have to be made factors
        multtrain = switch(input$dataset,
                           "Walmart" = MultData(Target=as.data.frame(TWStr[,1]), EntTable=TWStr[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                           "Yelp" = MultData(Target=as.data.frame(TYStr[,2]), EntTable=TYStr[,-2], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                           "Expedia" = MultData(Target=as.data.frame(TEStr[,1]), EntTable=TEStr[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                           "Flights" = MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds)
        )
        mytan = TANBayesMultLearn(multtrain)
        predvectr = TANBayesMultScore(mytan, multtrain)
        if(input$checkcv) {
          multtest = switch(input$dataset,
                             "Walmart" = MultData(Target=as.data.frame(TWSts[,1]), EntTable=TWSts[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                             "Yelp" = MultData(Target=as.data.frame(TYSts[,2]), EntTable=TYSts[,-2], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                             "Expedia" = MultData(Target=as.data.frame(TESts[,1]), EntTable=TESts[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                             "Flights" = MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds)
          )
          predvects = TANBayesMultScore(mytan, multtest);
        }
      }#end if on ml algo
      tim = tim + proc.time() - pt;
      yvectr = multtrain@Target;
      yvects = multtest@Target;
    }
    else if(input$dbterm == "Single table") {
      thisfds = append(thisfds, FD(LHS=input$lhs1, RHS=input$rhs1, UseLHS=input$usefk1))
      if(!is.null(input$lhs2)) {
        thisfds = append(thisfds, FD(LHS=input$lhs2, RHS=input$rhs2, UseLHS=input$usefk2))
      }
      if(!is.null(input$lhs3)) {
        thisfds = append(thisfds, FD(LHS=input$lhs3, RHS=input$rhs3, UseLHS=input$usefk3))
      }
      cat("FDs:\n")
      print(thisfds)
      singtrain = switch(input$dataset,
                         "Walmart" = SingData(Target=as.data.frame(Wtr[,1]), TargetName="weekly_sales",
                                              SingTable=Wtr[,setdiff(names(Wtr),c("weekly_sales"))], FDs=thisfds),
                         "Walmart (R)" = SingData(Target=as.data.frame(DWtr[,1]), TargetName="weekly_sales",
                                                  SingTable=DWtr[,setdiff(names(DWtr),c("weekly_sales", input$lhs1, input$lhs2))], FDs=thisfds),
                         "Yelp" = SingData(Target=as.data.frame(Ytr[,1]), TargetName="stars",
                                              SingTable=as.data.frame(Ytr[,setdiff(names(Ytr),c("stars"))]), FDs=thisfds),
                         "Yelp (R)" = SingData(Target=as.data.frame(DYtr[,1]), TargetName="stars",
                                                  SingTable=as.data.frame(DYtr[,setdiff(names(DYtr),c("stars", input$lhs1, input$lhs2))]), FDs=thisfds),
                         "Expedia" = SingData(Target=as.data.frame(Etr[,1]), TargetName="position",
                                              SingTable=as.data.frame(Etr[,setdiff(names(Etr),c("position","srch_idfk"))]), FDs=thisfds),
                         "Expedia (R)" = SingData(Target=as.data.frame(DEtr[,1]), TargetName="position",
                                              SingTable=as.data.frame(DEtr[,setdiff(names(DEtr),c("position", input$lhs1, input$lhs2))]), FDs=thisfds),
                         "Flights" = SingData(Target=as.data.frame(Ftr[,1]), TargetName="codeshare",
                                              SingTable=as.data.frame(Ftr[,setdiff(names(Ftr),c("codeshare"))]), FDs=thisfds),
                         "Flights (R)" = SingData(Target=as.data.frame(DFtr[,1]), TargetName="codeshare",
                                                  SingTable=as.data.frame(DFtr[,setdiff(names(DFtr),c("codeshare", input$lhs1, input$lhs2, input$lhs3))]), FDs=thisfds)
      )
      singtest = switch(input$dataset,
                        "Walmart" = SingData(Target=as.data.frame(Wts[,1]), TargetName="weekly_sales",
                                             SingTable=Wts[,setdiff(names(Wts),c("weekly_sales"))], FDs=thisfds),
                        "Walmart (R)" = SingData(Target=as.data.frame(DWts[,1]), TargetName="weekly_sales",
                                                 SingTable=DWts[,setdiff(names(DWts),c("weekly_sales", input$lhs1, input$lhs2))], FDs=thisfds),
                        "Yelp" = SingData(Target=as.data.frame(Yts[,1]), TargetName="stars",
                                             SingTable=as.data.frame(Yts[,setdiff(names(Yts),c("stars"))]), FDs=thisfds),
                        "Yelp (R)" = SingData(Target=as.data.frame(DYts[,1]), TargetName="stars",
                                                 SingTable=as.data.frame(DYts[,setdiff(names(DYts),c("stars", input$lhs1, input$lhs2))]), FDs=thisfds),
                        "Expedia" = SingData(Target=as.data.frame(Ets[,1]), TargetName="position",
                                             SingTable=as.data.frame(Ets[,setdiff(names(Ets),c("position","srch_idfk"))]), FDs=thisfds),
                        "Expedia (R)" = SingData(Target=as.data.frame(DEts[,1]), TargetName="position",
                                                 SingTable=as.data.frame(DEts[,setdiff(names(DEts),c("position", input$lhs1, input$lhs2))]), FDs=thisfds),
                        "Flights" = SingData(Target=as.data.frame(Fts[,1]), TargetName="codeshare",
                                             SingTable=as.data.frame(Fts[,setdiff(names(Fts),c("codeshare"))]), FDs=thisfds),
                        "Flights (R)" = SingData(Target=as.data.frame(DFts[,1]), TargetName="codeshare",
                                                 SingTable=as.data.frame(DFts[,setdiff(names(DFts),c("codeshare", input$lhs1, input$lhs2, input$lhs3))]), FDs=thisfds)
      )
      pt = proc.time();
      if(input$mlalgo == "lr") {
        mylr = LogRegDenseSingLearn(singtrain, lambda = input$lambdal2, alpha = input$alpha, eps = 0.001, maxiters = input$maxiters)
        predvectr = LogRegDenseSingScore(mylr, singtrain)
        if(input$checkcv) {
          predvects = LogRegDenseSingScore(mylr, singtest);
        }
      }
      else if(input$mlalgo == "nb") {
        mynb = NBayesSingLearn(singtrain)
        predvectr = NBayesSingScore(mynb, singtrain)
        if(input$checkcv) {
          predvects = NBayesSingScore(mynb, singtest);
        }
      }
      else if(input$mlalgo == "tan") {
        mytan = TANBayesSingLearn(singtrain)
        predvectr = TANBayesSingScore(mytan, singtrain)
        if(input$checkcv) {
          predvects = TANBayesSingScore(mytan, singtest);
        }
      }#end if on ml algo
      tim = tim + proc.time() - pt;
      yvectr = singtrain@Target;
      yvects = singtest@Target;
    }#end if on Multi table
    tabtr = table(data.matrix(yvectr), data.matrix(predvectr))
    if(input$mlalgo != "lr") {
      tabtr = table(yvectr[,1], as.character(predvectr))
    }
    cat("\nTrain Accuracy:", sum(diag(tabtr))/sum(tabtr), "\n");
    cat("Train Confusion Matrix:\n")
    print(tabtr)
    #cat("Model:\n")
    #print(mylogregsing)
    if(input$checkcv) {
      tabts = table(data.matrix(yvects), data.matrix(predvects))
      if(input$mlalgo != "lr") {
        tabts = table(yvects[,1], as.character(predvects))
      }
      cat("\nTest Accuracy:", sum(diag(tabts))/sum(tabts), "\n");
      cat("Test Confusion Matrix:\n")
      print(tabts)
    }
    cat("\nTotal runtime", tim[1], "seconds\n")
    #confusionMatrix(data.matrix(singtrain@Target), tpredsingtr)
  })

  output$trainreso <- renderPrint({
    trainres()
  })

  feplots <- eventReactive(input$buttonfe, {
    names = c("All", "NoR1")
    thisfds = list(); singtrain = NULL; singtest = NULL
    thiskfkds = list(); multtrain = NULL; multtest = NULL
    tim = 0.0
    if(input$dbterm == "Multi table") {
      thiskfkds = append(thiskfkds, KFKD(EntCol=input$fk1, AttCol=input$pk1, UseFK=input$usefk1))
      if(!is.null(input$fk2)) {
        thiskfkds = append(thiskfkds, KFKD(EntCol=input$fk2, AttCol=input$pk2, UseFK=input$usefk2))
        names = c("All", "NoR1", "NoR2", "NoR1R2")
      }
      if(!is.null(input$fk3)) {
        thiskfkds = append(thiskfkds, KFKD(EntCol=input$fk3, AttCol=input$pk3, UseFK=input$usefk3))
        names = c("All", "NoR1", "NoR2", "NoR3", "NoR1R2", "NoR1R3", "NoR2R3", "NoR1R2R3")
      }
      #create the alternative multdatas, with last one being singdata (s alone)
      altdatastr = switch(input$dataset,
                         "Walmart" = list(
                           MultData(Target=as.data.frame(WStr[,1]), EntTable=WStr[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(WStrnor1[,1]), EntTable=WStrnor1[,-1], AttTables=list(WR2), KFKDs=thiskfkds[2]),
                           MultData(Target=as.data.frame(WStrnor2[,1]), EntTable=WStrnor2[,-1], AttTables=list(WR1), KFKDs=thiskfkds[1]),
                           SingData(Target=as.data.frame(Wtr[,1]), TargetName="weekly_sales", SingTable=Wtr[,setdiff(names(WStr), "weekly_sales")], FDs=list())
                         ),
                         "Walmart (R)" = list(
                           MultData(Target=as.data.frame(DWStr[,1]), EntTable=DWStr[,-1], AttTables=list(DWR1, DWR2), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(DWStr[,1]), EntTable=DWStr[,setdiff(nDWS, c("weekly_sales", input$fk1))], 
                                    AttTables=list(DWR2), KFKDs=thiskfkds[2]),
                           MultData(Target=as.data.frame(DWStr[,1]), EntTable=DWStr[,setdiff(nDWS, c("weekly_sales", input$fk2))],
                                    AttTables=list(DWR1), KFKDs=thiskfkds[1]),
                           SingData(Target=as.data.frame(DWStr[,1]), TargetName="weekly_sales",
                                    SingTable=DWStr[,setdiff(nDWS, c("weekly_sales", input$fk1, input$fk2))], FDs=list())
                         ),
                         "Yelp" = list(
                           MultData(Target=as.data.frame(YStr[,1]), EntTable=YStr[,-1], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(YStrnor1[,1]), EntTable=YStrnor1[,-1], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(YStrnor2[,1]), EntTable=YStrnor2[,-1], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                           SingData(Target=as.data.frame(Ytr[,1]), TargetName="stars", SingTable=as.data.frame(Ytr[,c("useridfk", "businessidfk")]), FDs=list())
                         ),
                         "Yelp (R)" = list(
                           MultData(Target=as.data.frame(DYStr[,1]), EntTable=DYStr[,-1], AttTables=list(DYR1, DYR2), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(DYStr[,1]), EntTable=DYStr[,setdiff(nDYS, c("stars", input$fk1))],
                                    AttTables=list(DYR2), KFKDs=thiskfkds[2]),
                           MultData(Target=as.data.frame(DYStr[,1]), EntTable=DYStr[,setdiff(nDYS, c("stars", input$fk2))],
                                    AttTables=list(DYR1), KFKDs=thiskfkds[1]),
                           SingData(Target=as.data.frame(DYStr[,1]), TargetName="stars",
                                    SingTable=as.data.frame(DYStr[,setdiff(nDYS, c("stars", input$fk1, input$fk2))]), FDs=list())
                         ),
                         "Expedia" = list(
                           MultData(Target=as.data.frame(EStr[,1]), EntTable=EStr[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(EStrnor1[,1]), EntTable=EStrnor1[,-1], AttTables=list(ER2), KFKDs=thiskfkds[2]),
                           MultData(Target=as.data.frame(EStrnor2[,1]), EntTable=EStrnor2[,-1], AttTables=list(ER1), KFKDs=thiskfkds[1]),
                           SingData(Target=as.data.frame(Etr[,1]), TargetName="position", 
                                    SingTable=as.data.frame(Etr[,setdiff(names(EStr),c("position","srch_idfk"))]), FDs=list())
                         ),
                         "Expedia (R)" = list(
                           MultData(Target=as.data.frame(DEStr[,1]), EntTable=DEStr[,-1], AttTables=list(DER1, DER2), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(DEStr[,1]), EntTable=DEStr[,setdiff(nDES, c("position", input$fk1))],
                                    AttTables=list(DER2), KFKDs=thiskfkds[2]),
                           MultData(Target=as.data.frame(DEStr[,1]), EntTable=DEStr[,setdiff(nDES, c("position", input$fk2))],
                                    AttTables=list(DER1), KFKDs=thiskfkds[1]),
                           SingData(Target=as.data.frame(DEStr[,1]), TargetName="position",
                                    SingTable=as.data.frame(DEStr[,setdiff(nDES, c("position", input$fk1, input$fk2))]), FDs=list())
                         ),
                         "Flights" = list(
                           MultData(Target=as.data.frame(FStr[,1]), EntTable=FStr[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(FStrnor1[,1]), EntTable=FStrnor1[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                           MultData(Target=as.data.frame(FStrnor1[,1]), EntTable=FStrnor1[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                           MultData(Target=as.data.frame(FStrnor1[,1]), EntTable=FStrnor1[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                           MultData(Target=as.data.frame(FStrnor1r2[,1]), EntTable=FStrnor1r2[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                           MultData(Target=as.data.frame(FStrnor1r2[,1]), EntTable=FStrnor1r2[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                           MultData(Target=as.data.frame(FStrnor1r2[,1]), EntTable=FStrnor1r2[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                           SingData(Target=as.data.frame(Ftr[,1]), TargetName="codeshare", SingTable=as.data.frame(Ftr[,c(setdiff(names(FStr), "codeshare"))]), FDs=list())
                         ),
                         "Flights (R)" = list(
                           MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,-1], AttTables=list(DFR1, DFR2, DFR3), KFKDs=thiskfkds),
                           MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,setdiff(nDFS, c("codeshare", input$fk1))], 
                                    AttTables=list(DFR2, DFR3), KFKDs=thiskfkds[-1]),
                           MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,setdiff(nDFS, c("codeshare", input$fk2))], 
                                    AttTables=list(DFR1, DFR3), KFKDs=thiskfkds[-2]),
                           MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,setdiff(nDFS, c("codeshare", input$fk3))], 
                                    AttTables=list(DFR1, DFR2), KFKDs=thiskfkds[-3]),
                           MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,setdiff(nDFS, c("codeshare", input$fk1, input$fk2))],
                                    AttTables=list(DFR3), KFKDs=thiskfkds[3]),
                           MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,setdiff(nDFS, c("codeshare", input$fk1, input$fk3))], 
                                    AttTables=list(DFR2), KFKDs=thiskfkds[2]),
                           MultData(Target=as.data.frame(DFStr[,1]), EntTable=DFStr[,setdiff(nDFS, c("codeshare", input$fk2, input$fk3))], 
                                    AttTables=list(DFR1), KFKDs=thiskfkds[1]),
                           SingData(Target=as.data.frame(DFStr[,1]), TargetName="codeshare", 
                                    SingTable=as.data.frame(DFStr[,setdiff(nDFS, c("codeshare", input$fk1, input$fk2, input$fk3))]), FDs=list())
                         )
      )
      altdatasts = NULL
      if(input$checkcv) {
        altdatasts = switch(input$dataset,
                            "Walmart" = list(
                              MultData(Target=as.data.frame(WSts[,1]), EntTable=WSts[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(WStsnor1[,1]), EntTable=WStsnor1[,-1], AttTables=list(WR2), KFKDs=thiskfkds[2]),
                              MultData(Target=as.data.frame(WStsnor1[,1]), EntTable=WStsnor2[,-1], AttTables=list(WR1), KFKDs=thiskfkds[1]),
                              SingData(Target=as.data.frame(Wts[,1]), TargetName="weekly_sales", SingTable=Wts[,setdiff(names(WSts), "weekly_sales")], FDs=list())
                            ),
                            "Walmart (R)" = list(
                              MultData(Target=as.data.frame(DWSts[,1]), EntTable=DWSts[,-1], AttTables=list(DWR1, DWR2), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(DWSts[,1]), EntTable=DWSts[,setdiff(nDWS, c("weekly_sales", input$fk1, nDWR1))],
                                       AttTables=list(DWR2), KFKDs=thiskfkds[2]),
                              MultData(Target=as.data.frame(DWSts[,1]), EntTable=DWSts[,setdiff(nDWS, c("weekly_sales", input$fk2, nDWR2))],
                                       AttTables=list(DWR1), KFKDs=thiskfkds[1]),
                              SingData(Target=as.data.frame(DWSts[,1]), TargetName="weekly_sales",
                                       SingTable=DWSts[,setdiff(nDWS, c("weekly_sales", input$fk1, input$fk2))], FDs=list())
                            ),
                            "Yelp" = list(
                              MultData(Target=as.data.frame(YSts[,1]), EntTable=YSts[,-1], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(YStsnor1[,1]), EntTable=YStsnor1[,-1], AttTables=list(YR2), KFKDs=thiskfkds[2]),
                              MultData(Target=as.data.frame(YStsnor2[,1]), EntTable=YStsnor2[,-1], AttTables=list(YR1), KFKDs=thiskfkds[1]),
                              SingData(Target=as.data.frame(Yts$stars), TargetName="stars", SingTable=as.data.frame(Yts[,c("useridfk", "businessidfk")]), FDs=list())
                            ),
                            "Yelp (R)" = list(
                              MultData(Target=as.data.frame(DYSts[,1]), EntTable=DYSts[,-1], AttTables=list(DYR1, DYR2), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(DYSts[,1]), EntTable=DYSts[,setdiff(nDYS, c("stars", input$fk1, nDYR1))],
                                       AttTables=list(DYR2), KFKDs=thiskfkds[2]),
                              MultData(Target=as.data.frame(DYSts[,1]), EntTable=DYSts[,setdiff(nDYS, c("stars", input$fk2, nDYR2))],
                                       AttTables=list(DYR1), KFKDs=thiskfkds[1]),
                              SingData(Target=as.data.frame(DYSts[,1]), TargetName="stars",
                                       SingTable=as.data.frame(DYSts[,setdiff(nDYS, c("stars", input$fk1, input$fk2))]), FDs=list())
                            ),
                            "Expedia" = list(
                              MultData(Target=as.data.frame(ESts[,1]), EntTable=ESts[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(EStsnor1[,1]), EntTable=EStsnor1[,-1], AttTables=list(ER2), KFKDs=thiskfkds[2]),
                              MultData(Target=as.data.frame(EStsnor2[,1]), EntTable=EStsnor2[,-1], AttTables=list(ER1), KFKDs=thiskfkds[1]),
                              SingData(Target=as.data.frame(Ets[,1]), TargetName="position", 
                                       SingTable=as.data.frame(Ets[,setdiff(names(ESts),c("position","srch_idfk"))]), FDs=list())
                            ),
                            "Expedia (R)" = list(
                              MultData(Target=as.data.frame(DESts[,1]), EntTable=DESts[,-1], AttTables=list(DER1, DER2), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(DESts[,1]), EntTable=DESts[,setdiff(nDES, c("position", input$fk1, nDER1))],
                                       AttTables=list(DER2), KFKDs=thiskfkds[2]),
                              MultData(Target=as.data.frame(DESts[,1]), EntTable=DESts[,setdiff(nDES, c("position", input$fk2, nDER2))],
                                       AttTables=list(DER1), KFKDs=thiskfkds[1]),
                              SingData(Target=as.data.frame(DESts[,1]), TargetName="position",
                                       SingTable=as.data.frame(DESts[,setdiff(nDES, c("position", input$fk1, input$fk2))]), FDs=list())
                            ),
                            "Flights" = list(
                              MultData(Target=as.data.frame(FSts[,1]), EntTable=FSts[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(FStsnor1[,1]), EntTable=FStsnor1[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                              MultData(Target=as.data.frame(FStsnor1[,1]), EntTable=FStsnor1[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                              MultData(Target=as.data.frame(FStsnor1[,1]), EntTable=FStsnor1[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                              MultData(Target=as.data.frame(FStsnor1r2[,1]), EntTable=FStsnor1r2[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                              MultData(Target=as.data.frame(FStsnor1r2[,1]), EntTable=FStsnor1r2[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                              MultData(Target=as.data.frame(FStsnor1r2[,1]), EntTable=FStsnor1r2[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                              SingData(Target=as.data.frame(Fts[,1]), TargetName="codeshare", SingTable=as.data.frame(Fts[,c(setdiff(names(FSts), "codeshare"))]), FDs=list())
                            ),
                            "Flights (R)" = list(
                              MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,-1], AttTables=list(DFR1, DFR2, DFR3), KFKDs=thiskfkds),
                              MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,setdiff(nDFS, c("codeshare", input$fk1))], 
                                       AttTables=list(DFR2, DFR3), KFKDs=thiskfkds[-1]),
                              MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,setdiff(nDFS, c("codeshare", input$fk2))], 
                                       AttTables=list(DFR1, DFR3), KFKDs=thiskfkds[-2]),
                              MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,setdiff(nDFS, c("codeshare", input$fk3))], 
                                       AttTables=list(DFR1, DFR2), KFKDs=thiskfkds[-3]),
                              MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,setdiff(nDFS, c("codeshare", input$fk1, input$fk2))],
                                       AttTables=list(DFR3), KFKDs=thiskfkds[3]),
                              MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,setdiff(nDFS, c("codeshare", input$fk1, input$fk3))], 
                                       AttTables=list(DFR2), KFKDs=thiskfkds[2]),
                              MultData(Target=as.data.frame(DFSts[,1]), EntTable=DFSts[,setdiff(nDFS, c("codeshare", input$fk2, input$fk3))], 
                                       AttTables=list(DFR1), KFKDs=thiskfkds[1]),
                              SingData(Target=as.data.frame(DFSts[,1]), TargetName="codeshare", 
                                       SingTable=as.data.frame(DFSts[,setdiff(nDFS, c("codeshare", input$fk1, input$fk2, input$fk3))]), FDs=list())
                            )
        )
      }
      #learn and score for each altdata; get times and accuracies alone
      lals = length(altdatastr)
      times = c()
      accuraciestr = c()
      accuraciests = c()
      for(i in 1:lals) {
        predvectr = NULL; predvects = NULL;
        pt = proc.time();
        if(input$mlalgo == "lr") {
          mylogreg = NULL;
          if(i < lals) {
            mylogreg = LogRegDenseMultLearn(altdatastr[[i]], lambda = input$lambdal2, alpha = input$alpha, eps = 0.001, maxiters = input$maxiters) #param grid search? TODO
            predvectr = LogRegDenseMultScore(mylogreg, altdatastr[[i]])
            if(input$checkcv) {
              predvects = LogRegDenseMultScore(mylogreg, altdatasts[[i]]);
            }
          }
          else {
            mylogreg = LogRegDenseSingLearn(altdatastr[[i]], lambda = input$lambdal2, alpha = input$alpha, eps = 0.001, maxiters = input$maxiters) #param grid search?
            predvectr = LogRegDenseSingScore(mylogreg, altdatastr[[i]])
            if(input$checkcv) {
              predvects = LogRegDenseSingScore(mylogreg, altdatasts[[i]]);
            }
          }
        }
        else if(input$mlalgo == "nb") {
          mynb = NULL;
          if(i < lals) {
            mynb = NBayesMultLearn(altdatastr[[i]])
            predvectr = NBayesMultScore(mynb, altdatastr[[i]])
            if(input$checkcv) {
              predvects = NBayesMultScore(mynb, altdatasts[[i]]);
            }
          }
          else {
            mynb = NBayesSingLearn(altdatastr[[i]])
            predvectr = NBayesSingScore(mynb, altdatastr[[i]])
            if(input$checkcv) {
              predvects = NBayesSingScore(mynb, altdatasts[[i]]);
            }
          }
        }
        else if(input$mlalgo == "tan") {
          altdatastr = switch(input$dataset,
                              "Walmart" = list(
                                MultData(Target=as.data.frame(TWStr[,1]), EntTable=TWStr[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                                MultData(Target=as.data.frame(TWStr[,1]), EntTable=TWStr[,-1], AttTables=list(WR2), KFKDs=thiskfkds[2]),
                                MultData(Target=as.data.frame(TWStr[,1]), EntTable=TWStr[,-1], AttTables=list(WR1), KFKDs=thiskfkds[1]),
                                SingData(Target=as.data.frame(TWStr[,1]), TargetName="weekly_sales", SingTable=TWStr[,-1], FDs=list())
                              ),
                              "Yelp" = list(
                                MultData(Target=as.data.frame(TYStr[,2]), EntTable=TYStr[,-2], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                                MultData(Target=as.data.frame(TYStr[,2]), EntTable=TYStr[,-2], AttTables=list(YR2), KFKDs=thiskfkds[2]),
                                MultData(Target=as.data.frame(TYStr[,2]), EntTable=TYStr[,-2], AttTables=list(YR1), KFKDs=thiskfkds[1]),
                                SingData(Target=as.data.frame(TYStr[,2]), TargetName="stars", SingTable=as.data.frame(TYStr[,-2]), FDs=list())
                              ),
                              "Expedia" = list(
                                MultData(Target=as.data.frame(TEStr[,1]), EntTable=TEStr[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                                MultData(Target=as.data.frame(TEStr[,1]), EntTable=TEStr[,-1], AttTables=list(ER2), KFKDs=thiskfkds[2]),
                                MultData(Target=as.data.frame(TEStr[,1]), EntTable=TEStr[,-1], AttTables=list(ER1), KFKDs=thiskfkds[1]),
                                SingData(Target=as.data.frame(TEStr[,1]), TargetName="position", 
                                         SingTable=as.data.frame(TEStr[,setdiff(names(TEStr),c("position","srch_idfk"))]), FDs=list())
                              ),
                              "Flights" = list(
                                MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds),
                                MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                                MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR1, FR3), KFKDs=thiskfkds[-2]),
                                MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR1, FR2), KFKDs=thiskfkds[-3]),
                                MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                                MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR2), KFKDs=thiskfkds[2]),
                                MultData(Target=as.data.frame(TFStr[,1]), EntTable=TFStr[,-1], AttTables=list(FR1), KFKDs=thiskfkds[1]),
                                SingData(Target=as.data.frame(TFStr[,1]), TargetName="codeshare", SingTable=as.data.frame(TFStr[,-1]), FDs=list())
                              )
          )
          mytan = NULL;
          if(i < lals) {
            mytan = TANBayesMultLearn(altdatastr[[i]])
            predvectr = TANBayesMultScore(mytan, altdatastr[[i]])
            if(input$checkcv) {
              altdatasts = switch(input$dataset,
                                  "Walmart" = list(
                                    MultData(Target=as.data.frame(TWSts[,1]), EntTable=TWSts[,-1], AttTables=list(WR1, WR2), KFKDs=thiskfkds),
                                    MultData(Target=as.data.frame(TWSts[,1]), EntTable=TWSts[,-1], AttTables=list(WR2), KFKDs=thiskfkds[2]),
                                    MultData(Target=as.data.frame(TWSts[,1]), EntTable=TWSts[,-1], AttTables=list(WR1), KFKDs=thiskfkds[1]),
                                    SingData(Target=as.data.frame(TWSts[,1]), TargetName="weekly_sales", SingTable=TWSts[,-1], FDs=list())
                                  ),
                                  "Expedia" = list(
                                    MultData(Target=as.data.frame(TESts[,1]), EntTable=TESts[,-1], AttTables=list(ER1, ER2), KFKDs=thiskfkds),
                                    MultData(Target=as.data.frame(TESts[,1]), EntTable=TESts[,-1], AttTables=list(ER2), KFKDs=thiskfkds[2]),
                                    MultData(Target=as.data.frame(TESts[,1]), EntTable=TESts[,-1], AttTables=list(ER1), KFKDs=thiskfkds[1]),
                                    SingData(Target=as.data.frame(TESts[,1]), TargetName="position", 
                                             SingTable=as.data.frame(TESts[,setdiff(names(TESts),c("position","srch_idfk"))]), FDs=list())
                                  ),
                                  "Yelp" = list(
                                    MultData(Target=as.data.frame(TYSts[,2]), EntTable=TYSts[,-2], AttTables=list(YR1, YR2), KFKDs=thiskfkds),
                                    MultData(Target=as.data.frame(TYSts[,2]), EntTable=TYSts[,-2], AttTables=list(YR2), KFKDs=thiskfkds[2]),
                                    MultData(Target=as.data.frame(TYSts[,2]), EntTable=TYSts[,-2], AttTables=list(YR1), KFKDs=thiskfkds[1]),
                                    SingData(Target=as.data.frame(TYSts[,2]), TargetName="stars", SingTable=as.data.frame(TYSts[,-2]), FDs=list())
                                  ),
                                  "Flights" = list(
                                    MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR1, FR2, FR3), KFKDs=thiskfkds),
                                    MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR2, FR3), KFKDs=thiskfkds[-1]),
                                    MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR1, FR3), KFKDs=thiskfkds[-2]),
                                    MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR1, FR2), KFKDs=thiskfkds[-3]),
                                    MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR3), KFKDs=thiskfkds[3]),
                                    MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR2), KFKDs=thiskfkds[2]),
                                    MultData(Target=as.data.frame(TFSts[,1]), EntTable=TFSts[,-1], AttTables=list(FR1), KFKDs=thiskfkds[1]),
                                    SingData(Target=as.data.frame(TFSts[,1]), TargetName="codeshare", SingTable=as.data.frame(TFSts[,-1]), FDs=list())
                                  ))
              predvects = TANBayesMultScore(mytan, altdatasts[[i]]);
            }
          }
          else {
            mytan = TANBayesSingLearn(altdatastr[[i]])
            predvectr = TANBayesSingScore(mytan, altdatastr[[i]])
            if(input$checkcv) {
              predvects = TANBayesSingScore(mytan, altdatasts[[i]]);
            }
          }
        }#end if on ml algo
        tabtr = table(data.matrix(altdatastr[[i]]@Target), data.matrix(predvectr))
        if(input$mlalgo != "lr") {
          tabtr = table(altdatastr[[i]]@Target[,1], as.character(predvectr))
        }
        accuraciestr = c(accuraciestr, 100 * sum(diag(tabtr))/sum(tabtr))
        if(input$checkcv) {
          tabts = table(data.matrix(altdatasts[[i]]@Target), data.matrix(predvects))
          if(input$mlalgo != "lr") {
            tabts = table(altdatasts[[i]]@Target[,1], as.character(predvects))
          }
          accuraciests = c(accuraciests, 100 * sum(diag(tabts))/sum(tabts))
        }
        times = c(times, (proc.time() - pt)[1])
      }#end loop on all featvecs
    }#end if on Multi table
    
    #colors = c("#330000", "#990000", "#FF3333", "#FF6666")[1:lals]
    colors = c("red", "blue", "darkgreen", "darkgray", "pink", "lightblue", "lightgreen", "lightgray")[1:lals]
    pwidth = 0.5
    if(lals == 8) {
      pwidth = 0.4
    }
    times = ceiling(1000*times)/1000.0
    accuraciestr = ceiling(1000*accuraciestr)/1000.0
    if(!input$checkcv) {
      par(mar = c(2, 8, 2, 0), mfrow = c(2, 1)) #, pin=c(6, 1.8))
    }
    else {
      par(mar = c(2, 8, 2, 0), mfrow = c(3, 1)) #, pin=c(6, 1.8))
      accuraciests = ceiling(1000*accuraciests)/1000.0
    }
    
    bplt <- barplot(times, width = pwidth, xlim = c(0, 4), ylim = c(0, max(times)*1.2), xlab = "Feature Vectors", ylab = "Total Runtime (s)", 
                    col = colors, names.arg = names, cex = 1.5, cex.main = 1, cex.axis = 1.5, cex.lab = 1.5)
    text(x = bplt, y = times + max(times) * 0.07, labels = as.character(times), xpd = TRUE, cex = 1.4)
    
    bplt <- barplot(accuraciestr, width = pwidth, xlim = c(0, 4), ylim = c(0, 100), xlab = "Feature Vectors", ylab = "Train Accuracy (%)", 
                    col = colors, names.arg = names, beside = TRUE, cex = 1.5, cex.main = 1, cex.axis = 1.5, cex.lab = 1.5)
    text(x = bplt, y = accuraciestr + max(accuraciestr) * 0.1, labels = as.character(accuraciestr), xpd = TRUE, cex = 1.4)
    
    if(input$checkcv) {
      bplt <- barplot(accuraciests, width = pwidth, xlim = c(0, 4), ylim = c(0, 100), xlab = "Feature Vectors", ylab = "Test Accuracy (%)", 
                      col = colors, names.arg = names, beside = TRUE, cex = 1.5, cex.main = 1, cex.axis = 1.5, cex.lab = 1.5)
      text(x = bplt, y = accuraciests + max(accuraciests) * 0.1, labels = as.character(accuraciests), xpd = TRUE, cex = 1.4)
    }
    
  })
  
  output$feplotso <- renderPlot({
    feplots()
  })
  
})
