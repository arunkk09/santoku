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

shinyUI(fluidPage(
    list(tags$head(HTML('<h4><table><tr><td rowspan="2"><img src="http://umark.wisc.edu/brand/templates-and-downloads/downloads/print/UWCrest_4c.jpg" 
          border="0" style="padding-right:10px" width="34" height="40" alt="UW-Madison Database Group"/> 
          </td><td><b>Santoku</b></td></tr><tr><td>University of Wisconsin-Madison Database Group</td></tr></table></h4>'))),
    sidebarLayout(
      sidebarPanel(width = 6,
                   wellPanel(fluidRow(column(6, radioButtons("dbterm", "Database Type", c("Multi table", "Single table"))),
                                      column(6, selectInput("dataset", "Load Dataset", c("Walmart", "Walmart (R)", "Yelp", "Yelp (R)", "Expedia",
                                                                                         "Expedia (R)", "Flights", "Flights (R)")))),
                             uiOutput("uideps")),
                   wellPanel(fluidRow(column(6, radioButtons("mlalgo", "ML Model:", c("Logistic Regression" = "lr", "Naive Bayes" = "nb",
                                                                                          "TAN" = "tan"))),
                                      column(6, uiOutput("uimlpt"))),
                             fluidRow(div(class="padding2", column(3, checkboxInput("checkcv", "Validate", TRUE))),
                                      div(class="padding3", column(2, actionButton("buttontrain", "Learning"))),
                                      div(class="padding4", column(3, actionButton("buttonfe", "Feature Exploration")))))
                   ),
      mainPanel(width = 6,
                tabsetPanel(
                  tabPanel("Single Learning", verbatimTextOutput("trainreso")), 
                  tabPanel("Feature Exploration", plotOutput("feplotso"))
                  #tabPanel("Wiki", verbatimTextOutput("Wiki")),
                  #tabPanel("Analysis", tableOutput("plots"))
                  )
                )
      )#end sidebarLayout
))#end main
