library(tidyverse)
library(mice)
library(MASS)
library(glmnet)
#library(texreg)
library(randomForest)


X <- read.csv("train.csv",header = TRUE,na.strings = (c("NA","")),stringsAsFactors = TRUE)
Y <- read.csv("test.csv",header = TRUE,na.strings = (c("NA","")),stringsAsFactors = TRUE)


summary(X)
ColNameX <- colnames(X)
ColNameX

X2 <- X

summary(X2$Sex)
hist(X2$Age)
median(X2$Age,na.rm = TRUE)

X2$Sex <- 0
X2$Sex[X$Sex == "female"] <- 1
X2$Age[is.na(X$Age) == TRUE] <- median(X2$Age, na.rm = TRUE)

X2$Embarked <- 0
X2$Embarked[X$Embarked == "C"] <- 1
X2$Embarked[X$Embarked == "Q"] <- 2
X2$Embarked[X$Embarked == "S"] <- 3
X2$Embarked[X2$Embarked == 0] <-  NA
X2$Embarked[is.na(X2$Embarked) == TRUE] <- median(X2$Embarked,na.rm = TRUE)

X$CabinNum <- numeric(length(X$Cabin))
X$CabinNum[grep("A",X$Cabin)] <- 7
X$CabinNum[grep("B",X$Cabin)] <- 6
X$CabinNum[grep("C",X$Cabin)] <- 5
X$CabinNum[grep("D",X$Cabin)] <- 4
X$CabinNum[grep("E",X$Cabin)] <- 3
X$CabinNum[grep("F",X$Cabin)] <- 2
X$CabinNum[grep("G",X$Cabin)] <- 1
X$CabinNum[is.na(X$Cabin)==TRUE] <- NA
X$CabinNum[X$CabinNum==0] <- NA

X$Name2 <- numeric(length(X$Name))
X$Name2[grep("Master",X$Name)] <- "Master"
X$Name2[grep("Mrs",X$Name)] <- "Mr"
X$Name2[grep("Miss",X$Name)] <- "Miss"
X$Name2[grep("Mr",X$Name)] <- "Mr"
X$Name2[grep("Dr",X$Name)] <- "Dr"
X$Name2[grep("0",X$Name)] <- "Other"
