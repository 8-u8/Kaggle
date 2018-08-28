library(mice)
library(MASS)
library(glmnet)
#library(texreg)
library(randomForest)

#setwd("C:/Users/1U1_2/Desktop/Kaggle/Titanic")
getwd()

X <- read.csv("train.csv",header = TRUE,na.strings = (c("NA","")),stringsAsFactors = TRUE)
Y <- read.csv("test.csv",header = TRUE,na.strings = (c("NA","")),stringsAsFactors = TRUE)
summary(X)
X$SexNum <- numeric(length(X$Sex))
X$SexNum[X$Sex=="male"] <- 1
cor(X$Survived,X$SexNum)
hist(X$Survived[X$SexNum==1])
hist(X$Survived[X$SexNum==0])
summary(X)

X$PclassFactor <- as.factor(X$Pclass)
X$TicketClass <- as.numeric(X$Ticket)
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


X$Mrc <- numeric(length(X$SexNum))
X$Mrsc<- numeric(length(X$SexNum))
X$Missc<- numeric(length(X$SexNum))
X$Mastc<- numeric(length(X$SexNum))
X$Drc<- numeric(length(X$SexNum))

X$Pclass2<- numeric(length(X$SexNum))
X$Pclass3<- numeric(length(X$SexNum))


X$Mrc[X$Name2=="Mr"] <- 1
X$Mrsc[X$Name2=="Mrs"] <- 1
X$Missc[X$Name2=="Miss"] <- 1
X$Mastc[X$Name2=="Master"] <- 1
X$Drc[X$Name2=="Dr"] <- 1

X$AgeFix <- X$Age
X$AgeFix[is.na(X$Age)==TRUE & grep("Master",X$Name)] <- 
  median(X$Age[grep("Master",X$Name)],na.rm=TRUE)
X$AgeFix[is.na(X$Age)==TRUE & grep("Miss",X$Name)] <-  median(X$Age[grep("Miss",X$Name)],na.rm=TRUE)
X$AgeFix[is.na(X$Age)==TRUE & grep("Mrs",X$Name)] <-  median(X$Age[grep("Mrs",X$Name)],na.rm=TRUE)
X$AgeFix[is.na(X$Age)==TRUE & grep("Mr",X$Name)] <-  median(X$Age[grep("Mr",X$Name)],na.rm=TRUE)
X$AgeFix[is.na(X$Age)==TRUE & grep("Dr",X$Name)] <-  median(X$Age[grep("Dr",X$Name)],na.rm=TRUE)

library(glmnet)

fit0 <- rlm(Age~Name2+Sex+PclassFactor+SibSp+Parch,data=X,family = "gaussian")
#fit01 <- glmnet(X,X$Age,family="gaussian")

summary(fit0)

X$AgePred <- X$Age
X$AgePred[is.na(X$Age)==TRUE] <- fit0$coefficients[[1]]+
  X$Drc[is.na(X$Age)==TRUE]*fit0$coefficients[[2]]+
  X$Mastc[is.na(X$Age)==TRUE]*fit0$coefficients[[3]]+
  X$Missc[is.na(X$Age)==TRUE]*fit0$coefficients[[4]]+
  X$Mrc[is.na(X$Age)==TRUE]*fit0$coefficients[[5]]+
  X$SexNum[is.na(X$Age)==TRUE]*fit0$coefficients[[6]]+
  X$Pclass2[is.na(X$Age)==TRUE]*fit0$coefficients[[7]]+
  X$Pclass3[is.na(X$Age)==TRUE]*fit0$coefficients[[8]]+
  X$SibSp[is.na(X$Age)==TRUE]*fit0$coefficients[[9]]+
  X$Parch[is.na(X$Age)==TRUE]*fit0$coefficients[[10]]
  


summary(X$AgePred)

t.test(X$AgeFix,X$AgePred)


fit1 <- glm(Survived~AgeFix+Sex+Pclass+Fare+Parch+SibSp,data=X,family=binomial("logit"))
screenreg(list(fit1,fit2))

summary(X)

summary(Y)


Y$SexNum <- numeric(length(Y$Sex))
Y$SexNum[Y$Sex=="male"] <- 1

Y$PclassFactor <- as.factor(Y$Pclass)
Y$TicketClass <- as.numeric(Y$Ticket)
Y$CabinNum <- numeric(length(Y$Cabin))
Y$CabinNum[grep("A",Y$Cabin)] <- 7
Y$CabinNum[grep("B",Y$Cabin)] <- 6
Y$CabinNum[grep("C",Y$Cabin)] <- 5
Y$CabinNum[grep("D",Y$Cabin)] <- 4
Y$CabinNum[grep("E",Y$Cabin)] <- 3
Y$CabinNum[grep("F",Y$Cabin)] <- 2
Y$CabinNum[grep("G",Y$Cabin)] <- 1
Y$CabinNum[is.na(Y$Cabin)==TRUE] <- NA
Y$CabinNum[Y$CabinNum==0] <- NA

Y$Name2 <- numeric(length(Y$Name))
Y$Name2[grep("Master",Y$Name)] <- "Master"
Y$Name2[grep("Mrs",Y$Name)] <- "Mr"
Y$Name2[grep("Miss",Y$Name)] <- "Miss"
Y$Name2[grep("Mr",Y$Name)] <- "Mr"
Y$Name2[grep("Dr",Y$Name)] <- "Dr"
Y$Name2[grep("0",Y$Name)] <- NA


Y$Mrc <- numeric(length(Y$SexNum))
Y$Mrsc<- numeric(length(Y$SexNum))
Y$Missc<- numeric(length(Y$SexNum))
Y$Mastc<- numeric(length(Y$SexNum))
Y$Drc<- numeric(length(Y$SexNum))

Y$Pclass2<- numeric(length(Y$SexNum))
Y$Pclass3<- numeric(length(Y$SexNum))



Y$Mrc[Y$Name2=="Mr"] <- 1
Y$Mrsc[Y$Name2=="Mrs"] <- 1
Y$Missc[Y$Name2=="Miss"] <- 1
Y$Mastc[Y$Name2=="Master"] <- 1
Y$Drc[Y$Name2=="Dr"] <- 1

Y$AgeFix <- Y$Age
Y$AgeFix[is.na(Y$Age)==TRUE & grep("Master",Y$Name)] <- median(Y$Age[grep("Master",Y$Name)],na.rm=TRUE)
Y$AgeFix[is.na(Y$Age)==TRUE & grep("Miss",Y$Name)] <-  median(Y$Age[grep("Miss",Y$Name)],na.rm=TRUE)
Y$AgeFix[is.na(Y$Age)==TRUE & grep("Mrs",Y$Name)] <-  median(Y$Age[grep("Mrs",Y$Name)],na.rm=TRUE)
Y$AgeFix[is.na(Y$Age)==TRUE & grep("Mr",Y$Name)] <-  median(Y$Age[grep("Mr",Y$Name)],na.rm=TRUE)
Y$AgeFix[is.na(Y$Age)==TRUE & grep("Dr",Y$Name)] <-  median(Y$Age[grep("Dr",Y$Name)],na.rm=TRUE)



write.csv(submit,"Prediction2.csv")
summary(submit)
summary(Ans)


#ランダム・フォレストによる予測
#これが一番確定的で正確っぽい
#もう少し頑張ってみる

X$Name3 <- numeric(length(X$Name2))
X$Name3[grep("Master",X$Name)] <- 1
X$Name3[grep("Mrs",X$Name)] <- 2
X$Name3[grep("Miss",X$Name)] <- 2.2
X$Name3[grep("Mr",X$Name)] <- 3
X$Name3[grep("Dr",X$Name)] <- 4
X$Name3[grep("0",X$Name)] <- 0



fit3 <- randomForest(factor(Survived)~AgePred+Sex+Pclass+Parch+SibSp+Fare2+Embarked,data=X,na.action = na.omit)
fit3
summary(fit3)
Y$Survived <- predict(fit3,Y)
summary(Y$Survived)
summary(factor(Ans$Survived))
cbind(as.factor(Y$Survived),Ans$Survived)
