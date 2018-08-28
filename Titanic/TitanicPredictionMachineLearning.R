library(mice)
library(MASS)
library(glmnet)
library(texreg)
library(randomForest)

setwd("C:/Users/1U1_2/Desktop/Kaggle/Titanic")

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

cabinest <- glm(CabinNum~PclassFactor,data=X,family="poisson")
summary(cabinest)
X$CabinNum[is.na(X$CabinNum)==TRUE] <- cabinest$coefficients[[1]]+X$Pclass2[is.na(X$CabinNum)==TRUE]*cabinest$coefficients[[2]]+X$Pclass3[is.na(X$CabinNum)==TRUE]*cabinest$coefficients[[3]]
summary(X$CabinNum)


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
X$Pclass2[X$Pclass==2]<- 1
X$Pclass3[X$Pclass==3]<- 1


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

fit0 <- rlm(Age~Name2+Sex+PclassFactor+SibSp+Parch+CabinNum,data=X,family = "gaussian")
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
fit2 <- glm(Survived~AgePred+Sex+PclassFactor+Fare+Parch+SibSp+CabinNum,data=X,family=binomial("logit"))
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

fit00 <- glm(Age~Name2+Sex+PclassFactor+SibSp+Parch,data=Y,family = "gaussian")
summary(fit00)

Y$AgePred <- Y$Age
Y$AgePred[is.na(Y$Age)==TRUE] <- fit00$coefficients[[1]]+
  Y$Drc[is.na(Y$Age)==TRUE]*fit00$coefficients[[2]]+
  Y$Mastc[is.na(Y$Age)==TRUE]*fit00$coefficients[[3]]+
  Y$Missc[is.na(Y$Age)==TRUE]*fit00$coefficients[[4]]+
  Y$Mrc[is.na(Y$Age)==TRUE]*fit00$coefficients[[5]]+
  Y$SexNum[is.na(Y$Age)==TRUE]*fit00$coefficients[[6]]+
  Y$Pclass2[is.na(Y$Age)==TRUE]*fit00$coefficients[[7]]+
  Y$Pclass3[is.na(Y$Age)==TRUE]*fit00$coefficients[[8]]+
  Y$SibSp[is.na(Y$Age)==TRUE]*fit00$coefficients[[9]]+
  Y$Parch[is.na(Y$Age)==TRUE]*fit00$coefficients[[10]]
summary(Y$AgePred)

ODDS <- fit2$coefficients[[1]]+
  Y$AgePred*fit2$coefficients[[2]]+
  Y$SexNum*fit2$coefficients[[3]]+
  Y$Pclass2*fit2$coefficients[[4]]+
  Y$Pclass3*fit2$coefficients[[5]]+
  #Y$Fare*fit2$coefficients[[6]]+
  Y$Parch*fit2$coefficients[[7]]+
  Y$SibSp*fit2$coefficients[[8]]
length(ODDS)

summary(ODDS)

P <- exp(ODDS)/(1+exp(ODDS))

Y$Survived <- rbinom(length(P),c(0,1),P)
Y$Survived[P>0.5] <- 1

summary(Y$Survived)

submit <- data.frame(PassengerId = Y$PassengerId,Survived=Y$Survived)
submit
write.csv(submit,"Prediction3.csv",row.names = FALSE)
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

fit4 <- randomForest(CabinNum~AgePred+Sex+Pclass+Parch+SibSp+Fare2+Embarked,data=X,na.action = na.omit)
Y$CabinNum <- predict(fit4,Y)
fit3 <- randomForest(factor(Survived)~AgePred+Sex+Pclass+Parch+SibSp+CabinNum+Fare2+Embarked,data=X,na.action = na.omit)
fit3
summary(fit3)
Y$Survived <- predict(fit3,Y)
summary(Y$SurvivedNum)
summary(factor(Ans$Survived))
cbind(Y$SurvivedNum,Ans$Survived,(Y$SurvivedNum-Ans$Survived))

Y$SurvivedNum <- as.numeric(Y$Survived)
Y$SurvivedNum[Y$SurvivedNum==1] <- 0
Y$SurvivedNum[Y$SurvivedNum==2] <- 1
