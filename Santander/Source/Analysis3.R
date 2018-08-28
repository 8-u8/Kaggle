#Load libraries
library(forecast)
library(tidyverse)
library(data.table)
library(corrplot)
library(Metrics)
library(randomForest)
library(MASS)
library(rfUtilities)
library(caret)
library(xgboost)

TrainData <- read.table(file = 'train.csv', header = TRUE,sep = ",")
TestData <- read.table(file = 'test.csv', header = TRUE,sep = ",")
result <- read.table(file = "final.csv",sep = ",",header = TRUE)

FEATURE <- c("target",
             'f190486d6', 
             'X58e2e02e6', 
             'eeb9cd3aa', 
             'X9fd594eec', 
             'X6eef030c1', 
             'X15ace8c9f', 
             'fb0f5dbfe', 
             'X58e056e12', 
             'X20aa07010', 
             'X024c577b9', 
             'd6bb78916', 
             'b43a7cfd5', 
             'X58232a6fb', 
             'X1702b5bf0', 
             'X324921c7b', 
             'X62e59a501', 
             'X2ec5b290f', 
             'X241f0f867', 
             'fb49e4212', 
             'X66ace2992', 
             'f74e8f13d', 
             'X5c6487af1', 
             'X963a49cdc', 
             'X26fc93eb7', 
             'X1931ccfdd', 
             'X703885424', 
             'X70feb1494', 
             'X491b9ee45', 
             'X23310aa6f', 
             'e176a204a', 
             'X6619d81fc', 
             'X1db387535',
             'fc99f9426',
             'X91f701ba2',
             'X0572565c2',
             'X190db8488',
             'adb64ff71',
             'c47340d97',
             'c5a231d81',
             'X0ff32eb98'
             
)

TrainData <- TrainData[,c("ID",FEATURE)]
TestData <- TestData[,c("ID",FEATURE[-1])]
TestData$target <- result$target

TrainData$EZ     <- apply(TrainData[,3:42],1,function(x)length(x[x==0]))
TrainData$Mean   <- apply(TrainData[,3:42],1,function(x)mean(x,na.rm = TRUE))
TrainData$Max    <- apply(TrainData[,3:42],1,function(x)max(x))
TrainData$Min    <- apply(TrainData[,3:42],1,function(x)min(x))
TrainData$Gap    <- apply(TrainData[,3:42],1,function(x)exp(mean(log(x),na.rm = TRUE)))
#TrainData$NZMean <- apply(TrainData[,3:42],1,function(x)mean(x[x!=0],na.rm = TRUE))
#TrainData$NZMax  <- apply(TrainData[,3:42],1,function(x)max(x[x!=0]))
#TrainData$NZMin  <- apply(TrainData[,3:42],1,function(x)min(x[x!=0]))
#TrainData$NZMean[is.na(TrainData$NZMean)==TRUE] <- median(TrainData$Mean)
#TrainData$NZMax[is.na(TrainData$NZMax)==TRUE]   <- median(TrainData$Mix)
#TrainData$NZMin[is.na(TrainData$NZMin)==TRUE]   <- median(TrainData$Min)

TestData$EZ      <- apply(TestData[,3:42],1,function(x)length(x[x==0]))
TestData$Mean    <- apply(TestData[,3:42],1,function(x)mean(x,na.rm = TRUE))
TestData$Max     <- apply(TestData[,3:42],1,function(x)max(x))
TestData$Min     <- apply(TestData[,3:42],1,function(x)min(x))
TestData$Gap     <- apply(TestData[,3:42],1,function(x)exp(mean(log(x),na.rm = TRUE)))
#TestData$NZMean  <- apply(TestData[,3:42],1,function(x)mean(x[x!=0],na.rm = TRUE))
#TestData$NZMax   <- apply(TestData[,3:42],1,function(x)max(x[x!=0]))
#TestData$NZMin   <- apply(TestData[,3:42],1,function(x)min(x[x!=0]))
#TestData$NZMean[is.na(TestData$NZMean)==TRUE] <- median(TestData$Mean)
#TestData$NZMax[is.na(TestData$NZMax)==TRUE]   <- median(TestData$Mix)
#TestData$NZMin[is.na(TestData$NZMin)==TRUE]   <- median(TestData$Min)

model3 <- randomForest(log1p(target) ~ ., 
                       data = TrainData[,-1],
                       mtry = 13,
                       replace = TRUE,
                       importance = TRUE,
                       type = "regression",
                       ntree = 500,
                       nodesize = 10)

hist(model3$predicted %>% expm1())
fit3Train <- predict(model3,newdata = TrainData, type = "response") %>% 
  expm1()

fit3Test <- predict(model3,newdata = TestData, type = "response") %>% 
  expm1()
rmsle(model3$predicted %>% expm1(),TrainData$target)
plot(TestData$target,fit3Test)

TrainData$Pred1 <- fit3Train
TestData$Pred1 <- fit3Test

rmsle(TestData$Pred1,result$target)


xgbGrid <- expand.grid(eta = .3, 
                       max_depth = 4,
                       nrounds = c(1:500),
                       gamma = 0,
                       colsample_bytree = 0.8,
                       min_child_weight = 1,
                       subsample = 0.875
)

Weights <- apply(TrainData[,-1],1,function(x)runif(x))

model6 <- caret::train(target~.,data = TrainData[,-1],
                       tuneLength = 5,
                       preProcess = c("center","scale"),
                       method = "xgbTree",
                       trControl = trainControl("cv",number = 20),
                       tuneGrid = xgbGrid,
                       weights = Weights
)


fit6Train <- predict(model6,newdata = TrainData) %>% 
  expm1()

fit6Test <- predict(model6,newdata = TestData)%>% 
  expm1()
plot(fit6Train,TrainData$target)
plot(fit6Test,TestData$target)


TrainData$Pred2 <- fit6Train
TestData$Pred2 <- fit6Test


summary(fit6)

model6b<- caret::train(log1p(target)~.,data = TrainData[,-1],
                       tuneLength = 50,
                       preProcess = c("center","scale"),
                       method = "xgbTree",
                       trControl = trainControl("cv",number = 1000),
                       tuneGrid = xgbGrid,
                       na.action = na.omit
                       
)

fit6b <- predict(model6b,newdata = TestData) %>% 
  expm1()
summary(fit6b)

fit6bTrain <- predict(model6b,newdata = TrainData) %>% expm1()
fit6bTest <- predict(model6b,newdata = TestData) %>% expm1()

#TrainData$Pred3 <- fit6bTrain
#TestData$Pred3 <- fitb6Test

rmsle(TestData$Pred1,result$target)


SelectedData_Test <-  TestData %>% 
  dplyr::mutate(#fit0 = fit0,
    #fit1 = fit1,
    #fit2 = fit2,
    fit3 = fit3Test,
    #fit4 = fit4,
    fit6a = fit6Test,
    fit6b = fit6bTest
    #fit7 = fit7
  )

submission3 <- data.frame(ID = SelectedData_Test$ID, 
                          target = SelectedData_Test$fit3) 
submission6a <- data.frame(ID = SelectedData_Test$ID, 
                           target = SelectedData_Test$fit6a) 
submission6b <- data.frame(ID = SelectedData_Test$ID, 
                           target = SelectedData_Test$fit6b) 
write.csv(submission3,"submission3.csv", row.names = F)
write.csv(submission6a,"submission6a.csv", row.names = F)
write.csv(submission6b,"submission6b.csv", row.names = F)






model4 <- randomForest(log1p(target)~.,
                       metry = 13,
                       data = TrainData[,-c(1,43:49)],
                       replace = TRUE,
                       ntree = 600,
                       type = "regression",
                       nodesize = 5
)
fit4 <- predict(model4,newdata = SelectedData, type = "response") %>% 
  exp()
fit4p <- predict(model4,newdata = TestData, type = "response") %>% 
  expm1()

rmsle(fit4p,result$target)
