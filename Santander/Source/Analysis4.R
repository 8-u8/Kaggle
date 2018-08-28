library(tidyverse)
library(data.table)
library(corrplot)
library(Metrics)
library(randomForest)
library(MASS)
library(rfUtilities)
library(caret)
library(xgboost)

TrainData <- read.table(file = "train.csv",sep = ",",header = TRUE)
TestData  <- read.table(file = "test.csv",sep = ",",header = TRUE)
RESULT    <- read.table(file = "final.csv",sep = ",",header = TRUE)
RESULT2   <- read.table(file = "baseline_submission_with_leaks.csv",sep = ",",header = TRUE)
RESULT3   <- read.table(file = "baseline_submission_with_leaks_ROUNDED_MINUS2.csv",sep = ",",header = TRUE)
RESULT4   <- read.table(file = "submission3.csv",sep = ",",header = TRUE)
RESULT6   <- read.table(file = "submission6b.csv",sep = ",",header = TRUE)

#RESULT$target <- 0.6*RESULT$target+0.1*(RESULT2$target+RESULT3$target+RESULT4$target+RESULT6$target)
rm(RESULT2,RESULT3,RESULT4,RESULT6)

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
TestData <- TestData %>% 
  inner_join(.,RESULT,by = "ID")

TrainData$EZ     <- apply(TrainData[,3:42],1,function(x)length(x[x==0]))
TrainData$Mean   <- apply(TrainData[,3:42],1,function(x)mean(x,na.rm = TRUE))
TrainData$Max    <- apply(TrainData[,3:42],1,function(x)max(x))
TrainData$Min    <- apply(TrainData[,3:42],1,function(x)min(x))
TrainData$Gap    <- apply(TrainData[,3:42],1,function(x)exp(mean(log(x),na.rm = TRUE)))

TestData$EZ      <- apply(TestData[,3:42],1,function(x)length(x[x==0]))
TestData$Mean    <- apply(TestData[,3:42],1,function(x)mean(x,na.rm = TRUE))
TestData$Max     <- apply(TestData[,3:42],1,function(x)max(x))
TestData$Min     <- apply(TestData[,3:42],1,function(x)min(x))
TestData$Gap     <- apply(TestData[,3:42],1,function(x)exp(mean(log(x),na.rm = TRUE)))

TrainControl <- caret::trainControl(method = "boot",
                                    number = 2000,
                                    search = "random",
                                    returnData = TRUE)


model3 <- randomForest(log1p(target) ~ ., 
                       data = TrainData[,-1],
                       mtry = 13,
                       replace = TRUE,
                       importance = TRUE,
                       type = "regression",
                       ntree = 50,
                       TrainControl = TranControl,
                       nodesize = 7)

hist(model3$predicted %>% expm1())
fit3Train <- predict(model3,newdata = TrainData, type = "response") %>% 
  expm1()

fit3Test <- predict(model3,newdata = TestData, type = "response") %>% 
  expm1()
rmsle(model3$predicted %>% expm1(),TrainData$target)
plot(TestData$target,fit3Test)

TrainData$Pred1 <- fit3Train
TestData$Pred1 <- fit3Test

rmsle(TestData$Pred1,TestData$target)


xgbGrid <- expand.grid(eta = .3, 
                       max_depth = 4,
                       nrounds = c(1:750),
                       gamma = 0,
                       colsample_bytree = 0.8,
                       min_child_weight = 1,
                       subsample = 0.875
)

Weights <- rnorm(nrow(TrainData),0,1)

model6 <- caret::train(target~.,data = TrainData[,-1],
                       tuneLength = 5,
                       preProcess = c("center","scale"),
                       method = "xgbTree",
                       trControl = trainControl("cv",number = 20),
                       tuneGrid = xgbGrid,
                       weights = Weights
)


fit6Train <- predict(model6,newdata = TrainData) 

fit6Test <- predict(model6,newdata = TestData) 
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

rmsle(TestData$Pred1,RESULT$target)


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



