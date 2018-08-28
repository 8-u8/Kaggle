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

FEATURE <- c('f190486d6', 
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
             'X0ff32eb98',
             "target"
)

TrainData <- TrainData[,FEATURE]
TestData <- TestData[,FEATURE[-length(FEATURE)]]
TestData$target <- min(TrainData$target)


hist(TrainData$target,breaks =20)

BaseArima <- apply(TrainData[,1:40],1,function(x)forecast(auto.arima(rev(x)),h = 2)$mean[2])   
BaseArima2 <-  ((BaseArima - min(BaseArima))/(max(BaseArima) - min(BaseArima)))*(max(TrainData$target) - min(TrainData$target)) + min(TrainData$target)

NonZeroMean <- function(x){
  x[x == 0] <- mean(x[x!=0])
  return(x)
}
NonZero1Q <- function(data){
  data[data==0] <- quantile(x = data[data!=0], type = 1, na.rm = TRUE)
  return(data)
}

TrainData$zero <- apply(TrainData[,1:40],1,function(x)sum(x))
TrainData <- TrainData[which(TrainData$zero>0),]
dim(TrainData2)

ArimaNonZeroMean <- apply(TrainData[,1:40],1,function(x)forecast(auto.arima(rev(NonZeroMean(x))),h = 2)$mean[2])
ArimaNonZeroMean2 <- ((ArimaNonZeroMean - min(ArimaNonZeroMean))/(max(ArimaNonZeroMean) - min(ArimaNonZeroMean)))*(max(TrainData$target) - min(TrainData$target)) + min(TrainData$target)

cor(ArimaNonZeroMean2,TrainData$target)

MovingAvg <- function(x, n = 2){
  cx <- c(0,cumsum(x))
  rsum <- (cx[(n+1):length(cx)]-cx[1:(length(cx)-n)])/n
  return(rsum)
}

ArimaMovingAvg <- apply(TrainData[,1:40],
                        1,
                        function(x)forecast(
                        auto.arima(
                          rev(MovingAvg(NonZeroMean(x)))
                        ), h = 2
                        )$mean[2]
                        )

ArimaMovingAvgTrain <- apply(TrainData[,1:40],
                              1,
                              function(x)forecast(
                                auto.arima(
                                  rev(MovingAvg(NonZeroMean(x)))
                                ), h = 2
                              )$mean[2]
)

#TrainData$Arima <- ArimaMovingAvgTrain
TrainData$NZMean <- apply(TrainData[,1:40],1,function(x)mean(x[x!=0]))
TrainData$NZMax <- apply(TrainData[,1:40],1,function(x)max(x[x!=0]))
TrainData$NZMin <- apply(TrainData[,1:40],1,function(x)min(x[x!=0]))
TrainData$EZ <- apply(TrainData[,1:40],1,function(x)length(x[x==0]))
TrainData$Mean <- apply(TrainData[,1:40],1,function(x)mean(x))
TrainData$Max <- apply(TrainData[,1:40],1,function(x)max(x))
TrainData$Min <- apply(TrainData[,1:40],1,function(x)min(x))



#TestData$zero <-  apply(TestData[,1:40],1,function(x)sum(x))
TestData <- TestData[which(TestData$zero>0),]
ArimaMovingAvgTest <- apply(TestData[,1:40],
                                 1,
                                 function(x)forecast(
                                   auto.arima(
                                     rev(MovingAvg(NonZeroMean(x)))
                                   ), h = 2
                                 )$mean[2]
)
TestData$Arima <- ArimaMovingAvgTest
TestData$NZMean <- apply(TestData[,1:40],1,function(x)mean(x[x!=0]))
TestData$NZMax <- apply(TestData[,1:40],1,function(x)max(x[x!=0]))
TestData$NZMin <- apply(TestData[,1:40],1,function(x)min(x[x!=0]))
TestData$EZ <- apply(TestData[,1:40],1,function(x)length(x[x==0]))
TestData$Mean <- apply(TestData[,1:40],1,function(x)mean(x))
TestData$Max <- apply(TestData[,1:40],1,function(x)max(x))
TestData$Min <- apply(TestData[,1:40],1,function(x)min(x))



model3 <- randomForest(target ~ ., 
                       data = TrainData[,-42],
                       mtry = 13,
                       replace = TRUE,
                       importance = TRUE,
                       type = "regression",
                       nodesize = 5)

fit3 <- predict(model3,newdata = TestData, type = "response")
summary(fit3)


xgbGrid <- expand.grid(eta = .3, 
                       max_depth = 4,
                       nrounds = c(1:50),
                       gamma = 0,
                       colsample_bytree = 0.8,
                       min_child_weight = 1,
                       subsample = 0.875
)


model6 <- caret::train(target~.,data = TrainData[,-42],
                       tuneLength = 5,
                       preProcess = c("center","scale"),
                       method = "xgbTree",
                       trControl = trainControl("cv",number = 20),
                       tuneGrid = xgbGrid
)


fit6 <- predict(model6,newdata = TestData)
summary(fit6)

model6b<- caret::train(log1p(target)~.,data = TrainData[,-42],
                       tuneLength = 5,
                       preProcess = c("center","scale"),
                       method = "xgbTree",
                       trControl = trainControl("cv",number = 1000),
                       tuneGrid = xgbGrid
                       
)

fit6b <- predict(model6b,newdata = TestData) %>% 
  expm1()
summary(fit6b)


SelectedData_Test <-  TestData %>% 
  dplyr::mutate(#fit0 = fit0,
    #fit1 = fit1,
    #fit2 = fit2,
    fit3 = fit3,
    #fit4 = fit4,
    fit6a = fit6,
    fit6b = fit6b
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

gc()  

submission$target <- ifelse(submission$target < 0, 0, submission$target)
write.csv(submission,"submission6.csv", row.names = F)
gc()  


submission$target <- ifelse(submission$target < 0, 0, submission$target)
write.csv(submission,"submission6.csv", row.names = F)
gc()  
submission$target <- ifelse(submission$target < 0, mean(submission$target), submission$target)
write.csv(submission,"submission6.csv", row.names = F)
gc()  