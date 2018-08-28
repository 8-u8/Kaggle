#Load libraries
library(tidyverse)
library(data.table)
library(corrplot)
library(Metrics)
library(randomForest)
library(MASS)
library(rfUtilities)
library(caret)
library(xgboost)
#Reading Data
#TrainData
GetCor <- function(x){
  data.matrix(x) %>% 
    cor(method = "spearman") %>% 
    findCorrelation(cutoff = .98)
}

GetZeroVar <- function(x){
  var(x)!=0
}

TrainData <- read.table(file = "train.csv",sep = ",",header = TRUE) %>% 
  dplyr::select_if(GetZeroVar) %>% 
  dplyr::select(-GetCor(.))
summary(TrainData) %>% 
  broom::tidy() %>%  
  separate(Freq, c('Level',"Value"),":") %>% 
  mutate(Value = as.numeric(Value)) %>% 
  filter(trimws(Level) %in% "3rd Qu.") %>% 
  filter(!(trimws(Var2) %in% 'target')) %>% 
  arrange(desc(Value)) %>% 
  filter(!is.na(Value),
         Value > 0) %>%
  mutate(Var2 = as.character(Var2)) -> ColumnsToUse

TrainData %>% 
  dplyr::select(one_of("ID","target"),
         trimws(ColumnsToUse$Var2)) -> SelectedData1
colnames(SelectedData1) <- as.character(colnames(SelectedData1))

train <- fread("train.csv", header = TRUE)
GEE40 <- c('f190486d6', 
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


UseData <- TrainData[,c("ID","target",GEE40)]
ColumnsToUse <- colnames(UseData[-c(1,2)])

rm(TrainData,SelectedData1,SelectedData2,SelectedData)
gc()



train <- train[,c("ID","target","f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f","fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916","b43a7cfd5","58232a6fb"),with=F]
train <- train[ c(2072,3493,379,2972,2367,4415,2791,3980,194,1190,3517,811,4444) ]
head(train)


#TestData
TestData <- read.table(file = "test.csv",sep = ",",header = TRUE)

#TestData <- TestData %>% 
#  dplyr::select(-GetCor(.)) 

#summary(TestData) %>% 
#  broom::tidy() %>% 
#  separate(Freq, c('Level',"Value"),":") %>% 
#  mutate(Value = as.numeric(Value)) %>% 
#  filter(trimws(Level) %in% "3rd Qu.") %>% 
#  filter(!(trimws(Var2) %in% 'target')) %>% 
#  arrange(desc(Value)) %>% 
#  filter(!is.na(Value),
#         Value > 0)

TestData %>% 
  dplyr::mutate(target= 0) %>% 
  dplyr::select(one_of("ID","target"),
         trimws(ColumnsToUse)) -> SelectedData_Test


rm(TestData)
gc()


for(i in 1:40){
  UseData[,42+i] <- UseData[2+i]-UseData[,3+i]
  UseData[,82+i] <- UseData[42+i]-UseData[43+i]
  SelectedData_Test[,42+i] <- SelectedData_Test[,2+i]-SelectedData_Test[,3+i]
  SelectedData_Test[,82+i] <- SelectedData_Test[,42+i]-SelectedData_Test[,43+i]
}



#Visualization
coltabl <-  cor(SelectedData$target,SelectedData[,-c(1,2)])
hist(coltabl[1,])
summary(coltabl[1,])





ggplot()+
  geom_histogram(mapping = aes(x = SelectedData1[,39]),data = SelectedData1)

corrplot::corrplot(cor(SelectedData1[,c(3:42)]), method = "square", 
                   type = "lower", tl.cex = 0.5)


#modeling
#model0: OLS linear regression
#model1: OLS linear regression(log-scaled)
#model2; MLE linear regression(negative-binomial)
#model3: Random Forest
#####
model0 <- lm(target ~ ., data = SelectedData[,-1])
fit0 <- predict(model0,newdata = SelectedData, type = "response")
fit0 <- ifelse(fit0<0,0,fit0)

model1 <- lm(log(target) ~ ., data = SelectedData[,-1])
fit1 <- exp(predict(model1,newdata = SelectedData,type ="response"))

model2 <- glm.nb(target~.,data = SelectedData[,-1],link = log,init.theta = 2)
#summary(model2)
fit2 <- predict(model2, newdata = SelectedData,type ="response")
fit2 <- ifelse(fit2<0,0,fit2)
fit2[fit2 == max(fit2)] <- mean(fit2)
#####
model3 <- randomForest(target ~ ., 
                       data = UseData[,-1],
                       mtry = 13,
                       replace = TRUE,
                       importance = TRUE,
                       type = "regression",
                       nodesize = 5)

fit3 <- predict(model3,newdata = SelectedData, type = "response")
summary(fit3)

tuning <-  tuneRF(x = SelectedData[,-c(1,2)],y = SelectedData[,2],doBest = TRUE)
logtuning <- tuneRF(x = SelectedData[,-c(1,2)],y = log(SelectedData[,2]),doBest = TRUE)

model4 <- randomForest(log1p(target)~.,
                       metry = 13,
                       data = UseData[,-1],
                       replace = TRUE,
                       ntree = 600,
                       type = "regression",
                       nodesize = 5
                       )
fit4 <- predict(model4,newdata = SelectedData, type = "response") %>% 
  exp()
fit4p <- predict(model4,newdata = SelectedData_Test, type = "response") %>% 
  expm1()
summary(fit4p)

SelectedData$targpred <- fit4
SelectedData_Test$targpred <- fit4p

model4b <- randomForest(log(target)~log(.),
                       data = SelectedData[,-1],
                       replace = FALSE)
fit4b <- predict(model4b,newdata = SelectedData_Test, type = "response") %>% 
  exp1m()

library(glmnet)
library(glmnetUtils)

model5 <- glmnetUtils::cv.glmnet(target ~. , data = SelectedData[,-1],
                                 alpha = 1,
                                 nfolds = 1000
                                 )
fit5 <- predict(model5,newdata = SelectedData)



model6 <- caret::train(target~.,data = UseData[,-1],
                       tuneLength = 5,
                       preProcess = c("center","scale"),
                       method = "xgbLinear",
                       trControl = trainControl("cv",number = 20),
                       yLimits =c(0,Inf),
                       tuneGrid = xgbGrid
                       )
fit6 <- predict(model6b,newdata = UseData[,-1]) %>% 
  expm1()
fit6[fit6<0] <- 0
plot(fit6,UseData$target)


xgbGrid <- expand.grid(eta = .3, 
                       max_depth = 4,
                       nrounds = c(1:50),
                       gamma = 0,
                       colsample_bytree = 0.8,
                       min_child_weight = 1,
                       subsample = 0.875
                       )

model6b<- caret::train(log1p(target)~.,data = UseData[,-1],
                       tuneLength = 5,
                       preProcess = c("center","scale"),
                       method = "xgbTree",
                       trControl = trainControl("cv",number = 1000),
                       tuneGrid = xgbGrid

)



model6c<- caret::train(log(target)~.,data = SelectedData[,-1],
                       tuneLength = 5,
                       preProcess = c("center","scale"),
                       method = "xgbTree",
                       trControl = trainControl("cv",number = 20)
)


fit6b <- predict(model6b,newdata = SelectedData_Test) %>% 
  expm1()
summary(fit6b)

library(Metrics)
rmsle(fit6b,SelectedData$target)

SelectedData_Test <-  SelectedData_Test %>% 
  dplyr::mutate(#fit0 = fit0,
                #fit1 = fit1,
                #fit2 = fit2,
                #fit3 = fit3,
                #fit4 = fit4,
                #fit5 = fit5,
                fit6 = fit6b
                #fit7 = fit7
                )
#dplyr::select(colnames(SelectedData))

ggplot(data = SelectedData)+
  #geom_point(aes(x = target,y = fit0),color = "red")+
  #geom_point(aes(x = target,y = fit1),color = "blue")+
  #geom_point(aes(x = target,y = fit2),color = "yellow")+
  geom_point(aes(x = target,y = fit3),color = "green")+
  geom_point(aes(x = target,y = fit4),color = "red")


submission <- data.frame(ID = SelectedData_Test$ID, 
                         target = SelectedData_Test$fit6) 

submission$target <- ifelse(submission$target < 0, 0, submission$target)
write.csv(submission,"submission6.csv", row.names = F)
gc()  
