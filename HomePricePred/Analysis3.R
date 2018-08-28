library(tidyverse)
library(Rmisc)
library(knitr)
library(ggrepel)
library(plyr)
library(randomForest)
library(nnet)
library(glmnet)
library(MASS)
library(caret)
library(corrplot)
# 整形済みデータ
UseData <- read.csv("UseData.csv",header = TRUE)

#TrainDataの設定
TrainControl <- caret::trainControl(method = "boot",
                                    number = 2000,
                                    search = "random",
                                    returnData = TRUE)
# 全部ぶち込んでRandom Forest
fit1 <- randomForest(x=AllData[1:1460,-79], 
                     y=AllData$SalePrice[1:1460],
                     TrainControl = TranControl,
                     mtry = 13,
                     replace = TRUE,
                     importance = TRUE,
                     type = "regression",
                     ntree = 50
                     )

# Which features is important?
imp_RF <- importance(fit1)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]
ggplot(imp_DF[1:20,],
       aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) +
  geom_bar(stat = 'identity') +
  labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') +
  coord_flip() + theme(legend.position="none")


