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

XTrain <- read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
XTest  <- read.csv("test.csv",  header = TRUE, stringsAsFactors = FALSE)

testID <- XTest$Id
XTrain$Id <- NULL
XTest$Id  <- NULL
XTest$SalePrice <- NA

# Merge two data
AllData <- rbind(XTrain,XTest)

# Visualize
ggplot(data = AllData[!is.na(AllData$SalePrice),],
       aes(x = SalePrice))+geom_histogram(binwidth = 10000)+
  scale_x_continuous(breaks = seq(0,800000, by = 100000))
summary(AllData$SalePrice)

# checking numeric variables
numericVariables <- which(AllData %>% 
  sapply(., is.numeric)
)
numericVariableNames <- names(numericVariables)

# checking in-numeric variables
characterVariables <- which(AllData %>% 
                              sapply(., is.character))

characterVariables

AllNumVar <- AllData[,numericVariables]
CorNumVar <- cor(AllNumVar, use = "pairwise.complete.obs")

CorSorted <- as.matrix(sort(CorNumVar[,'SalePrice'], decreasing = TRUE))
CorHigh <- names(which(apply(CorSorted,1,function(x) abs(x)>0.5)))
CorNumVar <- CorNumVar[CorHigh,CorHigh]
library(corrplot)
corrplot.mixed(CorNumVar,tl.col = "black",tl.pos = "lt")

# OverAllQuality and SalePrice
ggplot(data = AllData[!is.na(AllData$SalePrice),],
       aes(x = factor(OverallQual),y = SalePrice))+
  geom_boxplot(col = 'blue')+
  labs(x = 'OverallQuality')+
  scale_y_continuous(breaks = seq(0,800000, by = 100000))

# Above Grade and SalePrice
ggplot(data = AllData[!is.na(AllData$SalePrice),],
       aes(x = GrLivArea, y = SalePrice))+
  geom_point(col = 'blue')+geom_smooth(method = 'lm',se = FALSE, color = 'black',
                                       aes(group = 1))+
  scale_y_continuous(breaks = seq(0,800000,by = 100000))+
  geom_text(aes(label = ifelse(AllData$GrLivArea[!is.na(AllData$SalePrice)]>4500,rownames(AllData),'')))

AllData[c(524,1299),c('SalePrice','GrLivArea','OverallQual')]      


# Completeness of Data
# pool
NACol <- which(colSums(is.na(AllData))>0)
sort(colSums(sapply(AllData[NACol],is.na)),decreasing = TRUE)

AllData$PoolQC[is.na(AllData$PoolQC)] <- 'None'
Qualities <-  c('None' = 0,
                'Po'   = 1,
                'Fa'   = 2,
                'TA'   = 3,
                'Gd'   = 4,
                'Ex'   = 5)
AllData$PoolQC <- as.integer(plyr::revalue(AllData$PoolQC,Qualities))
table(AllData$PoolQC)

AllData[AllData$PoolArea>0 & AllData$PoolQC == 0,c('PoolArea','PoolQC','OverallQual')]

AllData$PoolQC[2421] <- 2
AllData$PoolQC[2502] <- 3
AllData$PoolQC[2600] <- 2

table(AllData$MiscFeature)

# MiscFeature
AllData$MiscFeature[is.na(AllData$MiscFeature)] <- 'None'
AllData$MiscFeature <- AllData$MiscFeature %>% 
  as.factor()

# Alley
AllData$Alley[is.na(AllData$Alley)] <- 'None'
AllData$Alley <- AllData$Alley %>% 
  as.factor()

AllData$Fence[is.na(AllData$Fence)] <- 'None'
#AllData$Fence <- as.character(AllData$Fence)
AllData[!is.na(AllData$SalePrice),] %>% 
  dplyr::group_by(Fence) %>% 
  dplyr::summarise(median = median(SalePrice), counts =n())

AllData$Fence <- as.factor(AllData$Fence)

AllData$FireplaceQu[is.na(AllData$FireplaceQu)] <- 'None'
AllData$FireplaceQu <- as.integer(plyr::revalue(AllData$FireplaceQu,Qualities))

# lothogehoge
for(i in 1:nrow(AllData)){
  if(is.na(AllData$LotFrontage[i])){
    AllData$LotFrontage[i] <- as.integer(
      median(
        AllData$LotFrontage[AllData$Neighborhood==AllData$Neighborhood[i]],
        na.rm = TRUE))
  }
}

AllData$LotShape <- as.integer(revalue(AllData$LotShape,
                               c('IR3' = 0,
                                 'IR2' = 1,
                                 'IR1' = 2,
                                 'Reg' = 3)))

AllData$LotConfig <- as.factor(AllData$LotConfig)

# garage
AllData$GarageYrBlt[is.na(AllData$GarageYrBlt)] <- AllData$YearBuilt[is.na(AllData$GarageYrBlt)]

length(which(is.na(AllData$GarageType) & is.na(AllData$GarageFinish) & is.na(AllData$GarageCond) & is.na(AllData$GarageQual)))

knitr::kable(AllData[!is.na(AllData$GarageType) &
                       is.na(AllData$GarageFinish),
                     c('GarageCars', 
                       'GarageArea',
                       'GarageType',
                       'GarageCond',
                       'GarageQual', 
                       'GarageFinish')])

AllData$GarageCond[2127] <- names(sort(-table(AllData$GarageCond)))[1]
AllData$GarageQual[2127] <- names(sort(-table(AllData$GarageQual)))[1]
AllData$GarageFinish[2127] <- names(sort(-table(AllData$GarageFinish)))[1]

AllData$GarageCars[2577] <- 0
AllData$GarageArea[2577] <- 0
AllData$GarageType[2577] <- NA

AllData$GarageType[is.na(AllData$GarageType)] <- 'No Garage'
AllData$GarageType <- AllData$GarageType %>% 
  as.factor()

AllData$GarageFinish[is.na(AllData$GarageFinish)] <- 'None'
Finish <- c('None' = 0,
            'Unf'  = 1,
            'RFn'  = 2,
            'Fin'  = 3)
AllData$GarageFinish <- as.integer(revalue(AllData$GarageFinish, Finish))

AllData$GarageQual[is.na(AllData$GarageQual)] <- 'None'
AllData$GarageQual <- AllData$GarageQual %>% 
  plyr::revalue(.,Qualities) %>% 
  as.integer()

AllData$GarageCond[is.na(AllData$GarageCond)] <- 'None'
AllData$GarageCond <- AllData$GarageCond %>% 
  plyr::revalue(.,Qualities) %>% 
  as.integer()
table(AllData$GarageCond)

AllData[!is.na(AllData$BsmtFinType1) & (is.na(AllData$BsmtCond)|is.na(AllData$BsmtQual)|is.na(AllData$BsmtExposure)|is.na(AllData$BsmtFinType2)), c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')]

AllData$BsmtFinType2[333] <- names(sort(-table(AllData$BsmtFinType2)))[1]
AllData$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(AllData$BsmtExposure)))[1]
AllData$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(AllData$BsmtCond)))[1]
AllData$BsmtQual[c(2218, 2219)] <- names(sort(-table(AllData$BsmtQual)))[1]

AllData$BsmtQual[is.na(AllData$BsmtQual)] <- 'None'
AllData$BsmtQual <- AllData$BsmtQual %>% 
  plyr::revalue(., Qualities) %>% 
  as.integer()

AllData$BsmtCond[is.na(AllData$BsmtCond)] <- 'None'
AllData$BsmtCond <- AllData$BsmtCond %>% 
  plyr::revalue(., Qualities) %>% 
  as.integer()

Exposure <- c('None' = 0,
              'No'   = 1,
              'Mn'   = 2,
              'Av'   = 3,
              'Gd'   = 4)
AllData$BsmtExposure[is.na(AllData$BsmtExposure)] <- 'None'
AllData$BsmtExposure <- AllData$BsmtExposure %>% 
  plyr::revalue(., Exposure) %>% 
  as.integer()

FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
AllData$BsmtFinType1[is.na(AllData$BsmtFinType1)] <- 'None'
AllData$BsmtFinType1 <- AllData$BsmtFinType1 %>% 
  plyr::revalue(., FinType) %>% 
  as.integer()


AllData$BsmtFinType2[is.na(AllData$BsmtFinType2)] <- 'None'
AllData$BsmtFinType2 <- AllData$BsmtFinType2 %>% 
  plyr::revalue(., FinType) %>% 
  as.integer()

AllData$BsmtFullBath[is.na(AllData$BsmtFullBath)] <- 0
AllData$BsmtHalfBath[is.na(AllData$BsmtHalfBath)] <- 0
AllData$BsmtFinSF1[is.na(AllData$BsmtFinSF1)] <- 0
AllData$BsmtFinSF2[is.na(AllData$BsmtFinSF2)] <- 0
AllData$BsmtUnfSF[is.na(AllData$BsmtUnfSF)] <- 0
AllData$TotalBsmtSF[is.na(AllData$TotalBsmtSF)] <-0

AllData[is.na(AllData$MasVnrType) & !is.na(AllData$MasVnrArea), c('MasVnrType', 'MasVnrArea')]
AllData$MasVnrType[2611] <- names(sort(-table(AllData$MasVnrType)))[2]
AllData$MasVnrType[is.na(AllData$MasVnrType)] <- 'None'

Masonry <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)
AllData$MasVnrType<-AllData$MasVnrType %>% 
  revalue(., Masonry) %>% 
  as.integer()
table(AllData$MasVnrType)

AllData$MasVnrArea[is.na(AllData$MasVnrArea)] <-0

AllData$MSZoning[is.na(AllData$MSZoning)] <- names(sort(-table(AllData$MSZoning)))[1]
AllData$MSZoning <- AllData$MSZoning %>% 
  as.factor()
table(AllData$MSZoning)


AllData$KitchenQual[is.na(AllData$KitchenQual)] <- 'TA' 
AllData$KitchenQual<-AllData$KitchenQual %>%
  plyr::revalue(., Qualities) %>% 
  as.integer()
table(AllData$KitchenQual)

AllData$Utilities <- NULL

AllData$Functional[is.na(AllData$Functional)] <- names(sort(-table(AllData$Functional)))[1]
AllData$Functional <- AllData$Functional %>% 
  plyr::revalue(., c('Sal'=0,
                     'Sev'=1,
                     'Maj2'=2, 
                     'Maj1'=3,
                     'Mod'=4, 
                     'Min2'=5,
                     'Min1'=6,
                     'Typ'=7)) %>% 
  as.integer()

table(AllData$Functional)


AllData$Exterior1st[is.na(AllData$Exterior1st)] <- names(sort(-table(AllData$Exterior1st)))[1]
AllData$Exterior1st <- as.factor(AllData$Exterior1st)
table(AllData$Exterior1st)

AllData$Exterior2nd[is.na(AllData$Exterior2nd)] <- names(sort(-table(AllData$Exterior2nd)))[1]
AllData$Exterior2nd <- as.factor(AllData$Exterior2nd)

AllData$ExterQual<-as.integer(revalue(AllData$ExterQual, Qualities))

AllData$ExterCond<-as.integer(revalue(AllData$ExterCond, Qualities))

AllData$Electrical[is.na(AllData$Electrical)] <- names(sort(-table(AllData$Electrical)))[1]
AllData$Electrical <- as.factor(AllData$Electrical)
table(AllData$Electrical)

AllData$SaleType[is.na(AllData$SaleType)] <- names(sort(-table(AllData$SaleType)))[1]
AllData$SaleType <- as.factor(AllData$SaleType)

AllData$SaleCondition<- as.factor(AllData$SaleCondition)

Charcol <- names(AllData[,sapply(AllData, is.character)])

AllData$Foundation   <- as.factor(AllData$Foundation)
AllData$Heating      <- as.factor(AllData$Heating)
AllData$HeatingQC    <- as.integer(plyr::revalue(AllData$HeatingQC, Qualities))
AllData$CentralAir   <- as.integer(plyr::revalue(AllData$CentralAir, c('N'=0, 'Y'=1)))
AllData$RoofStyle    <- as.factor(AllData$RoofStyle)
AllData$RoofMatl     <- as.factor(AllData$RoofMatl)
AllData$LandContour  <- as.factor(AllData$LandContour)
AllData$LandSlope    <- as.integer(plyr::revalue(AllData$LandSlope, c('Sev'=0, 'Mod'=1, 'Gtl'=2)))
AllData$BldgType     <- as.factor(AllData$BldgType)
AllData$HouseStyle   <- as.factor(AllData$HouseStyle)
AllData$Neighborhood <- as.factor(AllData$Neighborhood)
AllData$Condition1   <- as.factor(AllData$Condition1)
AllData$Condition2   <- as.factor(AllData$Condition2)
AllData$Street       <- as.integer(plyr::revalue(AllData$Street, c('Grvl'=0, 'Pave'=1)))
AllData$PavedDrive   <- as.integer(plyr::revalue(AllData$PavedDrive, c('N'=0, 'P'=1, 'Y'=2)))

AllData$MoSold       <- as.factor(AllData$MoSold)
AllData$MSSubClass   <- as.factor(AllData$MSSubClass)

write.csv(AllData,"UseData.csv",row.names = FALSE)