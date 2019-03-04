library(tidyverse)

usedata <- readr::read_csv("Here is my dataset")

target_and_id <- usedata %>% 
  select(card_id, target, outliers)

usedata <- usedata %>% 
  dplyr::select(-target, -outliers) %>% 
  dplyr::mutate_if(is.numeric, funs(replace(., is.na(.),0))) %>% 
  dplyr::mutate_if(is.numeric, funs(replace(., is.infinite(.),0))) %>% 
  dplyr::mutate_if(is.numeric, funs(scale(.))) %>% 
  dplyr::inner_join(target_and_id, by = "card_id")

usedata$purchase_amount_mean <- log1p(usedata$purchase_amount_mean)


summary(usedata)
write.csv(usedata, "output/mergedFulldata(By_Tsai)_8_naomitted_scale.csv", row.names = FALSE)

trainClean <- readr::read_csv("train_clean.csv")
testClean  <- readr::read_csv("test_clean.csv")

colnames(trainClean)
testClean$outliers <- 0
testClean$target <- NA

mergedData <- trainClean %>% 
  dplyr::bind_rows(testClean)
for(COL in 1:ncol(mergedData)){
  if(is.numeric(mergedData[,COL])){
    mergedData[,COL][is.na(mergedData[,COL])] <- 0
    mergedData[,COL][is.infinite(mergedData[,COL])] <- 0
  }
}
write.csv(mergedData, "output/mergedFullData_6.csv", row.names = FALSE)

tomergedata <- mergedData %>% dplyr::select(-target, -outliers)
colnames(tomergedata)[-2] <- paste0(colnames(tomergedata)[-2],"_by_ein")
fullofdata <- mergedData %>% 
  dplyr::inner_join(tomergedata, by = "card_id")

write.csv(fullofdata, "output/merged_ein_Tsai.csv", row.names = FALSE)
