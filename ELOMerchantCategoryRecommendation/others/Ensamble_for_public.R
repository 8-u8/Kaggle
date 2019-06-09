# blending

library(tidyverse)
library(dummies) # to one hot
library(corrplot)

Num <- c(22:25,27:54)
fileName <- paste0("submission",Num,".csv")
SubmittedFile <- list()
samplesubmission <- readr::read_csv("sample_submission.csv")

# Load my historical submission
for(i in 1:length(Num)){
  print("read_csv")
  SubmittedFile[[i]] <- readr::read_csv(fileName[i])
  colnames(SubmittedFile[[i]]) <- c("card_id", paste0("target_",Num[i]))
  if(i == 1){
   Submitted_Merge <- SubmittedFile[[i]]
  }else{
    Submitted_Merge <- Submitted_Merge %>% 
      inner_join(SubmittedFile[[i]], by = "card_id")
  print(paste0("merged target ",Num[i],"th table"))
  }
}
# Loading other kagglers' submission.
ens1 <- readr::read_csv("pastSub/ens.csv") %>% 
  dplyr::rename(target_ens1 = target)
ens2 <- readr::read_csv("pastSub/submission_lgbxgb.csv")%>% 
  dplyr::rename(target_ens2 = target)
ens3 <- readr::read_csv("pastSub/combining_submission.csv")%>% 
  dplyr::rename(target_ens3 = target)
ens4 <- readr::read_csv("pastSub/submit_lgb_ein.csv")%>% 
  dplyr::rename(target_ens4 = target)
ens5 <- readr::read_csv("pastSub/submit_xgb_ein.csv")%>% 
  dplyr::rename(target_ens5 = target)
ens6 <- readr::read_csv("pastSub/RLS_ein.csv")%>% 
  dplyr::rename(target_ens6 = target)
ens7 <- readr::read_csv("pastSub/blend.csv")%>% 
  dplyr::rename(target_ens7 = target)
ens8 <- readr::read_csv("pastSub/blend2.csv")%>% 
  dplyr::rename(target_ens8 = target)
ens9 <- readr::read_csv("pastSub/blend3.csv")%>% 
  dplyr::rename(target_ens9 = target)
ens10 <- readr::read_csv("pastSub/Bestoutput.csv")%>% 
  dplyr::rename(target_ens10 = target)
ens11 <- readr::read_csv("pastSub/LGB_submission55.csv") %>% 
  dplyr::rename(target_ens11 = target)
ens12 <- readr::read_csv("pastSub/XGB_submission55.csv") %>% 
  dplyr::rename(target_ens12 = target)
ens13 <- readr::read_csv("pastSub/Stack_submission55.csv") %>% 
  dplyr::rename(target_ens13 = target)

Submitted_Merge <- Submitted_Merge %>% 
  dplyr::inner_join(ens1, by = "card_id") %>% 
  dplyr::inner_join(ens2, by = "card_id") %>% 
  dplyr::inner_join(ens3, by = "card_id") %>% 
  dplyr::inner_join(ens4, by = "card_id") %>% 
  dplyr::inner_join(ens5, by = "card_id") %>% 
  dplyr::inner_join(ens6, by = "card_id") %>% 
  dplyr::inner_join(ens7, by = "card_id") %>% 
  dplyr::inner_join(ens8, by = "card_id") %>% 
  dplyr::inner_join(ens9, by = "card_id") %>% 
  dplyr::inner_join(ens10, by = "card_id") %>% 
  dplyr::inner_join(ens11, by = "card_id") %>% 
  dplyr::inner_join(ens12, by = "card_id") %>% 
  dplyr::inner_join(ens13, by = "card_id")
corResult <- cor(Submitted_Merge[,-1],method = "pearson")
corrplot(corr = corResult, method = "square", type = "upper")
RowCOrr <- SubmittedFile[[23]] %>% 
  dplyr::inner_join(SubmittedFile[[24]], by = "card_id") %>% 
  dplyr::inner_join(SubmittedFile[[25]], by = "card_id")
RowCOrr$target_ens5 <- Submitted_Merge$target_ens5
RowCOrr$target_ens12 <- Submitted_Merge$target_ens12
  
RowCOrrMeantarg <- apply(RowCOrr[,-1],MARGIN = 1, FUN = mean)
Meantarget <- apply(Submitted_Merge[,-1], MARGIN = 1, FUN = mean)
Meantarget <- Meantarget * 0.9 + RowCOrrMeantarg*0.1
cor(Meantarget, RowCOrrMeantarg)
cor(ens8$target_ens8,Meantarget)
submit_mean <- data.frame(card_id = Submitted_Merge$card_id, target = Meantarget)

write.csv(submit_mean, "submission.csv", row.names = FALSE)
