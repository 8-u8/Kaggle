import datetime
import gc
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings
import optuna

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, BayesianRidge
from catboost import CatBoostRegressor

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
UseData = pd.read_csv("Here is aggregated dataset")



train   = pd.read_csv("train.csv")
test   = pd.read_csv("test.csv")
train_df = UseData[UseData['card_id'].isin(train['card_id'])]
test_df  = UseData[UseData['card_id'].isin(test['card_id'])]
train_df_pred = train_df[["card_id", "target","outliers"]]
FEATS_EXCLUDED = [
    'first_active_month', 'target', 'card_id', 'outliers',
    'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
    'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
    'OOF_PRED', 'month_0','outliers_pred']

train_df['rounded_target'] = train_df['target'].round(0)
train_df = train_df.sort_values('rounded_target').reset_index(drop=True)
vc = train_df['rounded_target'].value_counts()
vc = dict(sorted(vc.items()))
df = pd.DataFrame()
train_df['indexcol'],i = 0,1
for k,v in vc.items():
    step = train_df.shape[0]/v
    indent = train_df.shape[0]/(v+1)
    df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
    for j in range(0, v):
        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
    df = pd.concat([df2,df])
    i+=1
train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
del train_df['indexcol'], train_df['rounded_target']


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
    
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=True))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances_otherData.png')

# reduce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, debug= False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=False, random_state=514)
    else:
        folds = KFold(n_splits= num_folds, shuffle=False, random_state=514)
    #Folds_xgb = KFold(n_splits = num_folds, shuffle = True, random_state= 114)
    # Create arrays and dataframes to store results
    oof_preds_lgb = np.zeros(train_df.shape[0])
    sub_preds_lgb = np.zeros(test_df.shape[0])
    train_preds_lgb = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    #Features = [c for c in train_df.columns if c not in FEATS_EXCLUDED]
    #num_rounds = 20000

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats],train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)
        # params optimized by optuna and hand.
        params = {
            #'gpu_use_dp': False, 
            #'gpu_platform_id': 0, 
            #'gpu_device_id': 0, 
            #'device': 'gpu', 
            'objective': 'regression_l2', 
            'boosting_type': 'gbdt', 
            'metric': 'rmse', 
            'n_jobs': 4, 
            'max_depth': 8,#19, 
            'n_estimators': 2000, 
            'subsample_freq': 2, 
            'subsample_for_bin': 200000, 
            'min_data_per_group': 100, 
            'max_cat_to_onehot': 4, 
            'cat_l2': 10.0, 
            'cat_smooth': 10.0, 
            'max_cat_threshold': 32, 
            'metric_freq': 10, 
            'verbosity': -1, 
            'colsample_bytree': 0.6005138538429582,#0.5, 
            'learning_rate': 0.074878493529297897,#0.061033234451294376, 
            'min_child_samples': 80, 
            'min_child_weight': 100.0, 
            'min_split_gain': 1e-06, 
            'num_leaves': 47, 
            #'top_rate':0.05650844871956809,
            #'other_rate':0.8956221585581838,
            'reg_alpha': 5.6842888116329355, #10.0, 
            'reg_lambda': 0.5140961552906037,#10.0, 
            'subsample': 0.7119184557612507#0.9
            }

        lgb_model = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'valid'],
                        num_boost_round=10000,
                        early_stopping_rounds= 500,
                        verbose_eval=100
                        )

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(lgb_model.feature_importance(importance_type='gain', iteration=lgb_model.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof_preds_lgb[valid_idx] = lgb_model.predict(valid_x, num_iteration=lgb_model.best_iteration)
        sub_preds_lgb += lgb_model.predict(test_df[feats], num_iteration=lgb_model.best_iteration) / folds.n_splits
        train_preds_lgb += lgb_model.predict(train_df[feats], num_iteration = lgb_model.best_iteration)/folds.n_splits
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds_lgb[valid_idx])))
    display_importances(feature_importance_df)
    RMSE_LGB = rmse(valid_y, oof_preds_lgb[valid_idx])
    del lgb_train, lgb_test, train_idx, valid_idx
    gc.collect()
    return sub_preds_lgb, train_preds_lgb, RMSE_LGB


# XGBoost
def kfold_xgboost(train_df, test_df, num_folds, stratified = False,debug = False):
    print("Starting XGBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    # Cross validation model
    if stratified:
        Folds_xgb = StratifiedKFold(n_splits= num_folds, shuffle=False, random_state=514)
    else:
        Folds_xgb = KFold(n_splits= num_folds, shuffle=False, random_state=514)
    #Folds_xgb = KFold(n_splits = num_folds, shuffle = True, random_state= 114)
    oof_preds_xgb = np.zeros(len(train_df))
    sub_preds_xgb = np.zeros(len(test_df))
    train_preds_xgb = np.zeros(len(train_df))
    #feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
   
    xgb_params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric':'rmse',
            'gamma': 3.219599368724918,
            'max_depth': 17,#18, 
            'lambda': 1.0,#0.3108515038252404,
            'subsample': 0.01, 
            'colsample_bytree': 0.3, 
            'min_child_weight': 0.5, 
            'eta': 0.01, 
            'alpha': 1.0#1.8880254408453676
            }

    for fold_, (train_idx_xgb, valid_idx_xgb) in enumerate(Folds_xgb.split(train_df.values,train_df['outliers'].values)):
        train_x, train_y = train_df[feats].iloc[train_idx_xgb], train_df['target'].iloc[train_idx_xgb]
        valid_x, valid_y = train_df[feats].iloc[valid_idx_xgb], train_df['target'].iloc[valid_idx_xgb]

        print("fold n °{}".format(fold_))
        trn_Data = xgb.DMatrix(train_x, label = train_y)
        val_Data = xgb.DMatrix(valid_x, label = valid_y)
        watchlist = [(trn_Data, "Train"), (val_Data, "Valid")]
        print("xgb" + str(fold_) + "-" * 50)
        num_rounds = 10000
        xgb_model = xgb.train(xgb_params, trn_Data,num_rounds,watchlist,early_stopping_rounds=500, verbose_eval= 100)
        oof_preds_xgb[valid_idx_xgb] = xgb_model.predict(xgb.DMatrix(train_df.iloc[valid_idx_xgb][feats]), ntree_limit = xgb_model.best_ntree_limit + 50)
        sub_preds_xgb += xgb_model.predict(xgb.DMatrix(test_df[feats]),ntree_limit= xgb_model.best_ntree_limit)/Folds_xgb.n_splits
        train_preds_xgb = xgb_model.predict(xgb.DMatrix(train_df[feats]), ntree_limit=xgb_model.best_ntree_limit)/Folds_xgb.n_splits
        print('Fold %2d RMSE : %.6f' % (fold_ + 1, rmse(valid_y, oof_preds_xgb[valid_idx_xgb])))
        RMSE_XGB = rmse(valid_y, oof_preds_xgb[valid_idx_xgb])
        del xgb_model,train_idx_xgb,valid_idx_xgb
        gc.collect()
    return sub_preds_xgb, train_preds_xgb,RMSE_XGB

# Catboost
def kfold_catboost(train_df, test_df, num_folds, debug = False):
    Folds_cat = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=514)
    oof_preds_cat = np.zeros(len(train_df))
    sub_preds_cat = np.zeros(len(test_df))
    train_preds_cat = np.zeros(len(train_df))
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    for fold_, (train_idx_cat, valid_idx_cat) in enumerate(Folds_cat.split(train_df.values,train_df['outliers'].values)):
        print("catboost"+str(fold_))
        train_x, train_y = train_df[feats].iloc[train_idx_cat], train_df['target'].iloc[train_idx_cat]
        valid_x, valid_y = train_df[feats].iloc[valid_idx_cat], train_df['target'].iloc[valid_idx_cat]
        reg_cat = CatBoostRegressor(iterations = 2000,
                                    learning_rate=0.1,
                                    max_depth= 7,
                                    eval_metric="RMSE",
                                    random_seed=514,
                                    l2_leaf_reg=40,
                                    metric_period=50,
                                    random_strength=5,
                                    early_stopping_rounds=500,
                                    od_pval= 0.01,
                                    )
        cat_model = reg_cat.fit(train_x, train_y)
        #oof_preds_cat = reg_cat.predict(train_df[feats][valid_idx_cat])
        sub_preds_cat = reg_cat.predict(test_df[feats])
        train_preds_cat = reg_cat.predict(train_df[feats])
        #print('Fold %2d RMSE : %.6f' % (fold_ + 1, rmse(valid_y, oof_preds_cat[valid_idx_cat])))
        #RMSE_CAT = rmse(valid_y, oof_preds_cat)
        del train_x, train_y, valid_x, valid_y, reg_cat
        gc.collect
    return sub_preds_cat, train_preds_cat

#stacking
def stacking(train_lgb, train_xgb, train_cat,test_lgb,test_xgb,test_cat):
    stack_train = train_df_pred
    stack_train['target_lgb'] = train_lgb
    stack_train['target_xgb'] = train_xgb 
    stack_train['target_cat'] = train_cat 
    stack_test = test_df
    stack_test['target_lgb'] = test_lgb
    stack_test['target_xgb'] = test_xgb 
    stack_test['target_cat'] = test_cat 
    target = train_df['target']
    print(stack_train.columns)
    folds_stack = KFold(n_splits=6, shuffle=False, random_state=514)
    feats = ["target_lgb","target_xgb", "target_cat"]
    stack_test = stack_test[feats]
    xgb_params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eval_metric':'rmse',
        'gamma': 3.0,
        'max_depth': 3,#18, 
        'lambda': 10.0,
        'subsample': 0.5, 
        'colsample_bytree': 0.2, 
        'min_child_weight': 15.0, 
        'eta': 0.5, 
        'alpha': 10.0}

    for folds_, (train_idx, valid_idx) in enumerate(folds_stack.split(stack_train[feats].values, stack_train['outliers'].values)):
        train_x, train_y = stack_train[feats].iloc[train_idx], target.iloc[train_idx]
        valid_x, valid_y = stack_train[feats].iloc[valid_idx], target.iloc[valid_idx]
        
        train_stack = xgb.DMatrix(train_x, label = train_y)
        valid_stack = xgb.DMatrix(valid_x, label = valid_y)
        watchlist = [(train_stack, "Train"), (valid_stack, "Valid")]  
        num_rounds = 10000
        print("stack"+str(folds_))
        stack_xgb = xgb.train(xgb_params, train_stack,num_rounds, watchlist, early_stopping_rounds=50, verbose_eval= 1000)
        oof_stack_xgb = stack_xgb.predict(valid_stack, ntree_limit = stack_xgb.best_ntree_limit + 50)
        stack_preds_xgb = stack_xgb.predict(xgb.DMatrix(stack_test[feats]),ntree_limit= stack_xgb.best_ntree_limit)
        RMSE_stack = rmse(valid_y, oof_stack_xgb)
    return stack_preds_xgb, RMSE_stack

def main(debug=False):
    #num_rows = 10000 if debug else None
    with timer("Run LightGBM with kfold"):
        Preds_lgb = kfold_lightgbm(train_df, test_df, num_folds=6, stratified=False, debug=debug)
    with timer("Run XGBoost"):
        Preds_xgb = kfold_xgboost(train_df, test_df, num_folds=6,stratified=False, debug = debug)
    with timer("Run CatBoost"):
        Preds_cat = kfold_catboost(train_df, test_df, num_folds = 6, debug = debug)
    with timer("Run stack"):
        Preds_stack = stacking(Preds_lgb[1],Preds_xgb[1],Preds_cat[1],Preds_lgb[0],Preds_xgb[0],Preds_cat[0])
    if not debug:
        ens2 = pd.read_csv("ens.csv")

        submission = pd.read_csv("sample_submission.csv")
        submission['target'] = (Preds_stack[0] + ens2['target'])/2
        submission.to_csv(submission_file_name, index = False)
        submission['target'] = Preds_lgb[0]
        submission.to_csv(str("LGB_")+submission_file_name, index = False)
        submission['target'] = Preds_xgb[0]
        submission.to_csv(str("XGB_")+submission_file_name, index = False)
        submission['target'] = Preds_cat[0]
        submission.to_csv(str("Cat_")+submission_file_name, index = False)
        submission['target'] = Preds_stack[0]
        submission.to_csv(str("Stack_")+submission_file_name, index = False)
        print("LGB RMSE: ", Preds_lgb[2])
        print("XGB RMSE: ", Preds_xgb[2])
        print("STK RMSE: ", Preds_stack[1])
        #print("CAT RMSE: ", Preds_cat[2])


if __name__ == "__main__":
    submission_file_name = "submission64.csv"
    with timer("Full model run"):
        main(debug=False)
