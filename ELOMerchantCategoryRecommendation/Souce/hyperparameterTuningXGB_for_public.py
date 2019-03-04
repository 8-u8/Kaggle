import optuna
#import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


print("csv loading...")
UseData = pd.read_csv("Here is aggregated data")
train   = pd.read_csv("train.csv")
test   = pd.read_csv("test.csv")
print("csv read.")
train_df = UseData[UseData['card_id'].isin(train['card_id'])]
print(train_df.shape)
test_df  = UseData[UseData['card_id'].isin(test['card_id'])]
print(test_df.shape)

# by https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78903
#train_df['rounded_target'] = train_df['target'].round(0)
#train_df = train_df.sort_values('rounded_target').reset_index(drop=True)
#vc = train_df['rounded_target'].value_counts()
#vc = dict(sorted(vc.items()))
#df = pd.DataFrame()
#train_df['indexcol'],i = 0,1
#for k,v in vc.items():
#    step = train_df.shape[0]/v
#    indent = train_df.shape[0]/(v+1)
#    df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
#    for j in range(0, v):
#        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
#    df = pd.concat([df2,df])
#    i+=1
#train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
#del train_df['indexcol'], train_df['rounded_target']

#folds = KFold(n_splits = 5, shuffle = True, random_state= 114)
FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']
feats = [f for f in UseData.columns if f not in FEATS_EXCLUDED]

#preds_lgb = np.zeros(train_df.shape[0])
#sub_preds = np.zeros(test_df.shape[0])
#oof_preds_xgb = np.zeros()
#sub_preds_xgb = np.zeros()
#feature_importance_df = pd.DataFrame()
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def objective(trial):
    #Folds_xgb = KFold(n_splits= 5, shuffle=False, random_state=514)
    #for fold_, (train_idx, valid_idx) in enumerate(Folds_xgb.split(train_df[feats].values, train_df['outliers'].values)):
    train_x, train_y = train_df[feats], train_df['target']
    valid_x, valid_y = train_df[feats], train_df['target']
    #train_x = train_x.values
    #train_y = pd.Series(train_y)
    #valid_x = valid_x.values
    #valid_y = valid_y.values
    # set data structure
    #print("fold n °{}".format(fold_))
    xgb_train = xgb.DMatrix(train_x,label=train_y)
    #xgb_valid = xgb.DMatrix(valid_x,label=valid_y)
    # params optimized by optuna
    xgb_params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'gamma': trial.suggest_uniform('gamma', 1e-8, 10),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'eval_metric':'rmse',
            'lambda': trial.suggest_uniform('lambda', 1e-8,1.00),
            'subsample': trial.suggest_uniform('subsample', 1e-8, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree',0.01,1.0),
            'min_child_weight': trial.suggest_uniform('min_child_weight', 1e-8, 50),
            'verbosity': 1,
            'eta': trial.suggest_uniform('eta', 0, 0.5),
            'alpha': trial.suggest_uniform('alpha', 0, 10),
            'seed': int(2**5),
            'nthread': 9
            }

    reg_xgb = xgb.train(
                    xgb_params,
                    xgb_train
                    )

    preds_xgb = reg_xgb.predict(xgb.DMatrix(valid_x),ntree_limit=reg_xgb.best_ntree_limit)
    #pred_labels = np.rint(preds_lgb)

    RMSE = sklearn.metrics.mean_squared_error(valid_y, preds_xgb)
    return RMSE

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))
    
    print("best trial:")
    trial = study.best_trial

    print('value:{}'.format(trial.value))
    
    print("params:")
    for key, value in trial.params.items():
        print('{}:{}'.format(key, value))