import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

UseData = pd.read_csv("here is aggregated table")
train   = pd.read_csv("train.csv")
test   = pd.read_csv("test.csv")

train_df = UseData[UseData['card_id'].isin(train['card_id'])]
train_df.shape
test_df  = UseData[UseData['card_id'].isin(test['card_id'])]
test_df.shape

# by https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78903
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

folds = KFold(n_splits = 5, shuffle = True, random_state= 114)
FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED','month_0']
feats = [f for f in UseData.columns if f not in FEATS_EXCLUDED]

#preds_lgb = np.zeros(train_df.shape[0])
#sub_preds = np.zeros(test_df.shape[0])
#oof_preds_xgb = np.zeros()
#sub_preds_xgb = np.zeros()
#feature_importance_df = pd.DataFrame()

def objective(trial):
    
    train_x, train_y = train_df[feats], train_df['target']
    valid_x, valid_y = train_df[feats], train_df['target']
    #train_x = train_x.values
    #train_y = pd.Series(train_y)
    #valid_x = valid_x.values
    #valid_y = valid_y.values
    # set data structure
    lgb_train = lgb.Dataset(train_x,label=train_y)
    # params optimized by optuna
    params ={
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'rmse',
    'learning_rate': trial.suggest_uniform('learning_rate',1e-8,1.0),
    'subsample': trial.suggest_uniform('subsample', 1e-8, 1.0),
    'max_depth': trial.suggest_int('max_depth', 1, 20),
    #'top_rate': trial.suggest_uniform('top_rate', 0.1, 1.0),
    #'num_leaves': trial.suggest_int('num_leaves', 10, 100),
    'min_child_weight': trial.suggest_uniform('min_child_weight', 1e-3, 100),
    'reg_alpha': trial.suggest_uniform('reg_alpha', 0.1,50),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.01,1.0),
    'min_split_gain': trial.suggest_uniform('min_split_gain',1.0,10.0),
    'reg_lambda': trial.suggest_uniform('reg_lambda', 1e-8, 1.0),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1,20),
    'cat_l2' : trial.suggest_uniform('cat_l2', 0,15.0),
    'subsample_for_bin'
    #'alpha' : trial.suggest_uniform('alpha', 0, 50),
    #'bagging_freq':trial.suggest_int('bagging_freq',0,100),
    'verbose': -1   
    }
    params['num_leaves'] = trial.suggest_int('num_leaves', 1, params['max_depth']*10)
    params['top_rate'] = trial.suggest_uniform('top_rate',0.0,1.0)
    params['other_rate'] = trial.suggest_uniform('other_rate',0.0,1.0 - params['top_rate'])
    reg_lgb = lgb.train(
                    params,
                    lgb_train
                    )

    preds_lgb = reg_lgb.predict(valid_x)
    #pred_labels = np.rint(preds_lgb)
    
    RMSE = sklearn.metrics.mean_squared_error(valid_y, preds_lgb)
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