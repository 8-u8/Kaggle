# by https://www.kaggle.com/inversion/basic-feature-benchmark
# benchmark MAE 1.481
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

#%time
# loading TRAIN data
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(train.shape)

# data preparation
## functionized feature engineering.
## imspired by https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

def add_trend_feature(arr, abs_values=False): #by https://www.kaggle.com/andrekos/basic-feature-benchmark-with-quantiles
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]
    
def classic_sta_lta(x, length_sta, length_lta): #by https://www.kaggle.com/zikazika/useful-new-features-and-a-optimised-model
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta
    
## imspired by https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)

    #zc = np.fft.fft(xc)
    #print('adding basic features')
    X.loc[seg_id, 'mean'] = xc.mean()
    #X.loc[seg_id, 'median'] = xc.median()
    #X.loc[seg_id, 'mean_inversed']  =  1 / xc.mean()
    X.loc[seg_id, 'std']  = xc.std()
    X.loc[seg_id, 'max']  = xc.max()
    X.loc[seg_id, 'min']  = xc.min()
    #X.loc[seg_id, 'var']  = xc.var()
    X.loc[seg_id, 'skew'] = xc.skew()
    
    X.loc[seg_id, 'abs_max']  = np.abs(xc).max()
    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
    X.loc[seg_id, 'abs_std']  = np.abs(xc).std()
    
    X.loc[seg_id, 'iqr']      = np.subtract(*np.percentile(xc, [75, 25]))
    X.loc[seg_id, 'ave10']    = stats.trim_mean(xc, 0.1)

    #print('end: adding basic features. \n next: adding trend function')
    X.loc[seg_id, 'trend']     = add_trend_feature(xc)
    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)
    #print('end: adding trend function. \n next: quantile features')
    
    # quantile features
    # is it useful count encoding...?
    X.loc[seg_id, 'quantile_999'] = np.quantile(xc, 0.999)
    X.loc[seg_id, 'quantile_001'] = np.quantile(xc, 0.001)
    X.loc[seg_id, 'quantile_95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'quantile_99'] = np.quantile(xc, 0.99)
    X.loc[seg_id, 'quantile_05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'quantile_01'] = np.quantile(xc, 0.01)
    #print('end: quantile features \n next: autocorrelation.')
    #my original features...
    X.loc[seg_id, 'autocorr_lag1'] = xc.autocorr(lag = 1)
    X.loc[seg_id, 'autocorr_lag2'] = xc.autocorr(lag = 2)
    X.loc[seg_id, 'autocorr_lag3'] = xc.autocorr(lag = 3)
    X.loc[seg_id, 'autocorr_lag4'] = xc.autocorr(lag = 4)
    X.loc[seg_id, 'autocorr_lag5'] = xc.autocorr(lag = 5)
    
    X.loc[seg_id, 'autocorr_lag10'] = xc.autocorr(lag = 10)
    X.loc[seg_id, 'autocorr_lag50'] = xc.autocorr(lag = 50) # don't work significantly.
    
    X.loc[seg_id, 'autocorr_lag100'] = xc.autocorr(lag = 100)
    X.loc[seg_id, 'autocorr_lag200'] = xc.autocorr(lag = 200)
    
    X.loc[seg_id, 'autocorr_lag1000'] = xc.autocorr(lag = 1000)
    X.loc[seg_id, 'autocorr_lag1500'] = xc.autocorr(lag = 1500)
    
    #print('end: autocorrelation.  \n next: Hilbert transformation, and Hanns window function')
    # add the fft / rolled features.
    # Hirbelt transformation
    #X.loc[seg_id, 'Hilbert_Trns_mean']       = hilbert(xc).mean()
    X.loc[seg_id, 'Hilbert_Trns_abs_mean']   = np.abs(hilbert(xc)).mean()
    
    #X.loc[seg_id, 'Hilbert_Trns_mean_50000']     = np.mean(np.nonzero(hilbert(xc[:50000]))[0])
    #X.loc[seg_id, 'Hilbert_Trns_abs_mean_50000'] = np.mean(np.nonzero(np.abs(hilbert(xc[:50000])))[0])

    X.loc[seg_id, 'Hilbert_Trns_max']  = np.abs(hilbert(xc)).max() # don't work significantly(by importance)
    X.loc[seg_id, 'Hilbert_Trns_min']  = np.abs(hilbert(xc)).min()
    X.loc[seg_id, 'Hilbert_Trns_var']  = np.abs(hilbert(xc)).var() # don't work significantly
     
    # Hann's window
    X.loc[seg_id, 'Hann_window_mean_100']  = (convolve(xc, hann(100), mode='same') / sum(hann(100))).mean()
    X.loc[seg_id, 'Hann_window_mean_150']  = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()
    X.loc[seg_id, 'Hann_window_mean_200']  = (convolve(xc, hann(200), mode='same') / sum(hann(200))).mean()

    X.loc[seg_id, 'Hann_window_min_100']   = (convolve(xc, hann(100), mode='same') / sum(hann(100))).min()
    X.loc[seg_id, 'Hann_window_min_150']   = (convolve(xc, hann(150), mode='same') / sum(hann(150))).min()
    X.loc[seg_id, 'Hann_window_min_200']   = (convolve(xc, hann(200), mode='same') / sum(hann(200))).min()

    X.loc[seg_id, 'Hann_window_max_100']   = (convolve(xc, hann(100), mode='same') / sum(hann(100))).max()
    X.loc[seg_id, 'Hann_window_max_150']   = (convolve(xc, hann(150), mode='same') / sum(hann(150))).max()
    X.loc[seg_id, 'Hann_window_max_200']   = (convolve(xc, hann(200), mode='same') / sum(hann(200))).max()
    
    X.loc[seg_id, 'Hann_window_var_100']   = (convolve(xc, hann(100), mode='same') / sum(hann(100))).var()
    X.loc[seg_id, 'Hann_window_var_150']   = (convolve(xc, hann(150), mode='same') / sum(hann(150))).var()
    X.loc[seg_id, 'Hann_window_var_200']   = (convolve(xc, hann(200), mode='same') / sum(hann(200))).var()
    
    #print('end: Hilbert transformation, and Hanns window function. \n next: STA/LTA and other features')
    #STA / LTA
    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()
    X.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()
    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()
    X.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()
    X.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta(xc, 500, 20000).mean()
    
    
    #mean change ratio
    X.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000])  / xc[:50000 ][:-1]))[0])
    X.loc[seg_id, 'mean_change_rate_last_50000']  = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])
    X.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000])  / xc[:10000 ][:-1]))[0])
    X.loc[seg_id, 'mean_change_rate_last_10000']  = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])
    #print('all features caluclated!')



# traindata
seg_rows = 150_000
segments = int(np.floor(train.shape[0] / seg_rows))

X_train = pd.DataFrame(index    = range(segments),
                       dtype    = np.float64, 
                       columns  = ['mean', 'min', 'max', 'std'])
                        
y_train = pd.DataFrame(index    = range(segments),
                       dtype    = np.float64,
                       columns  = ['time_to_failure'])


for segment in range(segments):
    seg = train.iloc[segment * seg_rows : segment * seg_rows + seg_rows]
    create_features(segment, seg, X_train)
    y   = seg['time_to_failure'].values[-1]
    y_train.loc[segment, 'time_to_failure'] = y
    
X_train['max']  = X_train['max'].round()
X_train['Hilbert_count_max'] = X_train.groupby('max')['Hilbert_Trns_abs_mean'].transform('count')
X_train['Hilbert_order_max'] = X_train.groupby('max')['Hilbert_Trns_abs_mean'].count().rank(ascending = False)

#X_train['max_change_rate_count'] = X_train.groupby('max')['mean_change_rate_first_10000'].transform('count')
#X_train['max_change_rate_order'] = X_train.groupby('max')['mean_change_rate_first_10000'].count().rank(ascending = False)

X_train['max_autolag_100_count'] = X_train.groupby('max')['autocorr_lag100'].transform('count')
X_train['max_autolag_100_order'] = X_train.groupby('max')['autocorr_lag100'].count().rank(ascending = False)


#X_train['min']               = X_train['min'].round()
#X_train['Hilbert_count_min'] = X_train.groupby('min')['Hilbert_Trns_abs_mean'].transform('count')
#X_train['Hilbert_order_min'] = X_train.groupby('min')['Hilbert_Trns_abs_mean'].count().rank(ascending = False).describe()

#type(X_train['Hilbert_Trns_mean_50000'])

# test data preparation
submission  = pd.read_csv("../input/sample_submission.csv", index_col = 'seg_id')
X_test      = pd.DataFrame(columns = X_train.columns,
                          dtype   = np.float64,
                          index   = submission.index)

for seg_id in X_test.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_features(seg_id, seg, X_test)
X_test['max']  = X_test['max'].round()
X_test['Hilbert_count_max']   = X_test.groupby('max')['Hilbert_Trns_abs_mean'].transform('count')
X_test['Hilbert_order_max']   = X_test.groupby('max')['Hilbert_Trns_abs_mean'].count().rank(ascending = False)

X_test['max_autolag_100_count'] = X_test.groupby('max')['autocorr_lag100'].transform('count')
X_test['max_autolag_100_order'] = X_test.groupby('max')['autocorr_lag100'].count().rank(ascending = False)

#X_test['max_change_rate_count'] = X_test.groupby('max')['mean_change_rate_first_10000'].transform('count')
#X_test['max_change_rate_order'] = X_test.groupby('max')['mean_change_rate_first_10000'].count().rank(ascending = False)



#X_test['min']  = X_test['min'].round()
#X_test['Hilbert_count_min']   = X_test.groupby('max')['Hilbert_Trns_abs_mean'].transform('count')
#X_test['Hilbert_order_min']   = X_test.groupby('max')['Hilbert_Trns_abs_mean'].count().rank(ascending = False)


#Generalization
print(X_train.columns)
scaler         = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
print(X_train_scaled.describe())
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
print(X_test.shape)



# Support Vector Regression.
#svm = NuSVR()
#svm.fit(X_train_scaled,y_train.values.flatten())
print(X_train_scaled.columns)
print(X_test_scaled.columns)
del train

# LightGBM
n_fold        = 10
folds         = KFold(n_splits=n_fold, shuffle=True, random_state=42)
train_columns = X_train_scaled.columns.values

params = {'num_leaves': 51,
         'min_data_in_leaf': 10, 
         'objective':'gamma',
         'max_depth': -1,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": -1,
         "random_state": 42}
         
oof                   = np.zeros(len(X_train_scaled))
predictions           = np.zeros(len(X_test_scaled))
feature_importance_df = pd.DataFrame()
#run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled,y_train.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    
    X_tr, X_val = X_train_scaled.iloc[trn_idx], X_train_scaled.iloc[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx]       , y_train.iloc[val_idx]

    model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)
    model.fit(X_tr, 
              y_tr, 
              eval_set=[(X_tr, y_tr), (X_val, y_val)], 
              eval_metric='mae',
              verbose=1000, 
              early_stopping_rounds=500)
    oof[val_idx] = model.predict(X_val, num_iteration = model.best_iteration_)
    #feature importance
    fold_importance_df               = pd.DataFrame()
    fold_importance_df["Feature"]    = train_columns
    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
    fold_importance_df["fold"]       = fold_ + 1
    feature_importance_df            = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)
    #predictions
    #predictions += np.exp(model.predict(X_test_scaled, num_iteration=model.best_iteration_) / folds.n_splits)
    predictions += model.predict(X_test_scaled, num_iteration=model.best_iteration_) / folds.n_splits
    #print(np.sum(y_val - oof[val_idx])/len(y_val))
for i in range(0,10):
    print("The MAE is: \n "+str(np.sum(np.abs(y_val.values-oof[val_idx][i]))/len(y_val.values)))
    
feature_importance_df.sort_values(by = "importance", ascending = False)
#submission['time_to_failure'] = svm.predict(X_test_scaled)
submission['time_to_failure'] = predictions
submission.to_csv('simple_LGBM_submission.csv')
feature_importance_df.to_csv('feature_importance.csv')