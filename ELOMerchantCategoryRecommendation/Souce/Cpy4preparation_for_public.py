import logging
import os
import gc
import time
from datetime import datetime as dt

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from contextlib import contextmanager


warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s: %(lineno)d: %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

DATE_TODAY = dt(2019, 1, 26)






FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']



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
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

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

    return df
num_rows = None
# preprocessing train & test
def process_main_df(df):
    
    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (DATE_TODAY - df['first_active_month']).dt.days

    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    for f in feature_cols:    
        df['days_' + f] = df['elapsed_time'] * df[f]
        df['days_' + f + '_ratio'] = df[f] / df['elapsed_time']

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    df_feats = df.reindex(columns=feature_cols)
    df['features_sum'] = df_feats.sum(axis=1)
    df['features_mean'] = df_feats.mean(axis=1)
    df['features_max'] = df_feats.max(axis=1)
    df['features_min'] = df_feats.min(axis=1)
    df['features_var'] = df_feats.std(axis=1)
    df['features_prod'] = df_feats.product(axis=1)

    return df

def train_test(num_rows=None):

    def read_csv(filename):
        df = pd.read_csv(
            filename, index_col=['card_id'], parse_dates=['first_active_month'], nrows=num_rows)
        return df
    
    # load csv
    train_df = read_csv('train.csv')
    test_df = read_csv('test.csv') 
    logger.info("samples: train {}, test: {}".format(train_df.shape, test_df.shape))

    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30., 'outliers'] = 1

    train_df = reduce_mem_usage(process_main_df(train_df))
    test_df = reduce_mem_usage(process_main_df(test_df))

    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    for f in feature_cols:
        order_label = train_df.groupby([f])['outliers'].mean()
        train_df[f] = train_df[f].map(order_label)
        test_df[f] = test_df[f].map(order_label)    

    return train_df, test_df

def process_date(df):
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['month'] = df['purchase_date'].dt.month
    df['day'] = df['purchase_date'].dt.day
    df['hour'] = df['purchase_date'].dt.hour
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['weekday'] = df['purchase_date'].dt.weekday
    df['weekend'] = (df['purchase_date'].dt.weekday >= 5).astype(int)
    return df


def dist_holiday(df, col_name, date_holiday, date_ref, period=100):
    df[col_name] = np.maximum(np.minimum((pd.to_datetime(date_holiday) - df[date_ref]).dt.days, period), 0)


def historical_transactions(num_rows=None):
    """
    preprocessing historical transactions
    """
    na_dict = {
        'category_2': 1.,
        'category_3': 'A',
        'merchant_id': 'M_ID_00a6ca8a8a',
    }

    holidays = [
        ('Christmas_Prep_2017', '2017-11-01'), # Christmas preparation November 01 2017
        ('Christmas_Eve_2017', '2017-12-24'), # Christmas Eve: December 24 2017
        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
        ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
        ('Mothers_Day_2018', '2018-05-13'),
    ]

    # agg
    aggs = dict()
    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    aggs.update({col: ['nunique'] for col in col_unique})

    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']
    aggs.update({col: ['nunique', 'mean', 'min', 'max'] for col in col_seas})

    aggs_specific = {
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
        'installments': ['sum', 'max', 'mean', 'var', 'skew'],
        'purchase_date': ['max', 'min'],
        'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
        'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
        'authorized_flag': ['mean'],
        'weekend': ['mean'], # overwrite
        'weekday': ['mean'], # overwrite
        'day': ['nunique', 'mean', 'min'], # overwrite
        'category_1': ['mean'],
        'category_2': ['mean'],
        'category_3': ['mean'],
        'card_id': ['size', 'count'],
        'price': ['sum', 'mean', 'max', 'min', 'var'],
        'Christmas_Day_2017': ['mean', 'sum','var','max'],
        'Mothers_Day_2017': ['mean', 'sum','var','max'],
        'fathers_day_2017': ['mean', 'sum','var','max'],
        'Children_day_2017': ['mean', 'sum','var','max'],
        'Valentine_Day_2017': ['mean', 'sum','var','max'],
        'Black_Friday_2017': ['mean', 'sum','var','max'],
        'Mothers_Day_2018': ['mean', 'sum','var','max'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
    }
    aggs.update(aggs_specific)


    # starting to process
    # load csv
    df = pd.read_csv('historical_transactions.csv', nrows=num_rows)
    logger.info('read historical_transactions {}'.format(df.shape))
    print("historical transaction has read...")
    # fillna
    df.fillna(na_dict, inplace=True)
    df['installments'].replace({
        -1: np.nan, 999: np.nan}, inplace=True)

    # trim
    df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(np.int16)
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype(np.int16)
    df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C':2}).astype(np.int16)

    # additional features
    df['price'] = df['purchase_amount'] / df['installments']
    #df['observation_date'] = df.apply(lambda x: x['new_purchase_date'] - pd.DataOffset(months = x['new_month_lag']-1), axis = 1)
    # datetime features
    df = process_date(df)

    # holidays
    for d_name, d_day in holidays:
        dist_holiday(df, d_name, d_day, 'purchase_date')

    df['month_diff'] = (DATE_TODAY - df['purchase_date']).dt.days // 30
    df['month_diff'] += df['month_lag']

    # additional features
    df['duration'] = df['purchase_amount'] * df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

    # reduce memory usage
    df = reduce_mem_usage(df)

    for col in ['category_2', 'category_3']:
        df[col + '_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df[col + '_min'] = df.groupby([col])['purchase_amount'].transform('min')
        df[col + '_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df[col + '_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']
    df.head()
    df = df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])
    df.columns = ['hist_' + c for c in df.columns]

    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']

    df['hist_purchase_date_diff'] = (df['hist_purchase_date_max'] - df['hist_purchase_date_min']).dt.days
    df['hist_purchase_date_average'] = df['hist_purchase_date_diff'] / df['hist_card_id_size']
    df['hist_purchase_date_uptonow'] = (DATE_TODAY - df['hist_purchase_date_max']).dt.days
    df['hist_purchase_date_uptomin'] = (DATE_TODAY - df['hist_purchase_date_min']).dt.days

    # reduce memory usage
    df = reduce_mem_usage(df)

    return df


def new_merchant_transactions(num_rows=None):
    """
    preprocessing new_merchant_transactions
    """
    na_dict = {
        'category_2': 1.,
        'category_3': 'A',
        'merchant_id': 'M_ID_00a6ca8a8a',
    }

    holidays = [
        ('Christmas_Prep_2017', '2017-11-01'),  # Christmas Preparation: November 01 2017
        ('Christmas_Eve_2017', '2017-12-24'),  # Christmas Eve: December 24 2017
        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
         ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
         ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
         ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
        ('Mothers_Day_2018', '2018-05-13'),
    ]
    
    aggs = dict()
    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    aggs.update({col: ['nunique'] for col in col_unique})

    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']
    aggs.update({col: ['nunique', 'mean', 'min', 'max'] for col in col_seas})

    aggs_specific = {
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
        'installments': ['sum', 'max', 'mean', 'var', 'skew'],
        'purchase_date': ['max', 'min'],
        'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
        'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
        'authorized_flag': ['mean'],
        'weekend': ['mean'], # overwrite
        'weekday': ['mean'], # overwrite
        'day': ['nunique', 'mean', 'min'], # overwrite
        'category_1': ['mean'],
        'category_2': ['mean'],
        'category_3': ['mean'],
        'card_id': ['size', 'count'],
        'price': ['sum', 'mean', 'max', 'min', 'var'],
        'Christmas_Day_2017': ['mean', 'sum','var','max'],
        'fathers_day_2017': ['mean', 'sum','var','max'],
        'Mothers_Day_2017': ['mean', 'sum','var','max'],
        'Children_day_2017': ['mean', 'sum','var','max'],
        'fathers_day_2017': ['mean', 'sum','var','max'],
        'Black_Friday_2017': ['mean', 'sum','var','max'],
        'Valentine_Day_2017': ['mean', 'sum','var','max'],
        'Mothers_Day_2018': ['mean', 'sum','var','max'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew']#,
        #'observation_date':['max']
    }
    aggs.update(aggs_specific)

    # load csv
    df = pd.read_csv('new_merchant_transactions.csv', nrows=num_rows)
    logger.info('read new_merchant_transactions {}'.format(df.shape))
    
    # fillna
    df.fillna(na_dict, inplace=True)
    df['installments'].replace({
        -1: np.nan, 999: np.nan}, inplace=True)

    # trim
    df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
    df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2}).astype(int).astype(np.int16)

    # additional features
    df['price'] = df['purchase_amount'] / df['installments']
    #df['observation_date'] = df.apply(lambda x: x['new_purchase_date'] - pd.DataOffset(months = x['new_month_lag']-1), axis = 1)
    # datetime features
    df = process_date(df)
    for d_name, d_day in holidays:
        dist_holiday(df, d_name, d_day, 'purchase_date')

    df['month_diff'] = (DATE_TODAY - df['purchase_date']).dt.days // 30
    df['month_diff'] += df['month_lag']

    # additional features
    df['duration'] = df['purchase_amount'] * df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

    # reduce memory usage
    df = reduce_mem_usage(df)

    for col in ['category_2', 'category_3']:
        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df[col+'_min'] = df.groupby([col])['purchase_amount'].transform('min')
        df[col+'_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df[col+'_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    df = df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])
    df.columns = ['new_' + c for c in df.columns]

    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    
    df['new_purchase_date_diff'] = (df['new_purchase_date_max'] - df['new_purchase_date_min']).dt.days
    df['new_purchase_date_average'] = df['new_purchase_date_diff'] / df['new_card_id_size']
    df['new_purchase_date_uptonow'] = (DATE_TODAY - df['new_purchase_date_max']).dt.days
    df['new_purchase_date_uptomin'] = (DATE_TODAY - df['new_purchase_date_min']).dt.days

    # reduce memory usage
    df = reduce_mem_usage(df)

    return df
        

# additional features
def additional_features(df):
    
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days

    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    date_features = [
        'hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min']
    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    #
    df['card_id_total'] = df['new_card_id_size'] + df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']
    
    df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
    #df['purchase_amount_ratio'] = df['new_purchase_amount_sum'] / df['hist_purchase_amount_sum']

    df['purchase_amount_total_ratio'] = df['new_purchase_amount_sum']/(df['hist_purchase_amount_sum'])
    df['purchase_amount_mean_ratio'] = df['new_purchase_amount_mean']/(df['hist_purchase_amount_mean'])
    df['purchase_amount_max_ratio'] = df['new_purchase_amount_max']/(df['hist_purchase_amount_max'])
    df['purchase_amount_min_ratio'] = df['new_purchase_amount_min']/(df['hist_purchase_amount_min'])

    df['installments_total'] = df['new_installments_sum'] + df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean'] + df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max'] + df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']

    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']

    #
    df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
    
    df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
    df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
    df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']
        
    df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
    df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']
    df['duration_max'] = df['new_duration_var'] + df['hist_duration_var']
    df['duration_max'] = df['new_duration_skew'] + df['hist_duration_skew']

    df['month_lag_mean_ratio'] = df['new_month_lag_mean'] / df['hist_month_lag_mean']
    df['month_lag_max_ratio'] = df['new_month_lag_max'] / df['hist_month_lag_max']
    df['month_lag_min_ratio'] = df['new_month_lag_min'] / df['hist_month_lag_min']
    df['category_1_mean_ratio'] = df['new_category_1_mean'] / df['hist_category_1_mean']
        
    df['duration_mean_ratio'] = df['new_duration_mean'] / df['hist_duration_mean']
    df['duration_min_ratio'] = df['new_duration_min'] / df['hist_duration_min']
    df['duration_max_ratio'] = df['new_duration_max'] / df['hist_duration_max']
    df['duration_max_ratio'] = df['new_duration_var'] / df['hist_duration_var']
    df['duration_max_ratio'] = df['new_duration_skew'] / df['hist_duration_skew']

    df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean'] + df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min'] = df['new_amount_month_ratio_min'] + df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max'] = df['new_amount_month_ratio_max'] + df['hist_amount_month_ratio_max']

    df['amount_month_ratio_mean_ratio'] = df['new_amount_month_ratio_mean'] / df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min_ratio'] = df['new_amount_month_ratio_min'] / df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max_ratio'] = df['new_amount_month_ratio_max'] / df['hist_amount_month_ratio_max']
    
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']
    df['CLV_sq'] = df['new_CLV'] * df['hist_CLV']
    
    #['mean', 'sum','var','max'],
    df['Christmas_Day_2017_mean_ratio'] = df['new_Christmas_Day_2017_mean']/df['hist_Christmas_Day_2017_mean']
    df['Christmas_Day_2017_sum_ratio']  = df['new_Christmas_Day_2017_sum']/df['hist_Christmas_Day_2017_sum']
    df['Christmas_Day_2017_var_ratio']  = df['new_Christmas_Day_2017_var']/df['hist_Christmas_Day_2017_var']
    df['Christmas_Day_2017_max_ratio']  = df['new_Christmas_Day_2017_max']/df['hist_Christmas_Day_2017_max'] 

    df['Mothers_Day_2017_mean_ratio'] = df['new_Mothers_Day_2017_mean']/df['hist_Mothers_Day_2017_mean']
    df['Mothers_Day_2017_sum_ratio'] = df['new_Mothers_Day_2017_sum']/df['hist_Mothers_Day_2017_sum']
    df['Mothers_Day_2017_var_ratio'] = df['new_Mothers_Day_2017_var']/df['hist_Mothers_Day_2017_var']
    df['Mothers_Day_2017_max_ratio'] = df['new_Mothers_Day_2017_max']/df['hist_Mothers_Day_2017_max']

    df['fathers_day_2017_mean_ratio'] = df['new_fathers_day_2017_mean']/df['hist_fathers_day_2017_mean']
    df['fathers_day_2017_sum_ratio'] = df['new_fathers_day_2017_sum']/df['hist_fathers_day_2017_sum']
    df['fathers_day_2017_var_ratio'] = df['new_fathers_day_2017_var']/df['hist_fathers_day_2017_var']
    df['fathers_day_2017_max_ratio'] = df['new_fathers_day_2017_max']/df['hist_fathers_day_2017_max']

    df['Children_day_2017_mean_ratio'] = df['new_Children_day_2017_mean']/df['hist_Children_day_2017_mean']
    df['Children_day_2017_sum_ratio'] = df['new_Children_day_2017_sum']/df['hist_Children_day_2017_sum']
    df['Children_day_2017_var_ratio'] = df['new_Children_day_2017_var']/df['hist_Children_day_2017_var']
    df['Children_day_2017_max_ratio'] = df['new_Children_day_2017_max']/df['hist_Children_day_2017_max']

    df['Valentine_Day_2017_mean_ratio'] = df['new_Valentine_Day_2017_mean']/df['hist_Valentine_Day_2017_mean']
    df['Valentine_Day_2017_sum_ratio'] = df['new_Valentine_Day_2017_sum']/df['hist_Valentine_Day_2017_sum']
    df['Valentine_Day_2017_var_ratio'] = df['new_Valentine_Day_2017_var']/df['hist_Valentine_Day_2017_var']
    df['Valentine_Day_2017_max_ratio'] = df['new_Valentine_Day_2017_max']/df['hist_Valentine_Day_2017_max']

    df['Black_Friday_2017_mean_ratio'] = df['new_Black_Friday_2017_mean']/df['hist_Black_Friday_2017_mean']
    df['Black_Friday_2017_sum_ratio'] = df['new_Black_Friday_2017_sum']/df['hist_Black_Friday_2017_sum']
    df['Black_Friday_2017_var_ratio'] = df['new_Black_Friday_2017_var']/df['hist_Black_Friday_2017_var']
    df['Black_Friday_2017_max_ratio'] = df['new_Black_Friday_2017_max']/df['hist_Black_Friday_2017_max']

    df['Mothers_Day_2018_mean_ratio'] = df['new_Mothers_Day_2018_mean']/df['hist_Mothers_Day_2018_mean']
    df['Mothers_Day_2018_sum_ratio'] = df['new_Mothers_Day_2018_sum']/df['hist_Mothers_Day_2018_sum']
    df['Mothers_Day_2018_var_ratio'] = df['new_Mothers_Day_2018_var']/df['hist_Mothers_Day_2018_var']
    df['Mothers_Day_2018_max_ratio'] = df['new_Mothers_Day_2018_max']/df['hist_Mothers_Day_2018_max']

    df['Christmas_Day_2017_mean_sum'] = df['new_Christmas_Day_2017_mean']+df['hist_Christmas_Day_2017_mean']
    df['Christmas_Day_2017_sum_sum']  = df['new_Christmas_Day_2017_sum']+df['hist_Christmas_Day_2017_sum']
    df['Christmas_Day_2017_var_sum']  = df['new_Christmas_Day_2017_var']+df['hist_Christmas_Day_2017_var']
    df['Christmas_Day_2017_max_sum']  = df['new_Christmas_Day_2017_max']+df['hist_Christmas_Day_2017_max'] 

    df['Mothers_Day_2017_mean_sum'] = df['new_Mothers_Day_2017_mean']+df['hist_Mothers_Day_2017_mean']
    df['Mothers_Day_2017_sum_sum'] = df['new_Mothers_Day_2017_sum']+df['hist_Mothers_Day_2017_sum']
    df['Mothers_Day_2017_var_sum'] = df['new_Mothers_Day_2017_var']+df['hist_Mothers_Day_2017_var']
    df['Mothers_Day_2017_max_sum'] = df['new_Mothers_Day_2017_max']+df['hist_Mothers_Day_2017_max']

    df['fathers_day_2017_mean_sum'] = df['new_fathers_day_2017_mean']+df['hist_fathers_day_2017_mean']
    df['fathers_day_2017_sum_sum'] = df['new_fathers_day_2017_sum']+df['hist_fathers_day_2017_sum']
    df['fathers_day_2017_var_sum'] = df['new_fathers_day_2017_var']+df['hist_fathers_day_2017_var']
    df['fathers_day_2017_max_sum'] = df['new_fathers_day_2017_max']+df['hist_fathers_day_2017_max']

    df['Children_day_2017_mean_sum'] = df['new_Children_day_2017_mean']+df['hist_Children_day_2017_mean']
    df['Children_day_2017_sum_sum'] = df['new_Children_day_2017_sum']+df['hist_Children_day_2017_sum']
    df['Children_day_2017_var_sum'] = df['new_Children_day_2017_var']+df['hist_Children_day_2017_var']
    df['Children_day_2017_max_sum'] = df['new_Children_day_2017_max']+df['hist_Children_day_2017_max']

    df['Valentine_Day_2017_mean_sum'] = df['new_Valentine_Day_2017_mean']+df['hist_Valentine_Day_2017_mean']
    df['Valentine_Day_2017_sum_sum'] = df['new_Valentine_Day_2017_sum']+df['hist_Valentine_Day_2017_sum']
    df['Valentine_Day_2017_var_sum'] = df['new_Valentine_Day_2017_var']+df['hist_Valentine_Day_2017_var']
    df['Valentine_Day_2017_max_sum'] = df['new_Valentine_Day_2017_max']+df['hist_Valentine_Day_2017_max']

    df['Black_Friday_2017_mean_sum'] = df['new_Black_Friday_2017_mean']+df['hist_Black_Friday_2017_mean']
    df['Black_Friday_2017_sum_sum'] = df['new_Black_Friday_2017_sum']+df['hist_Black_Friday_2017_sum']
    df['Black_Friday_2017_var_sum'] = df['new_Black_Friday_2017_var']+df['hist_Black_Friday_2017_var']
    df['Black_Friday_2017_max_sum'] = df['new_Black_Friday_2017_max']+df['hist_Black_Friday_2017_max']

    df['Mothers_Day_2018_mean_sum'] = df['new_Mothers_Day_2018_mean']+df['hist_Mothers_Day_2018_mean']
    df['Mothers_Day_2018_sum_sum'] = df['new_Mothers_Day_2018_sum']+df['hist_Mothers_Day_2018_sum']
    df['Mothers_Day_2018_var_sum'] = df['new_Mothers_Day_2018_var']+df['hist_Mothers_Day_2018_var']
    df['Mothers_Day_2018_max_sum'] = df['new_Mothers_Day_2018_max']+df['hist_Mothers_Day_2018_max']

    df['Christmas_Day_2017_mean_diff'] = df['new_Christmas_Day_2017_mean']-df['hist_Christmas_Day_2017_mean']
    df['Christmas_Day_2017_sum_diff']  = df['new_Christmas_Day_2017_sum']-df['hist_Christmas_Day_2017_sum']
    df['Christmas_Day_2017_var_diff']  = df['new_Christmas_Day_2017_var']-df['hist_Christmas_Day_2017_var']
    df['Christmas_Day_2017_max_diff']  = df['new_Christmas_Day_2017_max']-df['hist_Christmas_Day_2017_max'] 

    df['Mothers_Day_2017_mean_diff'] = df['new_Mothers_Day_2017_mean']-df['hist_Mothers_Day_2017_mean']
    df['Mothers_Day_2017_sum_diff'] = df['new_Mothers_Day_2017_sum']-df['hist_Mothers_Day_2017_sum']
    df['Mothers_Day_2017_var_diff'] = df['new_Mothers_Day_2017_var']-df['hist_Mothers_Day_2017_var']
    df['Mothers_Day_2017_max_diff'] = df['new_Mothers_Day_2017_max']-df['hist_Mothers_Day_2017_max']

    df['fathers_day_2017_mean_diff'] = df['new_fathers_day_2017_mean']-df['hist_fathers_day_2017_mean']
    df['fathers_day_2017_sum_diff'] = df['new_fathers_day_2017_sum']-df['hist_fathers_day_2017_sum']
    df['fathers_day_2017_var_diff'] = df['new_fathers_day_2017_var']-df['hist_fathers_day_2017_var']
    df['fathers_day_2017_max_diff'] = df['new_fathers_day_2017_max']-df['hist_fathers_day_2017_max']

    df['Children_day_2017_mean_diff'] = df['new_Children_day_2017_mean']-df['hist_Children_day_2017_mean']
    df['Children_day_2017_sum_diff'] = df['new_Children_day_2017_sum']-df['hist_Children_day_2017_sum']
    df['Children_day_2017_var_diff'] = df['new_Children_day_2017_var']-df['hist_Children_day_2017_var']
    df['Children_day_2017_max_diff'] = df['new_Children_day_2017_max']-df['hist_Children_day_2017_max']

    df['Valentine_Day_2017_mean_diff'] = df['new_Valentine_Day_2017_mean']-df['hist_Valentine_Day_2017_mean']
    df['Valentine_Day_2017_sum_diff'] = df['new_Valentine_Day_2017_sum']-df['hist_Valentine_Day_2017_sum']
    df['Valentine_Day_2017_var_diff'] = df['new_Valentine_Day_2017_var']-df['hist_Valentine_Day_2017_var']
    df['Valentine_Day_2017_max_diff'] = df['new_Valentine_Day_2017_max']-df['hist_Valentine_Day_2017_max']

    df['Black_Friday_2017_mean_diff'] = df['new_Black_Friday_2017_mean']-df['hist_Black_Friday_2017_mean']
    df['Black_Friday_2017_sum_diff'] = df['new_Black_Friday_2017_sum']-df['hist_Black_Friday_2017_sum']
    df['Black_Friday_2017_var_diff'] = df['new_Black_Friday_2017_var']-df['hist_Black_Friday_2017_var']
    df['Black_Friday_2017_max_diff'] = df['new_Black_Friday_2017_max']-df['hist_Black_Friday_2017_max']

    df['Mothers_Day_2018_mean_diff'] = df['new_Mothers_Day_2018_mean']-df['hist_Mothers_Day_2018_mean']
    df['Mothers_Day_2018_sum_diff'] = df['new_Mothers_Day_2018_sum']-df['hist_Mothers_Day_2018_sum']
    df['Mothers_Day_2018_var_diff'] = df['new_Mothers_Day_2018_var']-df['hist_Mothers_Day_2018_var']
    df['Mothers_Day_2018_max_diff'] = df['new_Mothers_Day_2018_max']-df['hist_Mothers_Day_2018_max']

    df = reduce_mem_usage(df)

    return df

def main(debug=False):
    num_rows = 10000 if debug else None
    
    with timer("historical transactions"):
        hist_df = historical_transactions(num_rows)
        
    with timer("new merchants"):
        new_merchant_df = new_merchant_transactions(num_rows)
        
    with timer("additional features"):
        df = pd.concat([new_merchant_df, hist_df], axis=1)
        del new_merchant_df, hist_df
        gc.collect()

        train_df, test_df = train_test(num_rows)
        train_df = train_df.join(df, how='left', on='card_id')
        test_df = test_df.join(df, how='left', on='card_id')
        del df
        gc.collect()

        train_df = additional_features(train_df)
        test_df = additional_features(test_df)
        df = train_df.append(test_df)
        df.replace([np.inf, -np.inf],0)
        df.replace(np.nan, 0)
    with timer("writing csv..."):
        df.to_csv("output\\mergedFulldata(By_Tsai)_8.csv")
    #with timer("split train & test"):
    #    train_df = df[df['target'].notnull()]
    #    test_df = df[df['target'].isnull()]
    #    del df
    #    gc.collect()
    #with timer("Run LightGBM with kfold"):
    #    kfold_lightgbm(train_df, test_df, num_folds=15, stratified=False, debug=debug)

if __name__ == "__main__":
    #submission_file_name = "submission27.csv"
    with timer("Full model run"):
        main(debug=False)
