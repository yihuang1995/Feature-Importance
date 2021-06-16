import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import lightgbm as lgb
from lightgbm import LGBMClassifier
import copy 


def importance_plot(importance_dict):
    col_name = importance_dict.keys()
    col_importance = importance_dict.values()
    importance_df = pd.DataFrame()
    importance_df['col_name'] = col_name
    importance_df['col_importance'] = col_importance
    importance_df = importance_df.sort_values(by=['col_importance'],ascending=False)
    sns.set_context('paper')
    f, ax = plt.subplots(figsize = (6,15))
    sns.set_color_codes('pastel')
    sns.barplot(x = 'col_importance', y = 'col_name', data = importance_df,
                label = 'Total', color = 'b', edgecolor = 'w')
    sns.set_color_codes('muted')
    sns.despine(left = True, bottom = True)
    plt.show()

def spearman_ranking(df,target,absolute = False):
    corr_result = defaultdict(float)
    col_list = list(df.columns)
    col_list.remove(target)
    target_value = df[target].values
    for item in col_list:
        col_value = df[item].values
        corr_value,_ = stats.spearmanr(col_value, target_value)
        if absolute:
            corr_result[item] = abs(corr_value)
        else:
            corr_result[item] = corr_value
    return corr_result

def pca_ranking(df,target,n_components=8,absolute = False):
    col_list = list(df.columns)
    col_list.remove(target)
    data_scaled = pd.DataFrame(preprocessing.scale(df[col_list]),columns = col_list) 
    pca = PCA(n_components)
    pca.fit(data_scaled)
    explained_variance = pca.explained_variance_ratio_[0] # extract first component
    print(f'Explained variance for the first component:{explained_variance}')
    weight = pca.components_[0]
    corr_result = defaultdict(float)
    for i in range(len(col_list)):
        if absolute:
            corr_result[col_list[i]] = abs(weight[i])
        else:
            corr_result[col_list[i]] = weight[i]
    return corr_result

def mrmr_ranking(df,target):
    corr_result = defaultdict(float)
    corr_table = df.corr('spearman')
    col_list = list(df.columns)
    col_list.remove(target) 
    S = len(col_list) - 1 
    for item in col_list:
        corr_y = corr_table[item][target]
        corr_x = corr_table[item]
        corr_x = corr_x.drop([target, item], axis = 0)
        corr_result[item] = abs(corr_y) - sum([abs(x) for x in corr_x]) / S
    return corr_result

def dropcol_importances(model,X_train, y_train, X_valid, y_valid,metric,proba = False):
    model.fit(X_train, y_train)
    if proba == True:
        baseline = metric(y_valid, model.predict_proba(X_valid))
    else:
        baseline = metric(y_valid, model.predict(X_valid))
    imp = []
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_valid_ = X_valid.drop(col, axis=1)
        model_ = copy.deepcopy(model)
        model_.fit(X_train_, y_train)
        if proba == True:
            m = metric(y_valid, model_.predict_proba(X_valid_))
        else:
            m = metric(y_valid, model_.predict(X_valid_))
        imp.append(baseline - m)
    return imp

def permutation_importances(model, X_valid, y_valid,metric,proba = False):
    if proba == True:
        baseline = metric(y_valid, model.predict_proba(X_valid))
    else:
        baseline = metric(y_valid, model.predict(X_valid))
    imp = []
    for col in X_valid.columns:
        save = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        if proba == True:
            m = metric(y_valid, model.predict_proba(X_valid))
        else:
            m = metric(y_valid, model.predict(X_valid))
        X_valid[col] = save
        imp.append(baseline - m)
    return imp

def get_order_from_imp_dict(imp):
    return [k for k, _ in sorted(imp.items(), key=lambda item: item[1], reverse=True)]

# design to compare different feature importances 
def loss_feature_lgbm(X,y,order_list,num_feature = 8):
    loss_list = []
    folds = 5
    seed = 42
    shuffle = True
    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
    
    params = {'objective': 'binary'}
    for i in range(num_feature):
        X_ = X[order_list[:i+1]]
        loss = 0
        for train_idx, valid_idx in kf.split(X_, y):
            train_x,train_y = X_.iloc[train_idx,:], y[train_idx]
            valid_x,valid_y = X_.iloc[valid_idx,:], y[valid_idx]
            model = lgb.LGBMClassifier()
            model.fit(train_x,train_y)
            y_pred = model.predict_proba(valid_x)
            loss += log_loss(valid_y, y_pred)
        loss = loss/folds
        loss_list.append(loss)
    return loss_list

# select features automatically
def automatic_feature_search(X,y,order_list):
    folds = 5
    seed = 42
    shuffle = True
    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
    reversed_order = order_list[::-1]
    params = {'objective': 'binary'}
    ## baseline model
    loss = 0
    for train_idx, valid_idx in kf.split(X, y):
        train_x,train_y = X.iloc[train_idx,:], y[train_idx]
        valid_x,valid_y = X.iloc[valid_idx,:], y[valid_idx]
        model = lgb.LGBMClassifier()
        model.fit(train_x,train_y)
        y_pred = model.predict_proba(valid_x)
        loss += log_loss(valid_y, y_pred)
    min_loss = loss/folds #initialize with base model loss
    loss_list = [min_loss]
    col_index = 0 #record stop index
    for i in range(1,len(reversed_order)):
        col_index = i
        col = reversed_order[i:]
        X_ = X[col]
        loss = 0
        for train_idx, valid_idx in kf.split(X_, y):
            train_x,train_y = X_.iloc[train_idx,:], y[train_idx]
            valid_x,valid_y = X_.iloc[valid_idx,:], y[valid_idx]
            model = lgb.LGBMClassifier()
            model.fit(train_x,train_y)
            y_pred = model.predict_proba(valid_x)
            loss += log_loss(valid_y, y_pred)
        loss = loss/folds
        loss_list.append(loss)
        if loss > min_loss:
            col_index = i-1
            break
        else: 
            min_loss = loss
    return X_.columns[col_index:][::-1],loss_list