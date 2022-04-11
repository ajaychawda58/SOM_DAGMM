import pandas as pd
import os
import sys
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def load_data(file_path, names = None):
    names = names
    data = pd.read_csv(file_path, names = names)
    return data

def one_hot_encoding(data, cols):
    encode_data = data[cols]
    encoded_data = pd.DataFrame()
    for column in encode_data:
        new_data = pd.get_dummies(encode_data[column], prefix=column)
        encoded_data = pd.concat([encoded_data, new_data], axis=1)
    data.drop(cols, axis=1, inplace=True)
    data = pd.concat([data, encoded_data], axis=1)
    return data

def label_encoding(data, cols):
    encode_data = data[cols]
    encoded_data = pd.DataFrame()
    encoder = preprocessing.LabelEncoder()
    for column in encode_data:
        new_data = encoder.fit_transform(encode_data[column])
        new_data = pd.DataFrame(new_data, columns=[column])
        encoded_data = pd.concat([encoded_data, new_data], axis=1)
    data.drop(cols, axis=1, inplace=True)
    data = pd.concat([data, encoded_data], axis=1)
    return data

def get_labels(data, name):
    if name == 'credit_card':
        label = data['Class']
        data.drop(['Class'], axis = 1, inplace=True)        
    if name == 'arrhythmia':
        label = data['class']
        label = (np.where(label == (3|4|5|7|8|9|14|15), 0, 1))
        data.drop(['class'], axis = 1, inplace=True)
    if name == 'kdd':
        label = data[41]
        label = np.where(label == "normal", 0, 1)
        data.drop([41], axis = 1, inplace=True) 
    return label

def get_scores(y_pred, y):
    precision = precision_score(y_pred, y, average='binary')
    recall = recall_score(y_pred, y, average='binary')
    f1 = f1_score(y_pred, y, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return precision, recall, f1, auc

def get_confusion_matrix(y_pred, y):
    tn, fp, fn, tp = confusion_matrix(y_pred, y).ravel()
    return tn, fp, fn, tp

def normalize_cols(data):
    sc = MinMaxScaler (feature_range = (0,1))
    data = sc.fit_transform(data)
    data = pd.DataFrame(data)  
    return data  

def merge_cols(data_1, data_2):
    data = pd.concat([data_1, data_2], axis=1)
    return data

def remove_cols(data, cols):
    data.drop(cols, axis=1, inplace=True)
    return data

def split_data(data, Y, split=0.7):
    train = data.loc[:split*len(data),]
    test = data.loc[split*len(data)+1:,]
    Y_train = Y[:int(split*len(data)),]
    Y_test = Y[int(split*len(data))+1:,]
    return train, test, Y_train, Y_test

def fill_na(data):
    data = data.fillna(value = 0, axis=1)
    return data

def relative_euclidean_distance(x1, x2):
    num = torch.norm(x1 - x2, p=2, dim=1)  
    denom = torch.norm(x1, p=2, dim=1)  
    print(num.dtype)
    print(denom.dtype)
    return num / torch.max(denom)


def cosine_similarity(x1, x2, eps=1e-8):
    dot_prod = torch.sum(x1 * x2, dim=1)  
    dist_x1 = torch.norm(x1, p=2, dim=1)  
    dist_x2 = torch.norm(x2, p=2, dim=1)  
    return dot_prod / np.max(dist_x1*dist_x2, eps)