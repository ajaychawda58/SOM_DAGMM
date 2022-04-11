import os
import pickle
import gc
import sys
import argparse
import numpy as np

import torch 
from utils import *

from minisom import MiniSom
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from som_dagmm.model import DAGMM, SOM_DAGMM
from som_dagmm.compression_network import CompressionNetwork
from som_dagmm.estimation_network import EstimationNetwork
from som_dagmm.gmm import GMM, Mixture

from SOM import som_train, som_pred

#read_inputs
def parse_args():
    
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vehicle_claims', type=str)
    parser.add_argument('--embedding', dest='embed', help='one_hot, label', default='NULL', type=str)
    parser.add_argument('--features', dest='features', help='all, numerical, categorical', default='all', type=str)
    parser.add_argument('--threshold', dest='threshold', help='32', default = 20, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
save_path = os.path.join(args.dataset + "_" + args.features + "_" + args.embed)
#read data
# get labels from dataset and drop them if available
if args.dataset == 'credit_card':
    data = load_data('data/CreditCardFraud/creditcard.csv')
    Y = get_labels(data, args.dataset)
if args.dataset == 'arrhythmia':
    data = load_data('data/arrhythmia.csv')
    data = remove_cols(data, ['J'])
    Y = get_labels(data, args.dataset)
if args.dataset == 'kdd':
    names = [i for i in range(0,43)]
    data = load_data('data/NSL-KDD/KDDTrain+.txt', names)
    categorical_cols = [1,2,3,4]
    Y = get_labels(data, args.dataset)

#Select features
if args.features == "categorical":
    data = data[categorical_cols]
if args.features == "numerical":
    data = remove_cols(data, categorical_cols)

#encode categorical variables 
if args.embed == 'one_hot':
    data = one_hot_encoding(data, categorical_cols)
if args.embed == 'label_encode':
    data = label_encoding(data, categorical_cols)

# Remove columns with NA values
data = fill_na(data)
# normalize data
data = normalize_cols(data)
#test and train split
train_data, test_data, Y_train, Y_test = split_data(data, Y, 0.8)

#train_data = train_data.values.astype(np.float32)
print(train_data.shape)

#Convert to torch tensors
data = torch.tensor(data.values.astype(np.float32))
train_data = torch.tensor(train_data.values.astype(np.float32))
test_data = torch.tensor(test_data.values.astype(np.float32))

#Convert tensor to TensorDataset class.
dataset = TensorDataset(data)

#TrainLoader
dataloader = DataLoader(dataset, batch_size= 1024, shuffle=True)



compression = CompressionNetwork(data.shape[1])
estimation = EstimationNetwork()
gmm = GMM(2,6)
mix = Mixture(6)
dagmm = DAGMM(compression, estimation, gmm)
net = SOM_DAGMM(dagmm)
net.eval()
out = net(data)
threshold = np.percentile(out, 20)
pred = (out > threshold).numpy().astype(int)

# Precision, Recall, F1
p, r, f, a = get_scores(pred, Y)
print("Precision:", p, "Recall:", r, "F1 Score:", f, "AUROC:", a)


