# sklearn packages for base models other than NN
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle

# use pytorch to NN implementation
import torch
import torch.nn as nn 
import torch.nn.functional as F 

# scientific calculation tools
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, num_features):
        super(NeuralNet, self).__init__()
        # self.input_feats = input_feats

        # input layer, 1 hidden layer, output layer
        self.fc1 = nn.Linear(num_features, 480)
        self.fc2 = nn.Linear(480, 320)
        self.fc3 = nn.Linear(320, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class NeuralNetRegressor():
    def __init__(self, num_features):
        self.num_features = num_features