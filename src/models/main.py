import NeuralNetwork as nn
import numpy as np
import pandas as pd

df_train = pd.read_csv("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/processed/processed_train.csv")
X_train = df_train.drop(columns="Credit_Score")
y_train = df_train["Credit_Score"]

X, y = X_train.to_numpy(), y_train.to_numpy()

dense1 = nn.Layer_Dense(46, 64)
activation1 = nn.Activation_ReLU()
dense2 = nn.Layer_Dense(64, 64)
activation2 = nn.Activation_ReLU()
dense3 = nn.Layer_Dense(64, 3)

