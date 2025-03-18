import NeuralNetwork as nn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

# retrieve file path from env
load_dotenv()
train_file_path = os.getenv("TRAIN_FILE_PATH")

# split and process train data
df_train = pd.read_csv(train_file_path)
X_train = df_train.drop(columns="Credit_Score")
y_train = df_train["Credit_Score"]

X, y = X_train.to_numpy(), y_train.to_numpy()
X, y = X.astype(np.float32), y.astype(int)

def one_hot_encode(y, num_classes=3):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y = one_hot_encode(y)
print(y)

# initialize layers
dense1 = nn.Layer_Dense(46, 128)
activation1 = nn.Activation_ReLU()
dense2 = nn.Layer_Dense(128, 128)
activation2 = nn.Activation_ReLU()
dense3 = nn.Layer_Dense(128, 128)
activation3 = nn.Activation_ReLU()
dense4 = nn.Layer_Dense(128, 3)

# initialize optimizers
optimizer = nn.Optimizer_Adam(learning_rate=0.005, decay=1e-3)
loss_activation = nn.Activation_Softmax_Loss_CategoricalCrossentropy()

# train model with 30001 epochs
for epoch in range(30001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    dense4.forward(activation3.output)

    loss = loss_activation.forward(dense4.output, y)
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f"epoch: {epoch}, acc: {accuracy}, loss: {loss}, lr: {optimizer.current_learning_rate}")
    
    loss_activation.backward(loss_activation.output, y)
    dense4.backward(loss_activation.dinputs)
    activation3.backward(dense4.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.update_params(dense4)
    optimizer.post_update_params()