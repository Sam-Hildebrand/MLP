import mlp
import os
import numpy as np
import pandas as pd
import pickle
import requests
import gzip
import matplotlib.pyplot as plt

# Download MNIST dataset from github
url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
filename = "mnist.pkl.gz"

def download_mnist():
    if os.path.exists(filename):
        with gzip.open("mnist.pkl.gz", "rb") as f:
            train_data, val_data, test_data = pickle.load(f, encoding="latin1")
    else:
        print("Downloading MNIST dataset from Github")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download MNIST dataset. HTTP Status: {response.status_code}")

        with open("mnist.pkl.gz", "wb") as f:
            f.write(response.content)

        with gzip.open("mnist.pkl.gz", "rb") as f:
            train_data, val_data, test_data = pickle.load(f, encoding="latin1")

    # Extract inputs and labels
    train_x, train_y = train_data
    val_x, val_y = val_data
    test_x, test_y = test_data

    return train_x, train_y, val_x, val_y, test_x, test_y

# Load and preprocess the MNIST dataset
train_x, train_y, val_x, val_y, test_x, test_y = download_mnist()

train_y = np.array(train_y).reshape(-1, 1)  # Ensure correct shape
val_y = np.array(val_y).reshape(-1, 1)  # Ensure correct shape
test_y = np.array(test_y).reshape(-1, 1)    # Ensure correct shape

# Define the MLP model
layers = [
    mlp.Layer(28 * 28, 128, mlp.Softplus()),
    mlp.Layer(128, 64, mlp.Softplus()),
    mlp.Layer(64, 1, mlp.Linear())
]

perceptron = mlp.MultilayerPerceptron(layers)

# Train the model
training_loss, validation_loss = perceptron.train(
    train_x, train_y, val_x, val_y, 
    loss_func=mlp.CrossEntropy(),
    learning_rate=0.001,
    batch_size=32,
    epochs=10
)

plt.plot(training_loss, color='b', label='Training')
plt.plot(validation_loss, color='r',linestyle='dashed', label="Validation")
plt.title("Loss Curve", size=16)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

# Evaluate the model
y_pred = np.round(perceptron.forward(test_x))

table = pd.DataFrame({
    'Actual Number': test_y.flatten(),
    'Predicted Number': y_pred.flatten() 
})
print("\nActual Handwritten Number vs. Predicted Number:")
print(table)