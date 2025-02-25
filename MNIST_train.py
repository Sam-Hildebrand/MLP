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

    train_y = np.array(train_y).reshape(-1, 1)  # Ensure correct shape
    val_y = np.array(val_y).reshape(-1, 1)  # Ensure correct shape
    test_y = np.array(test_y).reshape(-1, 1)    # Ensure correct shape

    return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == "__main__":
    # Load and preprocess the MNIST dataset
    train_x, train_y, val_x, val_y, test_x, test_y = download_mnist()

    # Define the MLP model
    layers = [
        mlp.Layer(28 * 28, 512, mlp.Relu()),
        mlp.Layer(512, 256, mlp.Relu()),
        mlp.Layer(256, 128, mlp.Relu()),
        mlp.Layer(128, 64, mlp.Relu()),
        mlp.Layer(64, 10, mlp.Relu()),
        mlp.Layer(10, 1, mlp.Linear())
    ]

    perceptron = mlp.MultilayerPerceptron(layers)

    # Train the model
    training_loss, validation_loss = perceptron.train(
        train_x, train_y, val_x, val_y, 
        loss_func=mlp.SquaredError(),
        learning_rate=0.0001,
        batch_size=32,
        epochs=64
    )

    with open("MNIST_model.pkl", 'wb') as f:
        pickle.dump((perceptron, training_loss, validation_loss), f)

    print("Training Complete")