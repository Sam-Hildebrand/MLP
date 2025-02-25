import mlp
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from MNIST_train import download_mnist

if not os.path.exists("MNIST_model.pkl"):
    print("Run `python3 MNIST_train.py` first to train the model.")
    quit()
else: 
    with open("MNIST_model.pkl", "rb") as f:
        perceptron, training_loss, validation_loss = pickle.load(f, encoding="latin1")

    # Load and preprocess the MNIST dataset
    train_x, train_y, val_x, val_y, test_x, test_y = download_mnist()

    plt.plot(training_loss, color='b', label='Training')
    plt.plot(validation_loss, color='r',linestyle='dashed', label="Validation")
    plt.title("Loss Curve", size=16)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    # Evaluate the model
    y_pred = np.round(perceptron.forward(test_x))

    # Select one random sample per class (0-9)
    selected_images = []
    selected_labels = []
    predicted_labels = []

    for digit in range(10):
        indices = np.where(test_y.flatten() == digit)[0]
        if len(indices) > 0:
            idx = np.random.choice(indices)
            selected_images.append(test_x[idx].reshape(28, 28))
            selected_labels.append(test_y[idx][0])
            predicted_labels.append(int(y_pred[idx][0]))

    # Display images in a grid
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Actual vs. Predicted Numbers", fontsize=16)

    for i, ax in enumerate(axes.flat):
        ax.imshow(selected_images[i], cmap='gray')
        ax.set_title(f'Actual: {selected_labels[i]}\nPredicted: {predicted_labels[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Create and display the table
    sample_table = pd.DataFrame({
        'Actual Number': selected_labels,
        'Predicted Number': predicted_labels
    })
    print("\nSampled Handwritten Numbers vs. Predicted Numbers:")
    print(sample_table)

    # Calculate accuracy
    accuracy = np.mean(y_pred == test_y) * 100

    print(f"\nModel Accuracy: {accuracy:.2f}%")