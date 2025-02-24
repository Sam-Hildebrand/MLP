import mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x_train = np.random.randn(70, 1)
y_train = x_train

x_val = np.random.randn(15, 1)
y_val = x_val

x_test = np.array([[1],
                [2],
                [3],
                [4],
                [5]])

y_test = x_test

perceptron = mlp.MultilayerPerceptron((mlp.Layer(1, 12, mlp.Sigmoid()), mlp.Layer(12, 14, mlp.Sigmoid()), mlp.Layer(14, 1, mlp.Linear())))


training_loss, validation_loss = perceptron.train(x_train, y_train, x_val, y_val, mlp.SquaredError(), learning_rate=0.01, epochs=100, batch_size=16)

plt.plot(training_loss, color='b', label='Training')
plt.plot(validation_loss, color='r', label="Validation")
plt.title("Loss Curve", size=16)
plt.legend()
plt.show()

table = pd.DataFrame({
    'True Y': y_test.flatten(),
    'Predicted Y': perceptron.forward(x_test).flatten()
})

print(table)