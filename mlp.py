import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple
from tqdm import tqdm


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    # Shuffle the indices
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)

    # Reorder the arrays based on the shuffled indices
    train_x_shuffled = train_x[indices]
    train_y_shuffled = train_y[indices]

    # Yield batches from the shuffled data
    for i in range(0, train_x_shuffled.shape[0], batch_size):
        batch_x = train_x_shuffled[i:i + batch_size]
        batch_y = train_y_shuffled[i:i + batch_size]
        yield (batch_x, batch_y)

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        
        pos_mask = x >= 0
        out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        
        neg_mask = ~pos_mask
        out[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        
        return out

    def derivative(self, x: np.ndarray) -> np.ndarray:
        Sig_x = self.forward(x)
        return Sig_x * (1- Sig_x)

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.square(self.forward(x))

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0).astype(float)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / exp_x.sum()

    def derivative(self, x: np.ndarray) -> np.ndarray:
        Sx: np.ndarray = self.forward(x)
        print("Sx shape: ", Sx.shape)
        return np.diagflat(Sx) - np.dot(Sx, Sx.T)

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return Sigmoid().forward(x)

class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(Softplus().forward(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        tanh_softplus = np.tanh(Softplus().forward(x))
        return tanh_softplus + (x * (1 - tanh_softplus**2) * Sigmoid().forward(x))

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.square(y_true - y_pred)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true.reshape(-1,1))

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.sum(y_true.reshape(-1,1) * np.log(np.clip(y_pred, 1e-10, 1.0)), axis=1)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true.reshape(-1,1)

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):
        """
        Initializes a layer of neurons with optional dropout.

        :param fan_in: number of neurons in the previous layer.
        :param fan_out: number of neurons in this layer.
        :param activation_function: instance of an ActivationFunction.
        :param dropout_rate: probability of dropping a neuron activation (default 0.0 means no dropout).
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        self.activations = np.empty((fan_in, fan_out))
        self.delta = np.empty((fan_in, fan_out))

        self.input = None
        self.z = None
        self.dropout_mask = None

        # Initialize weights and biases
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.random.randn(1, fan_out)

    def forward(self, h: np.ndarray, training: bool) -> np.ndarray:
        """
        Computes the activations for this layer, applying dropout if in training mode.

        :param h: input to the layer.
        :param training: if True, apply dropout; otherwise, use full activations.
        :return: layer activations.
        """
        self.input = h
        self.z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(self.z)

        if training and self.dropout_rate > 0.0:
            self.dropout_mask = (np.random.rand(*self.activations.shape) > self.dropout_rate).astype(float)
            self.dropout_mask /= (1 - self.dropout_rate)
            self.activations *= self.dropout_mask
        else:
            self.dropout_mask = None

        return self.activations

    def backward(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagates the error through the layer.

        :param delta: delta term from the next layer.
        :return: (weight gradients, bias gradients).
        """

        f_prime = self.activation_function.derivative(self.z)

        # If dropout was applied, only propagate gradients for non-dropped units
        if self.dropout_mask is not None:
            f_prime *= self.dropout_mask

        delta_l = delta * f_prime

        dL_dW = np.dot(self.input.T, delta_l)
        dL_db = np.sum(delta_l, axis=0, keepdims=True)
        self.delta = np.dot(delta_l, self.W.T)
        return dL_dW, dL_db

class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x, training)

        return x

    def backward(self, loss_grad: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all: list = []
        dl_db_all: list = []

        for layer in reversed(self.layers):
            dL_dW, dL_db = layer.backward(loss_grad)
            loss_grad = layer.delta
            dl_dw_all.insert(0, dL_dW)
            dl_db_all.insert(0, dL_db)

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32,  rmsprop: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = np.zeros(epochs)
        validation_losses = np.zeros(epochs)

        for epoch in tqdm(range(epochs), desc='Training'):

            epoch_loss = 0.0
            n_train = 0.0

            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                y_pred = self.forward(batch_x, training=True)
                
                batch_loss = loss_func.loss(batch_y, y_pred)
                epoch_loss += np.mean(batch_loss)
                loss_grad = loss_func.derivative(batch_y, y_pred)
                dl_dw_all, dl_db_all = self.backward(loss_grad)

                for layer, dL_dW, dL_db in zip(self.layers, dl_dw_all, dl_db_all):
                    layer.W -= learning_rate * dL_dW
                    layer.b -= learning_rate * dL_db
                
                n_train += 1

            training_losses[epoch] = epoch_loss / n_train

            val_pred = self.forward(val_x)
            val_loss = loss_func.loss(val_y, val_pred)
            validation_losses[epoch] = np.mean(val_loss)

        return training_losses, validation_losses
