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
    for i in range(0, train_x.shape[0], batch_size):
        batch_x = train_x[i:i + batch_size]
        batch_y = train_y[i:i + batch_size]

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
        return 1 / (1 + np.exp(-x))

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
        e_zi = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_zi / np.sum(e_zi, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        Sx: np.ndarray = self.forward(x)
        return np.diagflat(Sx) -  Sx @ np.transpose(Sx)


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

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
        return 2 * (y_pred.reshape(-1,1) - y_true.reshape(-1,1))

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.sum(y_true * np.log(np.clip(y_pred, 1e-10, 1.0)), axis=1)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -y_true / np.clip(y_pred, 1e-10, 1.0)

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = np.empty((fan_in, fan_out))
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = np.empty((fan_in, fan_out))

        self.input = None
        self.z = None

        # Initialize weights and biaes
        self.W = np.random.randn(fan_in, fan_out) * 0.01 
        self.b = np.random.randn(1, fan_out) 

    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.input = h
        #print("h shape:", h.shape)
        #print("W shape:", self.W.shape)
        #print("self.b shape", self.b.shape)

        self.z = np.dot(h, self.W) + self.b

        #print("self.z shape:", self.z.shape)
        self.activations = self.activation_function.forward(self.z)
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """

        f_prime = self.activation_function.derivative(self.z)  # Compute activation derivative
        print("delta shape: ", delta.shape)
        print("f_prime shape: ", f_prime.shape)

        delta_l = delta * f_prime
        print("h shape: ", h.shape)

        dL_dW = np.dot(h.T, delta_l)
        dL_db = np.sum(delta_l, axis=0, keepdims=True) 

        #print("Self.W: ", self.W)
        print("Self.W shape: ", self.W.shape)
        self.delta = np.dot(delta_l, self.W.T) 
        print("dL_dW: ", dL_dW.shape)
        print("dL_db: ", dL_db.shape)
        return dL_dW, dL_db

class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all: list = []
        dl_db_all: list = []

        for layer in reversed(self.layers):
            dL_dW, dL_db = layer.backward(input_data, loss_grad)
            dl_dw_all.insert(0, dL_dW)
            dl_db_all.insert(0, dL_db)

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
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
        n_train = train_x.shape[0]

        for epoch in tqdm(range(epochs), desc='Training'):

            epoch_loss = 0.0

            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                output = self.forward(batch_x)
                
                #print("batch_y shape: ", batch_y.shape)
                batch_loss = loss_func.loss(batch_y, output)
                #print("batch loss shape", batch_loss.shape)
                epoch_loss += np.sum(batch_loss)
                loss_grad = loss_func.derivative(batch_y, output)
                #print("loss_grad: ", loss_grad.shape)
                dl_dw_all, dl_db_all = self.backward(loss_grad, output)

                for layer, dL_dW, dL_db in zip(self.layers, dl_dw_all, dl_db_all):
                    layer.W -= learning_rate * dL_dW
                    layer.b -= learning_rate * dL_db

            training_losses[epoch] = epoch_loss / n_train

            val_output = self.forward(val_x)
            val_loss = loss_func.loss(val_y, val_output)
            validation_losses[epoch] = np.mean(val_loss)

        return training_losses, validation_losses
