import numpy as np
import matplotlib.pyplot as plt
import mlp

def graph_activation_function(Activation_Function, function_name):
    # Generate values from -10 to 10
    x = np.linspace(-10, 10, 100)
    y = Activation_Function.forward(x)
    dy = Activation_Function.derivative(x)

    plt.plot(x, y, label=function_name + ' Function')
    plt.plot(x, dy, label=function_name + ' Derivative', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title(function_name + ' Function and Its Derivative')
    plt.legend()
    plt.grid()
    plt.show()

graph_activation_function(mlp.Sigmoid(), "Sigmoid")
graph_activation_function(mlp.Tanh(), "Tanh")
graph_activation_function(mlp.Relu(), "Relu")
graph_activation_function(mlp.Softmax(), "Softmax")
graph_activation_function(mlp.Linear(), "Linear")
graph_activation_function(mlp.Softplus(), "Softplus")
graph_activation_function(mlp.Mish(), "Mish")
