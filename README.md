# Abstract

Based on the starter code provided, `mlp.py` is an objective implementation of a classic multi-layer perceptron in python. With an experimentally designed model based on the Relu activation function, I was able to achieve 96.08% accuracy on the MNIST handwriting dataset. I experienced more difficult with the UCI ML miles per gallon dataset, with a 1.6 final validation loss. Despite this, I found the project very rewarding and beneficial as a learning experience in activation functions, traditional machine learning, as well as a needed refresher in linear algebra.

# Methodology

I began the project by attempting to understand and implement the forward and backward method for each activation functions. For many of the more complex activation functions this was especially difficult for me as I'm relatively inexperience with this mathematical style of coding, but reviewing the slides, my notes, and office hours helped me understand the activation functions enough to implement them. Additionally, since I am a very visual thinker, I found it helpful to write a file called `test_activation_functions.py` which used matplotlib to graph each activation function and its derivative (with the exception of Softmax due to its multidimensionality) on a chart so that I could visually compare the graph of the function to graphs in the slides.

During the early implementation of the activation functions I encountered what I called "dimensionality hell", meaning that when I fixed one error caused by incompatible matrix shapes, two or more arose to take its place. However, with patience and a few trips to office hours, I was able to power through "dimensionality hell" and on to finalizing and testing my multi-layer perceptron implementation.

I found it helpful during early testing to create a file called `simplest_test_ever.py`, which created a matrix of random values *X* and then assigned *X* to *y*. While this test was not at all interesting from a predictions standpoint, it did allow me to test that the `mlp` implementation was working correctly.

I made a small change to the object oriented design from the starter code that I will document here. While the starter code passes in a layer's input each time `layer.backward` is called in `mlp.backward`, I found it more intuitive to store the layer's input as `layer.input` during the `forward` call, thus only having to provide an appropriate loss gradient when calling `layer.backward`.

## MPG Design
In my testing, I found that deeper networks didn't seem to preform well on the MPG dataset. As such I settled on the simple architecture shown below:

*Code block 1 - Architecture for MPG prediction*
```python
layers = [
	mlp.Layer(7, 64, mlp.Sigmoid()),
	mlp.Layer(64, 1, mlp.Linear())
]
```

I trained the model for 64 epochs at a learning rate of 0.001, with a batch size of 64. I used mSGD for this model.

## MNIST Design
MNIST was much easier to train for and I found it much more rewarding because I was able to visualize my results. By the time of writing this I have heard in class that other students are getting >97% accuracy using the Softmax activation function and CrossEntropy loss. While both of those are implemented in my `mlp.py`, I decided that I likely do not have the knowledge or energy to beat those accuracy scores, so I decided to create my own challenge and beat that instead. Thus I've attempted to achieve the highest accuracy I can using only the Relu activation function in my layers. I settled on this rather large architecture included below:

*Code block 2 - Architecture for MNIST Classification*
```python
layers = [
	mlp.Layer(28 * 28, 512, mlp.Relu()),
	mlp.Layer(512, 256, mlp.Relu()),
	mlp.Layer(256, 128, mlp.Relu()),
	mlp.Layer(128, 64, mlp.Relu()),
	mlp.Layer(64, 10, mlp.Relu()),
	mlp.Layer(10, 1, mlp.Linear())
]
```

I used a learning rate of 0.001, batch size of 32, and made the discovery that by using rmsprop, I was able to reduce the number of iterations needed from 64 to 32, while upping the accuracy from ~94% to 96.08% It should be noted that this model took around 50 seconds per epoch to train on my laptop, meaning it required ~30 minutes to train. Because of this long wait time while training models, I modified `train_mnist.py` to save the model as a `.pkl` file, and wrote `eval_mnist.py` to read the `.pkl` file and preform testing and evaluation on the model. This allowed me to save good models and evaluate them without the need to retrain each time.

I should note that I did implement dropout in `mlp.py` but it didn't seem to be particularly useful in either model I trained. Perhaps I didn't change my hyperparameters granularly enough to notice a benefit from it.

# Results
Included below are results collected from the UCI ML MPG dataset and results collected from the MNIST handwriting dataset.

## MPG Results

In all honesty, I'm unsure if my loss calculations give meaningful results at all. My final training loss was 1.7343 and my final validation loss was 1.6549, however, as you'll see from the table of results below, my model appears to preform on par with linear regression at least.

*Figure 1 - Training and Validation Loss Curves after training a model on the UCI ML MPG dataset*
![[MPG_Loss_Curve.png]]

| displ. | cyl. | hp    | weight | accel. | model_year | origin | True MPG | Predicted MPG |
| ------ | ---- | ----- | ------ | ------ | ---------- | ------ | -------- | ------------- |
| 262.0  | 8.0  | 110.0 | 3221.0 | 13.5   | 75.0       | 1.0    | 20.0     | 19.807677     |
| 250.0  | 6.0  | 100.0 | 3781.0 | 17.0   | 74.0       | 1.0    | 16.0     | 17.202223     |
| 140.0  | 4.0  | 86.0  | 2790.0 | 15.6   | 82.0       | 1.0    | 27.0     | 29.777554     |
| 440.0  | 8.0  | 215.0 | 4735.0 | 11.0   | 73.0       | 1.0    | 13.0     | 10.863865     |
| 120.0  | 4.0  | 87.0  | 2979.0 | 19.5   | 72.0       | 2.0    | 21.0     | 22.573529     |
| 318.0  | 8.0  | 140.0 | 3735.0 | 13.2   | 78.0       | 1.0    | 19.4     | 18.522663     |
| 429.0  | 8.0  | 198.0 | 4341.0 | 10.0   | 70.0       | 1.0    | 15.0     | 11.084758     |
| 390.0  | 8.0  | 190.0 | 3850.0 | 8.5    | 70.0       | 1.0    | 15.0     | 12.899593     |
| 140.0  | 4.0  | 88.0  | 2890.0 | 17.3   | 79.0       | 1.0    | 22.3     | 26.731342     |
| 70.0   | 3.0  | 90.0  | 2124.0 | 13.5   | 73.0       | 3.0    | 18.0     | 28.538790     |
| 121.0  | 4.0  | 80.0  | 2670.0 | 15.0   | 79.0       | 1.0    | 27.4     | 28.353302     |


## MNIST Results
MNIST preformed much better, and as reported earlier, I was able to achieve 96.08% accuracy with a purely Relu-based network. A graph of the loss curve is given below.

*Figure 2 - Training and Validation Loss Curves after training a model on the MNIST handwriting dataset*
![[MNIST_Loss_Curve.png]]

*Figure 3- Samples from MNIST with predicted and actual numeric values*
![[MNIST_Samples 2.png]]

The MNIST model preformed very well, and seemed to run fairly efficiently after training. I wanted to bring up one sample I came across while testing the model. I know intuitively that a 96.08% accuracy is pretty good for such a dataset, however this particular handwritten "5" really brought my attention to it, since I'm unsure I could correctly identify it without the label.

*Figure 4 - Unusually written number five that the model successfully identified*
![[weird_5.png]]

# Code Repository Link
https://github.com/Sam-Hildebrand/MLP

> **Note:** `train_mnist.py` trains a model and saves it as a `.pkl` file. To evaluate this model run `eval_mnist.py`.  This allowed me to modify my testing code without retraining the model. Since I used a fairly large network, it can take a while to train. 

# Conclusion
Despite the frustrations, particularly with the MPG dataset, I have enjoyed this project. It allowed me to learn hands on in a way that I likely never would otherwise, since most online resources I had looked at before taking this class were either far too simplistic or too advanced for me to grasp without prior knowledge. One high-level thing I'm taking away from this project is the ability to turn mathematics into code is more of a skill than I thought it was. While I don't think I'm exceptionally bad at turning a formula into a line of code, this is definitely an area I'd like to improve in, and this project has helped me. While I didn't gain as much insight from the MPG dataset, the MNIST dataset definitely intrigued me with the idea that a relatively simple program can recognize handwritten numbers so accurately. 

I suppose this was my first machine learning project on this scale, and it has definitely got me hooked. I found it a little bit addicting, tweaking my model for MNIST and watching the accuracy rise or fall by a few fractions of a percent. Above what I learned, I think I'm addicted to this sort of thing now, which I guess is not a bad situation to be in.