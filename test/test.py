# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import (io, nn, preprocess)

# TODO: Write your test functions and associated docstrings below.

def simulate_binary():
	# sumulating a binary dataset
	num_points = 1000
	# Chose values for params (w vector) where w[-1] is a bias term
	w = [0.8, 2.6, 4.5, 1]
	# Generate example dataset and dependent variable y
	X = np.random.rand(num_points, len(w))
	y = (w[-1] + np.expand_dims(X.dot(w[:-1]), 1) + np.random.rand(num_points, 1)*0.1).flatten()
	y = (y > np.mean(y)) * 1
	# Split into training and validation sets
	split = int(0.6*num_points)
	X_train = X[:split]
	X_val = X[split:]
	y_train = y[:split]
	y_val = y[split:]
	return(X_train, X_val, y_train, y_val)

def test_forward():
    test_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
				 {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'},
				 {'input_dim': 64, 'output_dim': 1, 'activation': 'relu'}]

	test_nn = nn.NeuralNetwork(nn_arch = test_arch2, lr = 0.001, seed = 29, batch_size = 500,
								epochs = 50, loss_function = "mse")
	X_train, X_val, y_train, y_val = 
	
	(output, cache) = test_nn.forward(X)
	
    pass


def test_single_forward():
    W_curr = np.array([[.1, .2], [.3, .4], [.5, .6]])
    b_curr = np.array([[.1, .2, .3]])
    (A_sig, Z_sig) = nn._single_forward(W_curr, b_curr, A_prev, "sigmoid")
    assert Z_sig == 
    (A_relu, Z_relu) = nn._single_forward(W_curr, b_curr, A_prev, "relu")
    


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass
