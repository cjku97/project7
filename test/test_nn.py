import numpy as np
from nn import (io, nn, preprocess)


# TODO: Write your test functions and associated docstrings below.

"""
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
"""

def test_forward():
    sig = lambda x: 1.0/(1.0 + np.exp(-x))
    relu = lambda x: np.maximum(x, 0)
    test_arch = [{'input_dim': 16, 'output_dim': 4, 'activation': 'relu'},
    	{'input_dim': 4, 'output_dim': 16, 'activation': 'sigmoid'},
    	{'input_dim': 16, 'output_dim': 1, 'activation': 'relu'}]
    test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "mse")
    X = np.random.rand(16,10)
    (output, cache) = test_nn.forward(X)
    param_dict = test_nn._param_dict
    assert len(cache) == (2 * len(test_arch)) + 1
    assert len(param_dict) == 2 * len(test_arch)
    assert cache['Z1'].all() == (param_dict['W1'].dot(X) + param_dict['b1']).all()
    assert cache['A1'].all() == relu(cache['Z1']).all()
    assert cache['Z1'].shape == cache['A1'].shape
    assert cache['A1'].shape == (4, 10)
    assert cache['Z2'].all() == (param_dict['W2'].dot(cache['A1']) + param_dict['b2']).all()
    assert cache['A2'].all() == sig(cache['Z2']).all()
    assert cache['Z2'].shape == cache['A2'].shape
    assert cache['A2'].shape == (16, 10)
    assert cache['Z3'].all() == (param_dict['W3'].dot(cache['A2']) + param_dict['b3']).all()
    assert cache['A3'].all() == relu(cache['Z3']).all()
    assert cache['Z3'].shape == cache['A3'].shape
    assert cache['A3'].shape == (1, 10)
    assert cache['A3'].all() == output.all()


def test_single_forward():
    W = np.random.rand(4, 8)
    b = np.random.rand(4, 1)
    A = np.random.rand(8, 16)
    sig = lambda x: 1.0/(1.0 + np.exp(-x))
    relu = lambda x: np.maximum(x, 0)
    
    test_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'}]
    test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "bce")
    
    (A_sig, Z_sig) = test_nn._single_forward(W, b, A, "sigmoid")
    assert Z_sig.all() == (W.dot(A) + b).all()
    assert A_sig.all() == sig(Z_sig).all()
    (A_relu, Z_relu) = test_nn._single_forward(W, b, A, "relu")
    assert Z_relu.all() == (W.dot(A) + b).all()
    assert A_relu.all() == relu(Z_relu).all()    


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
