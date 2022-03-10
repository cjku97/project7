import numpy as np
from nn import (io, nn, preprocess)
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error


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
    test_arch = []
    test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "bce")    
    (A_sig, Z_sig) = test_nn._single_forward(W, b, A, "sigmoid")
    assert Z_sig.all() == (W.dot(A) + b).all()
    assert A_sig.all() == sig(Z_sig).all()
    (A_relu, Z_relu) = test_nn._single_forward(W, b, A, "relu")
    assert Z_relu.all() == (W.dot(A) + b).all()
    assert A_relu.all() == relu(Z_relu).all()    


def test_single_backprop():
    sig = lambda x: 1.0/(1.0 + np.exp(-x))
    sig_deriv = lambda x: sig(x) * (1 - sig(x))
    relu = lambda x: np.maximum(x, 0)
    relu_deriv = lambda x: (x > 0) * 1
    test_arch = []
    test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "mse")
    W1 = np.random.rand(4, 8)
    b1 = np.random.rand(4, 1)
    Z1 = np.random.rand(4, 20)
    A0 = np.random.rand(8, 20)
    dA1 = np.random.rand(4, 20)
    (dA0_sig, dW1_sig, db1_sig) = test_nn._single_backprop(W1, b1, Z1, A0, dA1, "sigmoid")
    sig_deriv_Z1 = sig_deriv(Z1)
    assert dA0_sig.all() == W1.T.dot(dA1 * sig_deriv_Z1).all()
    assert dW1_sig.all() == (dA1 * sig_deriv_Z1).dot(A0.T).all()
    assert db1_sig.all() == np.sum(dA1 * sig_deriv_Z1, axis = 1, keepdims = True).all()
    (dA0_relu, dW1_relu, db1_relu) = test_nn._single_backprop(W1, b1, Z1, A0, dA1, "relu")
    relu_deriv_Z1 = relu_deriv(Z1)
    assert dA0_relu.all() == W1.T.dot(dA1 * relu_deriv_Z1).all()
    assert dW1_relu.all() == (dA1 * relu_deriv_Z1).dot(A0.T).all()
    assert db1_relu.all() == np.sum(dA1 * relu_deriv_Z1, axis = 1, keepdims = True).all()


def test_predict():
    test_arch = [{'input_dim': 16, 'output_dim': 4, 'activation': 'relu'},
    	{'input_dim': 4, 'output_dim': 16, 'activation': 'sigmoid'},
    	{'input_dim': 16, 'output_dim': 1, 'activation': 'relu'}]
    test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "mse")
    X = np.random.rand(16,10)
    (output, cache) = test_nn.forward(X)
    predict = test_nn.predict(X)
    param_dict = test_nn._param_dict
    assert output.all() == predict.all()
    assert predict.shape == (10, )

def test_binary_cross_entropy():
	test_arch = []
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "mse")
	y_hat = np.array([0.2, 0.4, 0.8, 0.2, 0.4, 0.8, 0.2, 0.2, 0.9, 0.6])
	y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
	log_loss_test = log_loss(y, y_hat)
	bce = -(y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))
	bce_mean = np.sum(bce)/10
	nn_bce = test_nn._binary_cross_entropy(y, y_hat)
	assert abs(nn_bce - log_loss_test) < 1e-5
	assert abs(nn_bce - bce_mean) < 1e-5
	assert abs(log_loss_test - bce_mean) < 1e-5


def test_binary_cross_entropy_backprop():
	test_arch = []
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "mse")
	y_hat = np.array([0.2, 0.4, 0.8, 0.2, 0.4, 0.8, 0.2, 0.2, 0.9, 0.6])
	y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
	dA = (-y/y_hat) + ((1 - y)/(1 - y_hat))
	test_dA = test_nn._binary_cross_entropy_backprop(y, y_hat)
	assert dA.all() == test_dA.all()


def test_mean_squared_error():
	test_arch = []
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "mse")
	y_true = np.array([3, -0.5, 2, 7])
	y_pred = np.array([2.5, 0.0, 2, 8])
	mse = mean_squared_error(y_true, y_pred)
	test_mse = test_nn._mean_squared_error(y_true, y_pred)
	np_mse = np.square(y_true - y_pred).mean()
	assert abs(mse - test_mse) < 1e-5
	assert abs(mse - np_mse) < 1e-5
	assert abs(test_mse - np_mse) < 1e-5


def test_mean_squared_error_backprop():
	test_arch = []
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1, seed = 1, batch_size = 1, epochs = 1, loss_function = "mse")
	y_hat = np.array([3, -0.5, 2, 7])
	y = np.array([2.5, 0.0, 2, 8])
	dA = -2 * (y - y_hat)
	test_dA = test_nn._mean_squared_error_backprop(y, y_hat)
	assert dA.all() == test_dA.all()


def test_one_hot_encode():
    seq1 = ['AGA']
    assert preprocess.one_hot_encode_seqs(seq1)[0].tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    seq2 = ['A', 'T', 'C', 'G']
    assert np.array(preprocess.one_hot_encode_seqs(seq2)).tolist() == [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]


def test_sample_seqs():
    seqs = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
    labels = np.array([True, False, False, False, True, False, False, False, True])
    assert len(seqs) == len(labels)
    (new_seqs, new_labels) = preprocess.sample_seqs(seqs, labels)
    assert len(new_seqs) == len(new_labels)
    assert len(np.where(new_labels == True)[0]) == len(np.where(new_labels == False)[0])
