import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn import (io, nn, preprocess)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def main():
	# simulate dataset
	digits = load_digits()
	print(digits.data.shape)
	X_all = digits.data
	y_all = digits.target
	# view first digit
	# plt.gray()
	# plt.matshow(digits.images[0])
	# plt.show()
	# split into training and test datasets
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=42)
	print(X_train.shape)
	print(y_train.shape)
	
	test_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'sigmoid'},
				 {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'},
				 {'input_dim': 64, 'output_dim': 1, 'activation': 'sigmoid'}]
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 0.01, seed = 9, batch_size = 10,
								epochs = 10, loss_function = "bce")
	# print("PARAMETER DICTIONARY:")
	# print(test_nn._param_dict)
	(test_output, test_cache) = test_nn.forward(X_train.T)
	# print(test_cache)
	print("OUTPUT")
	print(test_output.shape)
	test_grad_dict = test_nn.backprop(y_train, test_output, test_cache)
	print(test_grad_dict.keys())
	# print(test_grad_dict)

	

if __name__ == "__main__":
    main()

