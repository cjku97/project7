import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn import (io, nn, preprocess)
from sklearn.datasets import load_digits

def main():
	# simulate dataset
	digits = load_digits()
	print(digits.data.shape)
	# view first digit
	plt.gray()
	plt.matshow(digits.images[0])
	plt.show()
	
	test_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'sigmoid'},
				 {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'},
				 {'input_dim': 64, 'output_dim': 1, 'activation': 'sigmoid'}]
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 0.01, seed = 9, batch_size = 10,
								epochs = 10, loss_function = "bce")
	# print("PARAMETER DICTIONARY:")
	# print(test_nn._param_dict)
	(test_output, test_cache) = test_nn.forward(digits.data.T)
	# print(test_cache)
	print("OUTPUT")
	print(test_output.shape)
	test_grad_dict = test_nn.backprop(digits.data.T, test_output, test_cache)
	# print(test_grad_dict)

	

if __name__ == "__main__":
    main()

