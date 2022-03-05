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
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 1e-6, seed = 29, batch_size = 10,
								epochs = 5, loss_function = "bce")
	# Since the final layer is a sigmoid function, the outputc is between 0 and 1, 
	# but the target digits are between 0 and 9. In order to make them match I divide
	# the y_train and y_test arrays by 10.
	(test_train_loss, test_val_loss) = test_nn.fit(X_train.T, y_train * 0.1, X_test.T, y_test * 0.1)
	print(test_train_loss)
	print(test_val_loss)
	# plot losses
	plt.figure()
	plt.plot(test_train_loss, label = "Per Epoch Loss for Training Set")
	plt.show()
	
	plt.figure()
	plt.plot(test_val_loss, label = "Per Epoch Loss for Test Set")

	

if __name__ == "__main__":
    main()

