import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn import (io, nn, preprocess)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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
	
	test_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
				 {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'},
				 {'input_dim': 64, 'output_dim': 1, 'activation': 'relu'}]

	test_nn = nn.NeuralNetwork(nn_arch = test_arch2, lr = 0.001, seed = 29, batch_size = 500,
								epochs = 50, loss_function = "mse")
	
	(test_train_loss, test_val_loss) = test_nn.fit(X_train.T, y_train, X_test.T, y_test)
	
	# plot losses
	plt.figure()
	plt.plot(test_train_loss)
	plt.title("Per Epoch Loss for Training Set")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	
	plt.figure()
	plt.plot(test_val_loss)
	plt.title("Per Epoch Loss for Test Set")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	
	# Evaluate
	model_pred = test_nn.predict(X_test.T)
	model_pred_digit = np.floor(model_pred)
	print('CONFUSION MATRIX')
	print(confusion_matrix(y_test, model_pred_digit))
	# print('CLASSFICATION REPORT')
	# print(classification_report(y_test, model_pred_digit))
	

if __name__ == "__main__":
    main()

