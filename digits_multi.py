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

	# split into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=42)
	print(X_train.shape)
	print(y_train.shape)
	
	test_arch = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'},
				 {'input_dim': 32, 'output_dim': 16, 'activation': 'relu'},
				 {'input_dim': 16, 'output_dim': 32, 'activation': 'relu'},
				 {'input_dim': 32, 'output_dim': 64, 'activation': 'relu'},
				 {'input_dim': 64, 'output_dim': 1, 'activation': 'sigmoid'}]

	nn_auto = nn.NeuralNetwork(nn_arch = test_arch, lr = 0.001, seed = 29, batch_size = 300,
								epochs = 55, loss_function = "mse")

	# because the last layer of the network is a sigmoid function the outputs are between 0 and 1
	# so I multiply the y arrays by 0.1 in order to match
	(train_auto_loss, val_auto_loss) = nn_auto.fit(X_train.T, y_train * 0.1, X_test.T, y_test * 0.1)
	
	print("DONE")
	# plot losses
	plt.figure()
	plt.plot(train_auto_loss)
	plt.title("Per Epoch Loss for Training Set")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	
	plt.figure()
	plt.plot(val_auto_loss)
	plt.title("Per Epoch Loss for Test Set")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	
	# Evaluate
	model_prob = nn_auto.predict(X_test.T)
	model_pred = np.floor(model_prob * 10)
	print(model_prob[0:10])
	print(model_pred[0:10])
	print(y_test[0:10])
	print('CONFUSION MATRIX')
	print(confusion_matrix(y_test, model_pred))
	print('CLASSFICATION REPORT')
	print(classification_report(y_test, model_pred))
	
	# Reconstruction Error
	(y_hat,cache) = nn_auto.forward(X_test.T)
	# the reconstructed images are stored in Ag -- the output of the second-to-last layer
	reconstruction = cache['A4']
	print("RECONSTRUCTION ERROR (MSE)")
	reconstruction_error = nn_auto._mean_squared_error(X_test.T, reconstruction)
	print(reconstruction_error)
	print("PREDICTION ERROR (MSE)")
	prediction_error = nn_auto._mean_squared_error(y_test, model_pred)
	print(prediction_error)
	print("On average, the predicted digit is within " + str(round(np.sqrt(prediction_error), 2)) + " of the actual digit")
	

if __name__ == "__main__":
    main()

