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
	
	# train to recognize digit 0
	(X_0, y_0) = preprocess.sample_seqs2(X_all, (y_all == 0))
	print(X_0.shape)
	print(y_0.shape)
	X_train, X_test, y_train, y_test = train_test_split(X_0, y_0, test_size=0.33, random_state=42)
	print(X_train.shape)
	print(y_train.shape)
	
	test_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'sigmoid'},
				 {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'},
				 {'input_dim': 64, 'output_dim': 16, 'activation': 'sigmoid'},
				 {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'},
				 {'input_dim': 64, 'output_dim': 1, 'activation': 'sigmoid'}]

	nn_0 = nn.NeuralNetwork(nn_arch = test_arch, lr = 0.001, seed = 29, batch_size = 300,
								epochs = 30, loss_function = "bce")

	(train_loss_0, val_loss_0) = nn_0.fit(X_train.T, y_train*1, X_test.T, y_test*1)

	# plot losses
	plt.figure()
	plt.plot(train_loss_0)
	plt.title("Per Epoch Loss for Training Set")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	
	plt.figure()
	plt.plot(val_loss_0)
	plt.title("Per Epoch Loss for Test Set")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
	
	# Evaluate
	model_prob = nn_0.predict(X_test.T)
	model_pred = (model_prob >= 0.5) * 1
	print(model_prob[0:10])
	print(model_pred[0:10])
	print(y_test[0:10])
	print('CONFUSION MATRIX')
	print(confusion_matrix(((y_test == 0) * 1), model_pred))
	print('CLASSFICATION REPORT')
	print(classification_report(((y_test == 0) * 1), model_pred))
	

if __name__ == "__main__":
    main()

