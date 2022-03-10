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
	# load dataset
	positives = io.read_text_file("data/rap1-lieb-positives.txt")
	negatives = io.read_fasta_file("data/yeast-upstream-1k-negative.fa")
	
	"""
	Because the negative sequences are longer than the positive sequences, I wrote a 
	function in preprocess.py, trim_seqs, that randomly selects a consecutive sequence
	of 17 nucleotides from each negative sequence.
	"""
	
	# trim negative sequences
	trim_negatives = preprocess.trim_seqs(negatives, len(positives[0]))
	
	"""
	Because there are more negative than positive examples in this dataset, I use an 
	upsampling scheme to increase the number of positive examples to match the number
	of negative examples. In preprocess.py, the sample_seqs function samples from 
	the positive (minority class) with replacement until there are the same number as the 
	negative (majority class) samples. 
	"""
	
	# generate training set
	all_seqs = np.asarray(positives + trim_negatives)
	print(all_seqs.shape)
	all_labels = np.asarray([True] * len(positives) + [False] * len(trim_negatives))
	print(all_labels.shape)
	(upsampled_seqs, upsampled_labels) = preprocess.sample_seqs(all_seqs, all_labels)
	print(upsampled_seqs.shape)
	print(upsampled_labels.shape)
	X_train, X_test, y_train, y_test = train_test_split(upsampled_seqs, upsampled_labels, 
														test_size=0.2, random_state=42)	
	y_train = y_train * 1
	y_test = y_test * 1
	
	# one hot encode sequences
	X_train_encode = np.asarray(preprocess.one_hot_encode_seqs(X_train))
	X_test_encode = np.asarray(preprocess.one_hot_encode_seqs(X_test))
	print(X_train_encode.shape)
	print(X_test_encode.shape)

	# train neural network
	test_arch = [{'input_dim': 68, 'output_dim': 17, 'activation': 'sigmoid'},
				  {'input_dim': 17, 'output_dim': 68, 'activation': 'sigmoid'},
				  {'input_dim': 68, 'output_dim': 1, 'activation': 'sigmoid'}]
	
	"""
	Selection of hyperparamaters:
	I selected the Binary Cross Entropy loss function because the network is using logistic
	regression to separate the input into two classes: Rap1 binding site or non-binding site.
	The other hyperparameters were selected through trial and error until I got acceptable
	results.
	"""
	
	test_nn = nn.NeuralNetwork(nn_arch = test_arch, lr = 0.005, seed = 29, batch_size = 300,
								epochs = 50, loss_function = "bce")
	
	(test_train_loss, test_val_loss) = test_nn.fit(X_train_encode.T, y_train, X_test_encode.T, y_test)
	
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
	model_prob = test_nn.predict(X_test_encode.T)
	model_pred = (model_prob >= 0.5) * 1
	print('CONFUSION MATRIX')
	print(confusion_matrix(y_test, model_pred))
	print('CLASSFICATION REPORT')
	print(classification_report(y_test, model_pred))
	
	# ROC Curve
	logit_roc_auc = roc_auc_score(y_test, model_pred)
	fpr, tpr, thresholds = roc_curve(y_test, model_prob)
	plt.figure()
	plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	# plt.savefig('TF_classifer_Log_ROC')
	plt.show()
	

if __name__ == "__main__":
    main()

