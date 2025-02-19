# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
import random

# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, Union[int, str]]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. nn_arch = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, 
            {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64 using the relu
            activation function, a 32 dimension hidden layer with the sigmoid activation function,
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self, nn_arch: List[Dict[str, Union[int, str]]], lr: float, seed: int, batch_size: int, epochs: int, loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
        	layer_idx = idx + 1
        	input_dim = layer['input_dim']
        	output_dim = layer['output_dim']
        	# initializing weight matrices
        	param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
        	# initializing bias matrices
        	param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict
    
    def _single_forward(self, W_curr: ArrayLike, b_curr: ArrayLike, A_prev: ArrayLike, activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # print("performing single forward pass")
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        if activation.lower() == "sigmoid":
        	A_curr = self._sigmoid(Z_curr)
        elif activation.lower() == "relu":
        	A_curr = self._relu(Z_curr)
        else:
        	raise ValueError("Please use valid activation function")
        return(A_curr, Z_curr)

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [num_features, batch_size].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # print("FORWARD")
        (num_features, batch_size) = X.shape
        # print("batch size: " + str(batch_size))
        # print("n features: " + str(num_features))
        # initialize cache dictionary
        cache = {}
        cache['X'] = X
        # iterate through layers of neural net
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            # print("layer: " + str(layer_idx))
            act_func = layer['activation']
            # get weight matrices
            W = self._param_dict['W' + str(layer_idx)]
            # get bias matrices
            b = self._param_dict['b' + str(layer_idx)]
            if layer_idx == 1:
            	if num_features == layer['input_dim']:
            		(A, Z) = self._single_forward(W, b, X, act_func)
            	else:
            		raise ValueError("Number of features must be input dim of first layer")
            else:
            	A_prev = cache['A' + str(layer_idx-1)]
            	if len(A_prev) == layer['input_dim']:
            		(A, Z) = self._single_forward(W, b, A_prev, act_func)
            	else:
            		raise ValueError("Problem with layer dimensions")
            # add to cache
            cache['A' + str(layer_idx)] = A
            cache['Z' + str(layer_idx)] = Z
        output = cache['A' + str(len(self.arch))] #is this correct??
        if len(output) != 1:
        	raise ValueError("Final layer must have output dim of 1")
        return(output[0], cache)

    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # print("performing single backward pass")
        # dZ_curr = dA_curr * phi'(Z_curr)
        if activation_curr.lower() == "sigmoid":
        	dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr.lower() == "relu":
        	dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        else:
        	raise ValueError("Current activation function is not valid")
        # dW_curr = dA_curr * phi'(Z_curr) * A_prev = dZ_curr * A_prev
        dW_curr = dZ_curr.dot(A_prev.T) 
        # db_curr = sum of dZ_curr (???)
        db_curr = np.sum(dZ_curr, axis = 1, keepdims = True)
        # dA_prev = dA_curr * phi'(Z_curr) * W_curr = dZ_curr * W_curr
        dA_prev = W_curr.T.dot(dZ_curr)
        return(dA_prev, dW_curr, db_curr)

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # print("BACKWARD")
        # initialize gradient dictionary
        grad_dict = {}
        # iterate through layers of neural net
        for idx, layer in enumerate(reversed(self.arch)):
            layer_idx = len(self.arch) - idx
            # print("layer: " + str(layer_idx))
            act_func = layer['activation']
            # get weight matrices
            W = self._param_dict['W' + str(layer_idx)]
            # get bias matrices
            b = self._param_dict['b' + str(layer_idx)]
            # get A matrices
            if layer_idx == 1:
            	A_prev = cache['X']
            else:
            	A_prev = cache['A' + str(layer_idx-1)]
            # get Z matrices
            Z = cache['Z' + str(layer_idx)]
            if idx == 0:
            	if self._loss_func.lower() == "bce":
            		initial_dA = self._binary_cross_entropy_backprop(y, y_hat)
            	elif self._loss_func.lower() == "mse":
            		initial_dA = self._mean_squared_error_backprop(y, y_hat)
            	elif self._loss_fun.lower() == "other":
            		initial_dA = self._loss_function_backprop(y, y_hat)
            	else:
            		raise ValueError("Please use valid loss function")
            	grad_dict['dA' + str(layer_idx)] = initial_dA
            	(dA_prev, dW, db) = self._single_backprop(W, b, Z, A_prev, initial_dA, act_func)
            else:
            	dA = grad_dict['dA' + str(layer_idx)]
            	(dA_prev, dW, db) = self._single_backprop(W, b, Z, A_prev, dA, act_func)
            # add to gradient dict
            grad_dict['dW' + str(layer_idx)] = dW
            grad_dict['db' + str(layer_idx)] = db   
            grad_dict['dA' + str(layer_idx-1)] = dA_prev        
        return(grad_dict)

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        # print("UPDATING PARAMETERS")
        # iterate through param dictionary
        for idx, layer in enumerate(reversed(self.arch)):
            layer_idx = idx + 1
            # print("layer: " + str(layer_idx))
            # get weight matrix
            W = self._param_dict['W' + str(layer_idx)]
            # get bias matrix
            b = self._param_dict['b' + str(layer_idx)]
            # get weight change matrix
            dW = grad_dict['dW' + str(layer_idx)]
            # get bias change matrix
            db = grad_dict['db' + str(layer_idx)]
            # update weight matrix
            # print("mean weight: " + str(np.mean(W)))
            # print("mean weight change: " + str(np.mean(self._lr * dW)))
            self._param_dict['W' + str(layer_idx)] = W - (self._lr * dW)
            # update bias matrix
            # print("mean bias: " + str(np.mean(b)))
            # print("mean bias change: " + str(np.mean(self._lr * db)))
            self._param_dict['b' + str(layer_idx)] = b - (self._lr * db)
        return

    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # initialize
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        epoch_counter = 0
        # loop through epochs
        while epoch_counter < self._epochs:
        	print("EPOCH: " + str(epoch_counter + 1))
        	# get random batch
        	batch_cols = random.sample(range(len(X_train[0])), self._batch_size)
        	X_batch = X_train[:, batch_cols]
        	y_batch = y_train[batch_cols]
        	# train on random batch
        	(batch_output, batch_cache) = self.forward(X_batch)
        	batch_grad_dict = self.backprop(y_batch, batch_output, batch_cache)
        	self._update_params(batch_grad_dict)
        	# get predictions
        	train_predict = self.predict(X_train)
        	val_predict = self.predict(X_val)
        	print(val_predict[0:10])
        	print(y_val[0:10])
        	if self._loss_func.lower() == "bce":
        		per_epoch_loss_train.append(self._binary_cross_entropy(y_train, train_predict))
        		per_epoch_loss_val.append(self._binary_cross_entropy(y_val, val_predict))
        	elif self._loss_func.lower() == "mse":
        		per_epoch_loss_train.append(self._mean_squared_error(y_train, train_predict))
        		per_epoch_loss_val.append(self._mean_squared_error(y_val, val_predict))
        	elif self._loss_fun.lower() == "other":
        		per_epoch_loss_train.append(self._loss_function(y_train, train_predict))
        		per_epoch_loss_val.append(self._loss_function(y_val, val_predict))
        	else:
        		raise ValueError("Please use valid loss function") 
        	epoch_counter = epoch_counter + 1       
        return(per_epoch_loss_train, per_epoch_loss_val)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # print("MAKING PREDICTION")
        (y_hat,_) = self.forward(X) # is this correct?
        return(y_hat)

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1/(1 + np.exp(-Z))
        return(nl_transform)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(Z, 0)
        return(nl_transform)

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sig_Z = self._sigmoid(Z)
        sig_prime_Z = sig_Z * (1 - sig_Z)
        dZ = dA * sig_prime_Z
        return(dZ)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        relu_prime_Z = (Z > 0) * 1
        dZ = dA * relu_prime_Z
        return(dZ)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        m = len(y)
        bce = -(y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))
        loss = bce.mean()
        return(loss)

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = (-y/y_hat) + ((1 - y)/(1 - y_hat))
        return(dA)

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        mse = np.square(y - y_hat)
        loss = mse.mean()
        return(loss)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = -2 * (y - y_hat)
        return(dA)

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        raise ValueError("Alternate loss function is not implemented")
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        raise ValueError("Alternate loss function backprop is not implemented")
        pass
