from abc import abstractmethod
from typing import Union

import numpy as np

from abc import abstractmethod

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)
    
class TanhActivation(ActivationLayer):
    """
    Tanh (Hyperbolic Tangent) activation function.
    Squashes values to range -1 and 1
    Formula: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """
    def activation_function(self, input):
        """
        Applies tanh activation
        """
        return np.tanh(input)
    
    def derivative(self, input):
        """
        Computes the derivative of tanh
        Derivative: f'(x) = 1 - tanh^2(x)
        """

        tanh_output = np.tanh(input)
        return 1 - tanh_output ** 2
    
class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function
    Transforms raw output scores into a probability distribution (sums to 1)
    Suitable for multi-class classification problems
    Formula: f(x_i) = e^(x_i) / sum(e^(x_j)) for all j
    """
    def activation_function(self, input):
        """
        Applies softmax activation with numerical stability
        Substracts max value to avoid overflow
        """
        #For numercial stability, substract the max value
        #This prevents overflow from large exponentials
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))

        #Compute softmax: e^x_i / sum(e^x_j)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def derivative(self, input):
        """
        Computes the derivative of softmax
        For simplicity and numerical stability with cross-entropy loss, 
        we return an identity-like gradient that works with the loss derivative

        Note: the full Jacobian of softmax is complex; when used with
        cross-entropy loss, the combined derivative simplifies to (y_pred - y_true)
        """
        #For softmax, the derivative is typically computed together with
        #the cross-entropy loss for numerical stability
        #The full Jacobian is: softmax(x_i)*(1-softmax(x_i)) for i==j
        #                       -softmax(x_i)*softmax(x_j) for i!=j
        #When combined with cross-entropy, this simplifies nicely
        #Return ones as placeholder - the actual gradient is handled
        #in combination with the loss function

        return np.ones_like(input)
    
    
