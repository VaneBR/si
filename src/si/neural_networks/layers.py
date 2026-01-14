from abc import ABCMeta, abstractmethod
import copy

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    
class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

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
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)

        # computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
    
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,) 

class Dropout(Layer):
    """
    Dropout layer - Técnica de regularização
    
    Durante o treino, desativa aleatoriamente uma fração de neurônios
    (definida por probability), ajudando a prevenir overfitting

    Durante a inferencia, todos os neuronios são mantidos ativos

    A saída é escalada durante o treino para manter a mesma magnitude
    esperada durante a inferencia (inverted dropout)
    """

    def __init__(self, probability:float):
        """
        Inicializa a camada de Dropout
        """
        super().__init__()
        
        if not 0 <= probability < 1:
            raise ValueError("probability deve estar entre 0 e 1, recebeu {probability}")
        
        self.probability = probability

        #Parametros estimados
        self.mask = None
        self.input = None
        self.output = None
    
    def forward_propagation(self, input_data, training=True):
        """
        Forward propagation com dropout
        
        Modo Treino: 
        1. Calcular fator de escala: 1/(1 - probability)
        2. Gerar máscara binomial com probabilidade (1-probability)
        3. Aplicar máscara e escalar: output = input * mas * scaling_factor
        
        Modo Inferencia:
        - Retornar input sem modificações
        """

        self.input = input_data

        if training:
            #Modo Treino: aplicar dropout
            #1. Calcular fator de escala (inverted dropout)
            #Isso mantem a magnitude esperada durante a inferencia
            scaling_factor = 1.0 / (1.0 - self.probability)

            #2. Gerar máscara binomial
            #Probabilidade de manter o neuronio = (1-probability)
            #Se probability=0.5, então 50% serão mantidos (1) e 50 % zerados (0)
            #Se probability=0.2, então 80% serão mantidos (1) e 20 % zerados (0)
            self.mask = np.random.binomial(
                n=1,                            #Bernoulli trial (0 ou 1)
                p=(1.0 - self.probability),     #Prob de manter (não dropar)
                size=input_data.shape           #Mesma forma que input
            )

            #3. Aplicar máscara e escalar
            self.output = input_data * self.mask * scaling_factor

        else:
        # Modo Inferencia: não aplicar dropout
        #Todos os neuronios estão ativos
            self.output = input_data
            self.mask=None #Não precisamos de mascara na inferencia
        
        return self.output
    
    def backward_propagation(self, output_error):
        """
        Backward propagation com dropout

        Simplesmente multiplica o erro pela mascara usada no forward
        Isto garante que o gradiente seja zero para neuronios que foram
        desativados (dropados)
        """

        if self.mask is None:
            #Multiplicar erro pela mascara
            #Neuronios que foram zerados no forward também tem gradiente zer
            return output_error * self.mask
        else:
            #Se não há mascara (inferencia), passar erro diretamente
            return output_error
        
    def output_shape(self):
        """
        Retorna a forma da saída 
        Dropout não altera a forma dos dados
        """
        return self.input_shape()
    
    def parameters(self):
        """
        Retorna o número de parâmetros treinaveis
        Dropout não tem parâmetros treináveis
        """
        return 0    