from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    
class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer
    Combines ideas from RMSprop and SGD with momentum
    Uses moving averages of gradients and squared gradients
    """
def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2= 0.999, epsilon=1e-8):
    super().__init__(learning_rate)
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon

    #Moving averages (initialized later)
    self.m = None #1st moment (mean of gradients)
    self.v = None #2nd moment (uncentered variance of gradients)
    self.t = 0    #Time step (epoch counter)

def update(self,w,grad_loss_w):
    """
    Updates parameters using Adam algorithm
    """

    #Initialize m and v if not yet initialized
    if self.m is None:
        self.m = np.zeros_like(w)
    if self.v is None:
        self.v = np.zeros_like(w)
    
    #Update time step
    self.t +=1

    #Update biased first moment estimate
    #m_t = beta_1*m_{t-1} + (1-beta_1)*grad
    self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

    #Update biased second moment estimate
    #v_t = beta_2*v_{t-1} + (1-beta_2)*(grad**2)
    self.v = self.beta_2 * self.v + (1 - self.beta_2)* (grad_loss_w ** 2)

    #Compute bias-corrected first moment estimate
    #m_hat=m_t / (1 - beta_1**t)
    m_hat = self.m / (1 - self.beta_1 ** self.t)

    #Compute bias-corrected second moment estimate
    #v_hat=v_t / (1 - beta_2**t)
    v_hat = self.v / (1 - self.beta_2 ** self.t)

    #Update weights
    #w_new = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

