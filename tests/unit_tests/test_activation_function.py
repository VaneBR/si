import unittest
from unittest import TestCase


import os

from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split

import numpy as np
from si.neural_networks.activation import (
    ActivationLayer, 
    TanhActivation, 
    SoftmaxActivation,
    SigmoidActivation,
    ReLUActivation
)

class TestSigmoidLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestRELULayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestTanhActivationInitialization(unittest.TestCase):
    """Testes para inicialização da TanhActivation"""
    
    def test_tanh_inherits_from_activation_layer(self):
        """Testa se TanhActivation herda de ActivationLayer"""
        tanh = TanhActivation()
        self.assertIsInstance(tanh, ActivationLayer)
    
    def test_tanh_initialization(self):
        """Testa inicialização básica"""
        tanh = TanhActivation()
        self.assertIsNotNone(tanh)
    
    def test_tanh_has_required_methods(self):
        """Testa se tem todos os métodos necessários"""
        tanh = TanhActivation()
        self.assertTrue(hasattr(tanh, 'activation_function'))
        self.assertTrue(hasattr(tanh, 'derivative'))
        self.assertTrue(hasattr(tanh, 'forward_propagation'))
        self.assertTrue(hasattr(tanh, 'backward_propagation'))
        self.assertTrue(hasattr(tanh, 'output_shape'))
        self.assertTrue(hasattr(tanh, 'parameters'))


class TestTanhActivationFunction(unittest.TestCase):
    """Testes para função de ativação Tanh"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.tanh = TanhActivation()
    
    def test_tanh_zero(self):
        """Testa tanh(0) = 0"""
        result = self.tanh.activation_function(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.0, places=10)
    
    def test_tanh_positive_values(self):
        """Testa tanh com valores positivos"""
        input_data = np.array([1.0, 2.0, 3.0])
        result = self.tanh.activation_function(input_data)
        
        # Tanh deve retornar valores entre 0 e 1 para positivos
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))
    
    def test_tanh_negative_values(self):
        """Testa tanh com valores negativos"""
        input_data = np.array([-1.0, -2.0, -3.0])
        result = self.tanh.activation_function(input_data)
        
        # Tanh deve retornar valores entre -1 e 0 para negativos
        self.assertTrue(np.all(result < 0))
        self.assertTrue(np.all(result > -1))
    
    def test_tanh_range(self):
        """Testa se tanh está no intervalo [-1, 1]"""
        input_data = np.linspace(-10, 10, 100)
        result = self.tanh.activation_function(input_data)
        
        self.assertTrue(np.all(result >= -1))
        self.assertTrue(np.all(result <= 1))
    
    def test_tanh_symmetry(self):
        """Testa simetria: tanh(-x) = -tanh(x)"""
        x = np.array([1.0, 2.0, 3.0])
        result_pos = self.tanh.activation_function(x)
        result_neg = self.tanh.activation_function(-x)
        
        np.testing.assert_array_almost_equal(result_neg, -result_pos)
    
    def test_tanh_asymptotic_behavior(self):
        """Testa comportamento assintótico"""
        # Para x muito grande, tanh(x) → 1
        large_positive = self.tanh.activation_function(np.array([10.0]))
        self.assertAlmostEqual(large_positive[0], 1.0, places=5)
        
        # Para x muito negativo, tanh(x) → -1
        large_negative = self.tanh.activation_function(np.array([-10.0]))
        self.assertAlmostEqual(large_negative[0], -1.0, places=5)
    
    def test_tanh_shape_preservation(self):
        """Testa se mantém o shape do input"""
        shapes = [(5,), (3, 4), (2, 3, 4)]
        
        for shape in shapes:
            input_data = np.random.randn(*shape)
            result = self.tanh.activation_function(input_data)
            self.assertEqual(result.shape, shape)
    
    def test_tanh_known_values(self):
        """Testa valores conhecidos de tanh"""
        # tanh(0) = 0
        self.assertAlmostEqual(
            self.tanh.activation_function(np.array([0.0]))[0], 
            0.0, 
            places=10
        )
        
        # tanh(∞) ≈ 1
        self.assertAlmostEqual(
            self.tanh.activation_function(np.array([100.0]))[0], 
            1.0, 
            places=5
        )


class TestTanhDerivative(unittest.TestCase):
    """Testes para derivada de Tanh"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.tanh = TanhActivation()
    
    def test_tanh_derivative_formula(self):
        """Testa se derivada segue fórmula: 1 - tanh²(x)"""
        x = np.array([0.0, 1.0, -1.0, 2.0])
        
        # Calcular derivada
        derivative = self.tanh.derivative(x)
        
        # Calcular usando fórmula: 1 - tanh²(x)
        tanh_x = np.tanh(x)
        expected = 1 - tanh_x ** 2
        
        np.testing.assert_array_almost_equal(derivative, expected)
    
    def test_tanh_derivative_at_zero(self):
        """Testa derivada em x=0: f'(0) = 1"""
        derivative = self.tanh.derivative(np.array([0.0]))
        self.assertAlmostEqual(derivative[0], 1.0, places=10)
    
    def test_tanh_derivative_range(self):
        """Testa se derivada está no intervalo (0, 1]"""
        x = np.linspace(-5, 5, 100)
        derivative = self.tanh.derivative(x)
        
        self.assertTrue(np.all(derivative > 0))
        self.assertTrue(np.all(derivative <= 1))
    
    def test_tanh_derivative_decreases_from_zero(self):
        """Testa se derivada diminui conforme |x| aumenta"""
        x_values = [0.0, 1.0, 2.0, 3.0]
        derivatives = [self.tanh.derivative(np.array([x]))[0] for x in x_values]
        
        # Derivadas devem diminuir
        for i in range(len(derivatives) - 1):
            self.assertGreater(derivatives[i], derivatives[i + 1])
    
    def test_tanh_derivative_symmetry(self):
        """Testa simetria: f'(-x) = f'(x)"""
        x = np.array([1.0, 2.0, 3.0])
        derivative_pos = self.tanh.derivative(x)
        derivative_neg = self.tanh.derivative(-x)
        
        np.testing.assert_array_almost_equal(derivative_pos, derivative_neg)
    
    def test_tanh_derivative_shape(self):
        """Testa se derivada mantém shape"""
        shapes = [(5,), (3, 4), (2, 3, 4)]
        
        for shape in shapes:
            x = np.random.randn(*shape)
            derivative = self.tanh.derivative(x)
            self.assertEqual(derivative.shape, shape)


class TestTanhForwardBackward(unittest.TestCase):
    """Testes para forward e backward propagation de Tanh"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.tanh = TanhActivation()
    
    def test_tanh_forward_propagation(self):
        """Testa forward propagation"""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        output = self.tanh.forward_propagation(input_data, training=True)
        
        # Verifica shape
        self.assertEqual(output.shape, input_data.shape)
        
        # Verifica se armazenou input
        np.testing.assert_array_equal(self.tanh.input, input_data)
        
        # Verifica se output é tanh(input)
        expected = np.tanh(input_data)
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_tanh_backward_propagation(self):
        """Testa backward propagation"""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Forward primeiro
        self.tanh.forward_propagation(input_data, training=True)
        
        # Backward
        output_error = np.ones_like(input_data)
        input_error = self.tanh.backward_propagation(output_error)
        
        # Verifica shape
        self.assertEqual(input_error.shape, output_error.shape)
        
        # Verifica se é derivative * output_error
        expected = self.tanh.derivative(input_data) * output_error
        np.testing.assert_array_almost_equal(input_error, expected)
    
    def test_tanh_gradient_flow(self):
        """Testa fluxo de gradientes"""
        x = np.random.randn(5, 3)
        
        # Forward
        output = self.tanh.forward_propagation(x, training=True)
        
        # Backward com gradiente conhecido
        gradient = np.ones_like(output)
        input_gradient = self.tanh.backward_propagation(gradient)
        
        # Gradiente deve ter mesmo shape
        self.assertEqual(input_gradient.shape, x.shape)
        
        # Valores do gradiente devem estar em (0, 1]
        self.assertTrue(np.all(input_gradient > 0))
        self.assertTrue(np.all(input_gradient <= 1))


class TestSoftmaxActivationInitialization(unittest.TestCase):
    """Testes para inicialização da SoftmaxActivation"""
    
    def test_softmax_inherits_from_activation_layer(self):
        """Testa se SoftmaxActivation herda de ActivationLayer"""
        softmax = SoftmaxActivation()
        self.assertIsInstance(softmax, ActivationLayer)
    
    def test_softmax_initialization(self):
        """Testa inicialização básica"""
        softmax = SoftmaxActivation()
        self.assertIsNotNone(softmax)
    
    def test_softmax_has_required_methods(self):
        """Testa se tem todos os métodos necessários"""
        softmax = SoftmaxActivation()
        self.assertTrue(hasattr(softmax, 'activation_function'))
        self.assertTrue(hasattr(softmax, 'derivative'))
        self.assertTrue(hasattr(softmax, 'forward_propagation'))
        self.assertTrue(hasattr(softmax, 'backward_propagation'))


class TestSoftmaxActivationFunction(unittest.TestCase):
    """Testes para função de ativação Softmax"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.softmax = SoftmaxActivation()
    
    def test_softmax_probabilities_sum_to_one(self):
        """Testa se probabilidades somam 1"""
        input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                               [2.0, 2.0, 2.0, 2.0]])
        result = self.softmax.activation_function(input_data)
        
        # Soma de cada linha deve ser 1
        sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(2))
    
    def test_softmax_all_positive(self):
        """Testa se todos os valores são positivos"""
        input_data = np.random.randn(5, 4)
        result = self.softmax.activation_function(input_data)
        
        self.assertTrue(np.all(result > 0))
    
    def test_softmax_range(self):
        """Testa se valores estão no intervalo [0, 1]"""
        input_data = np.random.randn(10, 5)
        result = self.softmax.activation_function(input_data)
        
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
    
    def test_softmax_uniform_input(self):
        """Testa softmax com input uniforme"""
        # Todos iguais → probabilidades iguais
        input_data = np.array([[1.0, 1.0, 1.0, 1.0]])
        result = self.softmax.activation_function(input_data)
        
        expected = np.array([[0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_softmax_max_input(self):
        """Testa se maior input tem maior probabilidade"""
        input_data = np.array([[1.0, 5.0, 2.0, 3.0]])
        result = self.softmax.activation_function(input_data)
        
        # Índice 1 (valor 5.0) deve ter maior probabilidade
        max_prob_idx = np.argmax(result[0])
        self.assertEqual(max_prob_idx, 1)
    
    def test_softmax_numerical_stability(self):
        """Testa estabilidade numérica com valores grandes"""
        # Valores muito grandes podem causar overflow sem estabilização
        input_data = np.array([[1000.0, 1001.0, 1002.0]])
        result = self.softmax.activation_function(input_data)
        
        # Não deve ter NaN ou Inf
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        
        # Soma deve ser 1
        self.assertAlmostEqual(np.sum(result), 1.0, places=5)
    
    def test_softmax_negative_inputs(self):
        """Testa softmax com inputs negativos"""
        input_data = np.array([[-1.0, -2.0, -3.0, -4.0]])
        result = self.softmax.activation_function(input_data)
        
        # Ainda deve somar 1
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)
        
        # Maior valor (-1.0) deve ter maior probabilidade
        self.assertEqual(np.argmax(result[0]), 0)
    
    def test_softmax_shape_preservation(self):
        """Testa se mantém shape (n_samples, n_classes)"""
        shapes = [(1, 3), (5, 4), (10, 10), (100, 5)]
        
        for shape in shapes:
            input_data = np.random.randn(*shape)
            result = self.softmax.activation_function(input_data)
            self.assertEqual(result.shape, shape)
    
    def test_softmax_batch_processing(self):
        """Testa processamento em batch"""
        # Batch de 3 samples, 4 classes cada
        input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                               [4.0, 3.0, 2.0, 1.0],
                               [2.0, 2.0, 2.0, 2.0]])
        result = self.softmax.activation_function(input_data)
        
        # Cada linha deve somar 1
        for i in range(3):
            self.assertAlmostEqual(np.sum(result[i]), 1.0, places=10)
    
    def test_softmax_temperature_effect(self):
        """Testa efeito da magnitude dos inputs"""
        base_input = np.array([[1.0, 2.0, 3.0]])
        
        # Input normal
        result_normal = self.softmax.activation_function(base_input)
        
        # Input com valores maiores (mais confiante)
        result_high = self.softmax.activation_function(base_input * 10)
        
        # Probabilidade do máximo deve ser maior com valores mais altos
        self.assertGreater(np.max(result_high), np.max(result_normal))


class TestSoftmaxDerivative(unittest.TestCase):
    """Testes para derivada de Softmax"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.softmax = SoftmaxActivation()
    
    def test_softmax_derivative_returns_ones(self):
        """
        Testa se derivada retorna ones (placeholder).
        A derivada real é complexa e calculada junto com a loss.
        """
        input_data = np.array([[1.0, 2.0, 3.0]])
        derivative = self.softmax.derivative(input_data)
        
        np.testing.assert_array_equal(derivative, np.ones_like(input_data))
    
    def test_softmax_derivative_shape(self):
        """Testa se derivada mantém shape"""
        shapes = [(1, 3), (5, 4), (10, 10)]
        
        for shape in shapes:
            input_data = np.random.randn(*shape)
            derivative = self.softmax.derivative(input_data)
            self.assertEqual(derivative.shape, shape)
    
    def test_softmax_derivative_with_cross_entropy(self):
        """
        Testa conceito: softmax + cross-entropy simplifica para (y_pred - y_true).
        Este é um teste conceitual, não da implementação atual.
        """
        # Quando combinado com cross-entropy, o gradiente é simples
        # Este teste apenas documenta o conceito
        y_pred = np.array([[0.1, 0.7, 0.2]])  # Softmax output
        y_true = np.array([[0, 1, 0]])         # One-hot
        
        # Gradiente combinado é simplesmente:
        combined_gradient = y_pred - y_true
        
        # Verificar que é simples e não envolve Jacobiano complexo
        self.assertEqual(combined_gradient.shape, y_pred.shape)


class TestSoftmaxForwardBackward(unittest.TestCase):
    """Testes para forward e backward propagation de Softmax"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.softmax = SoftmaxActivation()
    
    def test_softmax_forward_propagation(self):
        """Testa forward propagation"""
        input_data = np.array([[1.0, 2.0, 3.0],
                               [3.0, 2.0, 1.0]])
        output = self.softmax.forward_propagation(input_data, training=True)
        
        # Verifica shape
        self.assertEqual(output.shape, input_data.shape)
        
        # Verifica se armazenou input
        np.testing.assert_array_equal(self.softmax.input, input_data)
        
        # Verifica se cada linha soma 1
        for i in range(output.shape[0]):
            self.assertAlmostEqual(np.sum(output[i]), 1.0, places=10)
    
    def test_softmax_backward_propagation(self):
        """Testa backward propagation"""
        input_data = np.array([[1.0, 2.0, 3.0]])
        
        # Forward primeiro
        self.softmax.forward_propagation(input_data, training=True)
        
        # Backward
        output_error = np.array([[0.1, 0.2, 0.3]])
        input_error = self.softmax.backward_propagation(output_error)
        
        # Verifica shape
        self.assertEqual(input_error.shape, output_error.shape)
    
    def test_softmax_gradient_flow(self):
        """Testa fluxo de gradientes"""
        x = np.random.randn(5, 4)
        
        # Forward
        output = self.softmax.forward_propagation(x, training=True)
        
        # Backward
        gradient = np.random.randn(*output.shape)
        input_gradient = self.softmax.backward_propagation(gradient)
        
        # Shape deve ser preservado
        self.assertEqual(input_gradient.shape, x.shape)


class TestActivationLayerGeneral(unittest.TestCase):
    """Testes gerais para todas as activation layers"""
    
    def test_all_activations_output_shape(self):
        """Testa output_shape para todas as activations"""
        activations = [
            ReLUActivation(),
            SigmoidActivation(),
            TanhActivation(),
            SoftmaxActivation()
        ]
        
        for activation in activations:
            activation.set_input_shape((10, 20))
            self.assertEqual(activation.output_shape(), (10, 20))
    
    def test_all_activations_parameters(self):
        """Testa que todas as activations têm 0 parâmetros"""
        activations = [
            ReLUActivation(),
            SigmoidActivation(),
            TanhActivation(),
            SoftmaxActivation()
        ]
        
        for activation in activations:
            self.assertEqual(activation.parameters(), 0)
    
    def test_all_activations_layer_name(self):
        """Testa layer_name para todas as activations"""
        activations = [
            (ReLUActivation(), "ReLUActivation"),
            (SigmoidActivation(), "SigmoidActivation"),
            (TanhActivation(), "TanhActivation"),
            (SoftmaxActivation(), "SoftmaxActivation")
        ]
        
        for activation, expected_name in activations:
            self.assertEqual(activation.layer_name(), expected_name)


class TestTanhVsSigmoid(unittest.TestCase):
    """Testes comparativos entre Tanh e Sigmoid"""
    
    def test_tanh_centered_at_zero(self):
        """Testa que Tanh é centrada em zero, Sigmoid não"""
        tanh = TanhActivation()
        sigmoid = SigmoidActivation()
        
        # tanh(0) = 0
        self.assertAlmostEqual(tanh.activation_function(np.array([0.0]))[0], 0.0)
        
        # sigmoid(0) = 0.5
        self.assertAlmostEqual(sigmoid.activation_function(np.array([0.0]))[0], 0.5)
    
    def test_tanh_range_vs_sigmoid_range(self):
        """Testa ranges diferentes"""
        x = np.linspace(-5, 5, 100)
        
        tanh = TanhActivation()
        sigmoid = SigmoidActivation()
        
        tanh_output = tanh.activation_function(x)
        sigmoid_output = sigmoid.activation_function(x)
        
        # Tanh: [-1, 1]
        self.assertTrue(np.all(tanh_output >= -1))
        self.assertTrue(np.all(tanh_output <= 1))
        
        # Sigmoid: [0, 1]
        self.assertTrue(np.all(sigmoid_output >= 0))
        self.assertTrue(np.all(sigmoid_output <= 1))


class TestSoftmaxEdgeCases(unittest.TestCase):
    """Testes para casos extremos de Softmax"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.softmax = SoftmaxActivation()
    
    def test_softmax_single_class(self):
        """Testa softmax com uma única classe"""
        input_data = np.array([[5.0]])
        result = self.softmax.activation_function(input_data)
        
        # Deve ser 1.0 (única probabilidade)
        self.assertAlmostEqual(result[0, 0], 1.0)
    
    def test_softmax_two_classes(self):
        """Testa softmax com duas classes (caso binário)"""
        input_data = np.array([[1.0, 2.0]])
        result = self.softmax.activation_function(input_data)
        
        # Soma deve ser 1
        self.assertAlmostEqual(np.sum(result), 1.0)
        
        # Classe 2 deve ter maior probabilidade
        self.assertGreater(result[0, 1], result[0, 0])
    
    def test_softmax_with_zeros(self):
        """Testa softmax com zeros no input"""
        input_data = np.array([[0.0, 0.0, 0.0, 0.0]])
        result = self.softmax.activation_function(input_data)
        
        # Todas as probabilidades devem ser iguais
        expected = np.array([[0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_softmax_extreme_values(self):
        """Testa com valores extremamente diferentes"""
        input_data = np.array([[1.0, 100.0, 2.0]])
        result = self.softmax.activation_function(input_data)
        
        # Classe do meio deve ter probabilidade ~1
        self.assertGreater(result[0, 1], 0.99)