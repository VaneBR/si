import unittest
import numpy as np

from si.neural_networks.optimizers import (
    Optimizer,
    SGD,
    Adam
)

class TestAdamInitialization(unittest.TestCase):
    """Testes para inicialização do Adam optimizer"""
    
    def test_adam_inherits_from_optimizer(self):
        """Testa se Adam herda de Optimizer"""
        adam = Adam()
        self.assertIsInstance(adam, Optimizer)
    
    def test_adam_default_initialization(self):
        """Testa inicialização com valores default"""
        adam = Adam()
        
        self.assertEqual(adam.learning_rate, 0.001)
        self.assertEqual(adam.beta_1, 0.9)
        self.assertEqual(adam.beta_2, 0.999)
        self.assertEqual(adam.epsilon, 1e-8)
        self.assertIsNone(adam.m)
        self.assertIsNone(adam.v)
        self.assertEqual(adam.t, 0)
    
    def test_adam_custom_initialization(self):
        """Testa inicialização com valores customizados"""
        adam = Adam(learning_rate=0.01, beta_1=0.95, beta_2=0.998, epsilon=1e-7)
        
        self.assertEqual(adam.learning_rate, 0.01)
        self.assertEqual(adam.beta_1, 0.95)
        self.assertEqual(adam.beta_2, 0.998)
        self.assertEqual(adam.epsilon, 1e-7)
    
    def test_adam_has_update_method(self):
        """Testa se tem método update"""
        adam = Adam()
        self.assertTrue(hasattr(adam, 'update'))
        self.assertTrue(callable(adam.update))


class TestAdamUpdate(unittest.TestCase):
    """Testes para método update do Adam"""
    
    def setUp(self):
        """Setup para cada teste"""
        np.random.seed(42)
        self.adam = Adam(learning_rate=0.01)
        self.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.gradients = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    def test_adam_update_returns_array(self):
        """Testa se update retorna array"""
        updated = self.adam.update(self.weights, self.gradients)
        self.assertIsInstance(updated, np.ndarray)
    
    def test_adam_update_preserves_shape(self):
        """Testa se update preserva shape dos weights"""
        updated = self.adam.update(self.weights, self.gradients)
        self.assertEqual(updated.shape, self.weights.shape)
    
    def test_adam_initializes_m_and_v(self):
        """Testa se inicializa m e v na primeira chamada"""
        self.assertIsNone(self.adam.m)
        self.assertIsNone(self.adam.v)
        
        self.adam.update(self.weights, self.gradients)
        
        self.assertIsNotNone(self.adam.m)
        self.assertIsNotNone(self.adam.v)
        self.assertEqual(self.adam.m.shape, self.weights.shape)
        self.assertEqual(self.adam.v.shape, self.weights.shape)
    
    def test_adam_updates_timestep(self):
        """Testa se incrementa timestep t"""
        self.assertEqual(self.adam.t, 0)
        
        self.adam.update(self.weights, self.gradients)
        self.assertEqual(self.adam.t, 1)
        
        self.adam.update(self.weights, self.gradients)
        self.assertEqual(self.adam.t, 2)
    
    def test_adam_updates_m(self):
        """Testa atualização de m (primeiro momento)"""
        self.adam.update(self.weights, self.gradients)
        
        # m deve ser não-zero após primeira atualização
        self.assertTrue(np.any(self.adam.m != 0))
        
        # m deve ter valores na ordem dos gradientes
        self.assertTrue(np.all(np.abs(self.adam.m) <= np.abs(self.gradients)))
    
    def test_adam_updates_v(self):
        """Testa atualização de v (segundo momento)"""
        self.adam.update(self.weights, self.gradients)
        
        # v deve ser não-zero após primeira atualização
        self.assertTrue(np.any(self.adam.v != 0))
        
        # v deve ser positivo (quadrados de gradientes)
        self.assertTrue(np.all(self.adam.v >= 0))
    
    def test_adam_weights_decrease_with_positive_gradient(self):
        """Testa se weights diminuem com gradiente positivo"""
        positive_gradients = np.ones_like(self.weights)
        updated = self.adam.update(self.weights, positive_gradients)
        
        # Com gradientes positivos, weights devem diminuir
        self.assertTrue(np.all(updated < self.weights))
    
    def test_adam_weights_increase_with_negative_gradient(self):
        """Testa se weights aumentam com gradiente negativo"""
        negative_gradients = -np.ones_like(self.weights)
        updated = self.adam.update(self.weights, negative_gradients)
        
        # Com gradientes negativos, weights devem aumentar
        self.assertTrue(np.all(updated > self.weights))
    
    def test_adam_bias_correction(self):
        """Testa correção de viés (bias correction)"""
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
        
        # Primeira atualização
        updated_1 = adam.update(self.weights, self.gradients)
        m_hat_1 = adam.m / (1 - adam.beta_1 ** adam.t)
        v_hat_1 = adam.v / (1 - adam.beta_2 ** adam.t)
        
        # m_hat e v_hat devem ser maiores que m e v (correção de viés)
        self.assertTrue(np.all(np.abs(m_hat_1) >= np.abs(adam.m)))
        self.assertTrue(np.all(v_hat_1 >= adam.v))
    
    def test_adam_learning_rate_effect(self):
        """Testa efeito do learning rate"""
        adam_small = Adam(learning_rate=0.001)
        adam_large = Adam(learning_rate=0.1)
        
        weights = np.array([[1.0, 2.0]])
        gradients = np.array([[0.5, 0.5]])
        
        updated_small = adam_small.update(weights, gradients)
        updated_large = adam_large.update(weights, gradients)
        
        # Diferença deve ser maior com learning rate maior
        diff_small = np.abs(weights - updated_small)
        diff_large = np.abs(weights - updated_large)
        
        self.assertTrue(np.all(diff_large > diff_small))


class TestAdamConvergence(unittest.TestCase):
    """Testes de convergência do Adam"""
    
    def test_adam_multiple_updates(self):
        """Testa múltiplas atualizações consecutivas"""
        adam = Adam(learning_rate=0.01)
        weights = np.array([[5.0, 5.0]])
        
        # Gradiente constante apontando para zero
        gradient = np.array([[1.0, 1.0]])
        
        for _ in range(10):
            weights = adam.update(weights, gradient)
        
        # Weights devem ter diminuído
        self.assertTrue(np.all(weights < 5.0))
    
    def test_adam_convergence_to_minimum(self):
        """Testa convergência para mínimo simples"""
        adam = Adam(learning_rate=0.1)
        
        # Função quadrática simples: f(x) = x^2
        # Gradiente: f'(x) = 2x
        # Mínimo em x = 0
        
        x = np.array([[5.0]])
        
        for _ in range(100):
            gradient = 2 * x  # Gradiente de x^2
            x = adam.update(x, gradient)
        
        # Deve estar próximo de zero
        self.assertLess(np.abs(x[0, 0]), 0.5)
    
    def test_adam_vs_sgd_convergence(self):
        """Compara convergência Adam vs SGD"""
        adam = Adam(learning_rate=0.01)
        sgd = SGD(learning_rate=0.01, momentum=0.0)
        
        weights_adam = np.array([[10.0]])
        weights_sgd = np.array([[10.0]])
        
        gradient = np.array([[2.0]])
        
        # Várias iterações
        for _ in range(5):
            weights_adam = adam.update(weights_adam, gradient)
            weights_sgd = sgd.update(weights_sgd, gradient)
        
        # Ambos devem ter diminuído
        self.assertLess(weights_adam[0, 0], 10.0)
        self.assertLess(weights_sgd[0, 0], 10.0)


class TestAdamMomentumBehavior(unittest.TestCase):
    """Testes para comportamento de momentum do Adam"""
    
    def test_adam_accumulates_momentum(self):
        """Testa se Adam acumula momentum"""
        adam = Adam(learning_rate=0.01, beta_1=0.9)
        weights = np.array([[1.0]])
        gradient = np.array([[0.1]])
        
        # Primeira atualização
        adam.update(weights, gradient)
        m_after_1 = adam.m.copy()
        
        # Segunda atualização com mesmo gradiente
        adam.update(weights, gradient)
        m_after_2 = adam.m.copy()
        
        # m deve ter aumentado (acumulação de momentum)
        self.assertTrue(np.all(np.abs(m_after_2) > np.abs(m_after_1)))
    
    def test_adam_adapts_to_gradient_changes(self):
        """Testa adaptação a mudanças de gradiente"""
        adam = Adam(learning_rate=0.01)
        weights = np.array([[1.0, 1.0]])
        
        # Gradientes alternados
        grad1 = np.array([[1.0, 0.0]])
        grad2 = np.array([[0.0, 1.0]])
        
        for _ in range(5):
            weights = adam.update(weights, grad1)
            weights = adam.update(weights, grad2)
        
        # Adam deve ter adaptado para ambas as direções
        self.assertTrue(np.any(adam.m != 0))
        self.assertTrue(np.any(adam.v != 0))


class TestAdamEdgeCases(unittest.TestCase):
    """Testes para casos extremos do Adam"""
    
    def test_adam_zero_gradient(self):
        """Testa com gradiente zero"""
        adam = Adam(learning_rate=0.01)
        weights = np.array([[1.0, 2.0]])
        zero_gradient = np.zeros_like(weights)
        
        updated = adam.update(weights, zero_gradient)
        
        # Weights não devem mudar significativamente
        np.testing.assert_array_almost_equal(updated, weights, decimal=6)
    
    def test_adam_large_gradient(self):
        """Testa com gradiente muito grande"""
        adam = Adam(learning_rate=0.01)
        weights = np.array([[1.0]])
        large_gradient = np.array([[1000.0]])
        
        updated = adam.update(weights, large_gradient)
        
        # Não deve dar NaN ou Inf
        self.assertFalse(np.any(np.isnan(updated)))
        self.assertFalse(np.any(np.isinf(updated)))
    
    def test_adam_single_weight(self):
        """Testa com um único weight"""
        adam = Adam(learning_rate=0.01)
        weight = np.array([5.0])
        gradient = np.array([1.0])
        
        updated = adam.update(weight, gradient)
        
        self.assertEqual(updated.shape, weight.shape)
        self.assertLess(updated[0], weight[0])
    
    def test_adam_different_shapes(self):
        """Testa com diferentes shapes de weights"""
        adam = Adam(learning_rate=0.01)
        
        shapes = [(10,), (5, 5), (2, 3, 4), (100,)]
        
        for shape in shapes:
            weights = np.random.randn(*shape)
            gradients = np.random.randn(*shape)
            
            updated = adam.update(weights, gradients)
            
            self.assertEqual(updated.shape, shape)


class TestAdamHyperparameters(unittest.TestCase):
    """Testes para efeito dos hiperparâmetros"""
    
    def test_adam_beta1_effect(self):
        """Testa efeito de beta_1 no momentum"""
        adam_low = Adam(learning_rate=0.01, beta_1=0.5)
        adam_high = Adam(learning_rate=0.01, beta_1=0.95)
        
        weights = np.array([[1.0]])
        gradient = np.array([[0.5]])
        
        # Múltiplas iterações
        for _ in range(5):
            adam_low.update(weights, gradient)
            adam_high.update(weights, gradient)
        
        # Beta_1 maior deve acumular mais momentum
        self.assertGreater(np.abs(adam_high.m[0]), np.abs(adam_low.m[0]))
    
    def test_adam_beta2_effect(self):
        """Testa efeito de beta_2 na variância"""
        adam_low = Adam(learning_rate=0.01, beta_2=0.9)
        adam_high = Adam(learning_rate=0.01, beta_2=0.999)
        
        weights = np.array([[1.0]])
        gradient = np.array([[0.5]])
        
        # Múltiplas iterações
        for _ in range(5):
            adam_low.update(weights, gradient)
            adam_high.update(weights, gradient)
        
        # Beta_2 maior deve manter v maior (decay mais lento)
        self.assertGreater(adam_high.v[0], adam_low.v[0])
    
    def test_adam_epsilon_prevents_division_by_zero(self):
        """Testa se epsilon previne divisão por zero"""
        adam = Adam(learning_rate=0.01, epsilon=1e-8)
        
        weights = np.array([[1.0]])
        tiny_gradient = np.array([[1e-10]])
        
        # Mesmo com gradiente minúsculo, não deve dar erro
        updated = adam.update(weights, tiny_gradient)
        
        self.assertFalse(np.any(np.isnan(updated)))
        self.assertFalse(np.any(np.isinf(updated)))

if __name__ == '__main__':
    unittest.main()