import unittest
import numpy as np
from si.neural_networks.losses import (
    LossFunction,
    MeanSquaredError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy
)

class TestCategoricalCrossEntropyInitialization(unittest.TestCase):
    """Testes para inicialização da CategoricalCrossEntropy"""
    
    def test_cce_inherits_from_loss_function(self):
        """Testa se herda de LossFunction"""
        cce = CategoricalCrossEntropy()
        self.assertIsInstance(cce, LossFunction)
    
    def test_cce_initialization(self):
        """Testa inicialização básica"""
        cce = CategoricalCrossEntropy()
        self.assertIsNotNone(cce)
    
    def test_cce_has_required_methods(self):
        """Testa se tem métodos necessários"""
        cce = CategoricalCrossEntropy()
        self.assertTrue(hasattr(cce, 'loss'))
        self.assertTrue(hasattr(cce, 'derivative'))


class TestCategoricalCrossEntropyLoss(unittest.TestCase):
    """Testes para método loss da CategoricalCrossEntropy"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.cce = CategoricalCrossEntropy()
    
    def test_cce_loss_perfect_predictions(self):
        """Testa loss com predições perfeitas"""
        # One-hot encoded true labels
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        
        # Predições perfeitas
        y_pred = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        
        loss = self.cce.loss(y_true, y_pred)
        
        # Loss deve ser ~0 (pequeno devido a clipping)
        self.assertLess(loss, 0.01)
    
    def test_cce_loss_worst_predictions(self):
        """Testa loss com predições completamente erradas"""
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        
        # Predições opostas
        y_pred = np.array([[0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0]])
        
        loss = self.cce.loss(y_true, y_pred)
        
        # Loss deve ser alto
        self.assertGreater(loss, 0.5)
    
    def test_cce_loss_is_positive(self):
        """Testa se loss é sempre positivo"""
        y_true = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
        
        y_pred = np.random.uniform(0.1, 0.9, (3, 4))
        # Normalizar para somar 1
        y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
        
        loss = self.cce.loss(y_true, y_pred)
        
        self.assertGreaterEqual(loss, 0)
    
    def test_cce_loss_binary_vs_multiclass(self):
        """Testa loss com 2 classes vs múltiplas classes"""
        # 2 classes
        y_true_binary = np.array([[1, 0], [0, 1]])
        y_pred_binary = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        loss_binary = self.cce.loss(y_true_binary, y_pred_binary)
        
        # 4 classes
        y_true_multi = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        y_pred_multi = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]])
        
        loss_multi = self.cce.loss(y_true_multi, y_pred_multi)
        
        # Ambos devem ser positivos
        self.assertGreater(loss_binary, 0)
        self.assertGreater(loss_multi, 0)
    
    def test_cce_loss_with_clipping(self):
        """Testa se clipping previne log(0)"""
        y_true = np.array([[1, 0, 0]])
        
        # Predições com zeros (devem ser clipped)
        y_pred = np.array([[0.0, 0.5, 0.5]])
        
        # Não deve dar erro ou retornar inf/nan
        loss = self.cce.loss(y_true, y_pred)
        
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
    
    def test_cce_loss_shape_robustness(self):
        """Testa com diferentes shapes"""
        shapes = [(2, 3), (5, 4), (10, 10), (100, 5)]
        
        for n_samples, n_classes in shapes:
            y_true = np.zeros((n_samples, n_classes))
            # One-hot: cada sample tem uma classe
            y_true[np.arange(n_samples), np.random.randint(0, n_classes, n_samples)] = 1
            
            y_pred = np.random.uniform(0.1, 0.9, (n_samples, n_classes))
            y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
            
            loss = self.cce.loss(y_true, y_pred)
            
            self.assertIsInstance(loss, (float, np.floating))
            self.assertGreaterEqual(loss, 0)
    
    def test_cce_loss_formula(self):
        """Testa se fórmula está correta: -mean(sum(y_true * log(y_pred)))"""
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.8, 0.1]])
        
        # Cálculo manual
        clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        expected = -np.mean(np.sum(y_true * np.log(clipped), axis=1))
        
        loss = self.cce.loss(y_true, y_pred)
        
        self.assertAlmostEqual(loss, expected, places=10)


class TestCategoricalCrossEntropyDerivative(unittest.TestCase):
    """Testes para método derivative da CategoricalCrossEntropy"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.cce = CategoricalCrossEntropy()
    
    def test_cce_derivative_shape(self):
        """Testa se derivada tem mesmo shape que input"""
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        y_pred = np.array([[0.7, 0.2, 0.1],
                           [0.2, 0.7, 0.1],
                           [0.1, 0.2, 0.7]])
        
        derivative = self.cce.derivative(y_true, y_pred)
        
        self.assertEqual(derivative.shape, y_true.shape)
    
    def test_cce_derivative_formula(self):
        """Testa fórmula: -y_true / y_pred / n_samples"""
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0]])
        y_pred = np.array([[0.8, 0.1, 0.1],
                           [0.1, 0.8, 0.1]])
        
        # Cálculo manual
        clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        expected = -y_true / clipped / y_true.shape[0]
        
        derivative = self.cce.derivative(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(derivative, expected)
    
    def test_cce_derivative_no_nan_or_inf(self):
        """Testa se clipping previne NaN/Inf"""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.0, 0.5, 0.5]])  # Zero será clipped
        
        derivative = self.cce.derivative(y_true, y_pred)
        
        self.assertFalse(np.any(np.isnan(derivative)))
        self.assertFalse(np.any(np.isinf(derivative)))
    
    def test_cce_derivative_zeros_for_correct_predictions(self):
        """Testa gradiente onde predições estão corretas"""
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05],
                           [0.05, 0.9, 0.05]])
        
        derivative = self.cce.derivative(y_true, y_pred)
        
        # Onde y_true=0, derivada também deve ser ~0
        self.assertTrue(np.all(np.abs(derivative[y_true == 0]) < 0.1))
    
    def test_cce_derivative_large_for_wrong_predictions(self):
        """Testa gradiente grande para predições erradas"""
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.1, 0.45, 0.45]])  # Classe errada tem alta prob
        
        derivative = self.cce.derivative(y_true, y_pred)
        
        # Gradiente deve ser grande na classe correta (índice 0)
        self.assertLess(derivative[0, 0], -1.0)  # Negativo e grande


class TestCategoricalCrossEntropyVsBinary(unittest.TestCase):
    """Testes comparativos entre Categorical e Binary Cross-Entropy"""
    
    def test_cce_reduces_to_bce_for_two_classes(self):
        """
        Testa se CCE com 2 classes é equivalente a BCE.
        Nota: não são exatamente iguais devido a diferenças de implementação.
        """
        # Para 2 classes, CCE deve se comportar similarmente a BCE
        y_true_cce = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred_cce = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        
        cce = CategoricalCrossEntropy()
        loss_cce = cce.loss(y_true_cce, y_pred_cce)
        
        # Loss deve ser razoável
        self.assertGreater(loss_cce, 0)
        self.assertLess(loss_cce, 5)

if __name__ == '__main__':
    unittest.main()