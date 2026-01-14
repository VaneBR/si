import unittest
import numpy as np
from si.neural_networks.layers import Dropout, Layer


class TestDropoutInitialization(unittest.TestCase):
    """Testes para inicialização da camada Dropout"""
    
    def test_dropout_inherits_from_layer(self):
        """Testa se Dropout herda de Layer"""
        dropout = Dropout(probability=0.5)
        self.assertIsInstance(dropout, Layer)
    
    def test_dropout_init_valid_probability(self):
        """Testa inicialização com probabilidade válida"""
        dropout = Dropout(probability=0.5)
        self.assertEqual(dropout.probability, 0.5)
        self.assertIsNone(dropout.mask)
        self.assertIsNone(dropout.input)
        self.assertIsNone(dropout.output)
    
    def test_dropout_init_probability_zero(self):
        """Testa inicialização com probability=0 (sem dropout)"""
        dropout = Dropout(probability=0.0)
        self.assertEqual(dropout.probability, 0.0)
    
    def test_dropout_init_probability_high(self):
        """Testa inicialização com probabilidade alta"""
        dropout = Dropout(probability=0.9)
        self.assertEqual(dropout.probability, 0.9)
    
    def test_dropout_init_invalid_probability_negative(self):
        """Testa se levanta erro com probabilidade negativa"""
        with self.assertRaises(ValueError):
            Dropout(probability=-0.1)
    
    def test_dropout_init_invalid_probability_greater_than_one(self):
        """Testa se levanta erro com probabilidade > 1"""
        with self.assertRaises(ValueError):
            Dropout(probability=1.5)
    
    def test_dropout_init_probability_exactly_one(self):
        """Testa se levanta erro com probabilidade = 1"""
        with self.assertRaises(ValueError):
            Dropout(probability=1.0)


class TestDropoutForwardPropagationTraining(unittest.TestCase):
    """Testes para forward propagation em modo training"""
    
    def setUp(self):
        """Setup para cada teste"""
        np.random.seed(42)
        self.dropout = Dropout(probability=0.5)
        self.input = np.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]])
    
    def test_forward_training_returns_array(self):
        """Testa se forward retorna um array"""
        output = self.dropout.forward_propagation(self.input, training=True)
        self.assertIsInstance(output, np.ndarray)
    
    def test_forward_training_same_shape(self):
        """Testa se output tem o mesmo shape que input"""
        output = self.dropout.forward_propagation(self.input, training=True)
        self.assertEqual(output.shape, self.input.shape)
    
    def test_forward_training_creates_mask(self):
        """Testa se cria máscara durante training"""
        self.dropout.forward_propagation(self.input, training=True)
        self.assertIsNotNone(self.dropout.mask)
        self.assertEqual(self.dropout.mask.shape, self.input.shape)
    
    def test_forward_training_mask_is_binary(self):
        """Testa se máscara contém apenas 0s e 1s"""
        self.dropout.forward_propagation(self.input, training=True)
        unique_values = np.unique(self.dropout.mask)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))
    
    def test_forward_training_stores_input(self):
        """Testa se armazena o input"""
        self.dropout.forward_propagation(self.input, training=True)
        np.testing.assert_array_equal(self.dropout.input, self.input)
    
    def test_forward_training_stores_output(self):
        """Testa se armazena o output"""
        output = self.dropout.forward_propagation(self.input, training=True)
        np.testing.assert_array_equal(self.dropout.output, output)
    
    def test_forward_training_zeros_some_values(self):
        """Testa se alguns valores são zerados"""
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        output = dropout.forward_propagation(self.input, training=True)
        
        # Com probability=0.5, cerca de 50% devem ser zero
        zeros_count = np.sum(output == 0)
        total_count = output.size
        
        # Permitir alguma variação devido à aleatoriedade
        self.assertGreater(zeros_count, 0)
        self.assertLess(zeros_count, total_count)
    
    def test_forward_training_scaling_factor(self):
        """Testa se aplica o scaling factor correto"""
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        output = dropout.forward_propagation(self.input, training=True)
        
        # Valores não-zero devem ser escalados por 1/(1-probability) = 2
        non_zero_output = output[output != 0]
        non_zero_input = self.input[dropout.mask == 1]
        
        expected_scaling = 1.0 / (1.0 - 0.5)
        np.testing.assert_array_almost_equal(
            non_zero_output, 
            non_zero_input * expected_scaling
        )
    
    def test_forward_training_high_probability(self):
        """Testa com probabilidade alta (mais dropout)"""
        np.random.seed(42)
        dropout = Dropout(probability=0.8)
        output = dropout.forward_propagation(self.input, training=True)
        
        # Com probability=0.8, cerca de 80% devem ser zero
        zeros_ratio = np.sum(output == 0) / output.size
        self.assertGreater(zeros_ratio, 0.5)  # Pelo menos 50% zeros
    
    def test_forward_training_low_probability(self):
        """Testa com probabilidade baixa (menos dropout)"""
        np.random.seed(42)
        dropout = Dropout(probability=0.2)
        output = dropout.forward_propagation(self.input, training=True)
        
        # Com probability=0.2, cerca de 20% devem ser zero
        zeros_ratio = np.sum(output == 0) / output.size
        self.assertLess(zeros_ratio, 0.5)  # Menos de 50% zeros
    
    def test_forward_training_probability_zero(self):
        """Testa com probability=0 (sem dropout)"""
        dropout = Dropout(probability=0.0)
        output = dropout.forward_propagation(self.input, training=True)
        
        # Não deve ter zeros (todos mantidos)
        self.assertEqual(np.sum(output == 0), 0)
        np.testing.assert_array_equal(output, self.input)


class TestDropoutForwardPropagationInference(unittest.TestCase):
    """Testes para forward propagation em modo inference"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.dropout = Dropout(probability=0.5)
        self.input = np.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]])
    
    def test_forward_inference_returns_input_unchanged(self):
        """Testa se retorna input sem modificações"""
        output = self.dropout.forward_propagation(self.input, training=False)
        np.testing.assert_array_equal(output, self.input)
    
    def test_forward_inference_no_mask_created(self):
        """Testa se não cria máscara em modo inference"""
        self.dropout.forward_propagation(self.input, training=False)
        self.assertIsNone(self.dropout.mask)
    
    def test_forward_inference_no_scaling(self):
        """Testa se não aplica scaling em inference"""
        output = self.dropout.forward_propagation(self.input, training=False)
        
        # Output deve ser exatamente igual ao input
        np.testing.assert_array_equal(output, self.input)
        
        # Nenhum valor deve ser zero (a menos que já fosse no input)
        if not np.any(self.input == 0):
            self.assertEqual(np.sum(output == 0), 0)
    
    def test_forward_inference_stores_input(self):
        """Testa se armazena input mesmo em inference"""
        self.dropout.forward_propagation(self.input, training=False)
        np.testing.assert_array_equal(self.dropout.input, self.input)
    
    def test_forward_inference_stores_output(self):
        """Testa se armazena output em inference"""
        output = self.dropout.forward_propagation(self.input, training=False)
        np.testing.assert_array_equal(self.dropout.output, output)
    
    def test_forward_inference_different_shapes(self):
        """Testa inference com diferentes shapes"""
        shapes = [(10,), (5, 3), (2, 3, 4), (1, 1, 1, 10)]
        
        for shape in shapes:
            input_data = np.random.randn(*shape)
            dropout = Dropout(probability=0.5)
            output = dropout.forward_propagation(input_data, training=False)
            
            np.testing.assert_array_equal(output, input_data)
            self.assertEqual(output.shape, shape)


class TestDropoutBackwardPropagation(unittest.TestCase):
    """Testes para backward propagation"""
    
    def setUp(self):
        """Setup para cada teste"""
        np.random.seed(42)
        self.dropout = Dropout(probability=0.5)
        self.input = np.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]])
    
    def test_backward_with_mask(self):
        """Testa backward propagation após forward training"""
        # Forward para criar máscara
        self.dropout.forward_propagation(self.input, training=True)
        
        # Erro de saída
        output_error = np.ones_like(self.input)
        
        # Backward
        input_error = self.dropout.backward_propagation(output_error)
        
        # Verificar shape
        self.assertEqual(input_error.shape, output_error.shape)
    
    def test_backward_applies_mask(self):
        """Testa se aplica a máscara corretamente"""
        # Forward para criar máscara
        self.dropout.forward_propagation(self.input, training=True)
        mask = self.dropout.mask.copy()
        
        # Erro de saída
        output_error = np.ones_like(self.input)
        
        # Backward
        input_error = self.dropout.backward_propagation(output_error)
        
        # Erro deve ser zero onde máscara é zero
        np.testing.assert_array_equal(
            input_error[mask == 0],
            np.zeros(np.sum(mask == 0))
        )
    
    def test_backward_preserves_non_masked_gradients(self):
        """Testa se preserva gradientes não mascarados"""
        # Forward para criar máscara
        self.dropout.forward_propagation(self.input, training=True)
        mask = self.dropout.mask.copy()
        
        # Erro de saída customizado
        output_error = np.array([[1.0, 2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0, 8.0]])
        
        # Backward
        input_error = self.dropout.backward_propagation(output_error)
        
        # Onde máscara é 1, erro deve ser preservado
        np.testing.assert_array_equal(
            input_error[mask == 1],
            output_error[mask == 1]
        )
    
    def test_backward_without_mask(self):
        """Testa backward em modo inference (sem máscara)"""
        # Forward em modo inference (sem criar máscara)
        self.dropout.forward_propagation(self.input, training=False)
        
        # Erro de saída
        output_error = np.ones_like(self.input)
        
        # Backward deve passar erro sem modificação
        input_error = self.dropout.backward_propagation(output_error)
        
        np.testing.assert_array_equal(input_error, output_error)
    
    def test_backward_gradient_flow(self):
        """Testa fluxo completo de gradientes"""
        np.random.seed(42)
        dropout = Dropout(probability=0.3)
        
        # Forward
        output = dropout.forward_propagation(self.input, training=True)
        
        # Backward com gradientes
        gradient = np.random.randn(*self.input.shape)
        input_gradient = dropout.backward_propagation(gradient)
        
        # Gradiente deve ter mesmo shape
        self.assertEqual(input_gradient.shape, gradient.shape)
        
        # Onde output foi zerado, gradiente também deve ser zero
        zeros_in_output = (output == 0)
        zeros_in_gradient = (input_gradient == 0)
        np.testing.assert_array_equal(zeros_in_output, zeros_in_gradient)


class TestDropoutOutputShape(unittest.TestCase):
    """Testes para método output_shape"""
    
    def test_output_shape_returns_input_shape(self):
        """Testa se retorna input_shape"""
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((10, 20))
        
        self.assertEqual(dropout.output_shape(), (10, 20))
    
    def test_output_shape_1d(self):
        """Testa com shape 1D"""
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((100,))
        
        self.assertEqual(dropout.output_shape(), (100,))
    
    def test_output_shape_3d(self):
        """Testa com shape 3D"""
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape((32, 32, 3))
        
        self.assertEqual(dropout.output_shape(), (32, 32, 3))
    
    def test_output_shape_after_forward(self):
        """Testa output_shape após forward propagation"""
        dropout = Dropout(probability=0.5)
        input_data = np.random.randn(10, 5)
        dropout.set_input_shape(input_data.shape)
        
        dropout.forward_propagation(input_data, training=True)
        
        self.assertEqual(dropout.output_shape(), input_data.shape)


class TestDropoutParameters(unittest.TestCase):
    """Testes para método parameters"""
    
    def test_parameters_returns_zero(self):
        """Testa se retorna 0 (sem parâmetros treináveis)"""
        dropout = Dropout(probability=0.5)
        self.assertEqual(dropout.parameters(), 0)
    
    def test_parameters_different_probabilities(self):
        """Testa se sempre retorna 0 independente da probability"""
        for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dropout = Dropout(probability=prob)
            self.assertEqual(dropout.parameters(), 0)


class TestDropoutLayerName(unittest.TestCase):
    """Testes para método layer_name"""
    
    def test_layer_name_is_dropout(self):
        """Testa se nome da layer é 'Dropout'"""
        dropout = Dropout(probability=0.5)
        self.assertEqual(dropout.layer_name(), "Dropout")


class TestDropoutStatisticalProperties(unittest.TestCase):
    """Testes estatísticos do comportamento do Dropout"""
    
    def test_dropout_rate_statistical(self):
        """Testa se taxa de dropout é aproximadamente correta"""
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        
        # Input grande para estatísticas confiáveis
        input_data = np.ones((1000, 1000))
        
        # Múltiplas execuções
        dropout_rates = []
        for _ in range(10):
            output = dropout.forward_propagation(input_data, training=True)
            dropout_rate = np.sum(output == 0) / output.size
            dropout_rates.append(dropout_rate)
        
        # Média deve estar próxima de 0.5
        mean_rate = np.mean(dropout_rates)
        self.assertAlmostEqual(mean_rate, 0.5, delta=0.05)
    
    def test_expected_value_preserved(self):
        """Testa se valor esperado é preservado com scaling"""
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        
        # Input constante
        input_data = np.ones((10000,)) * 10.0
        
        # Múltiplas execuções
        outputs = []
        for _ in range(100):
            output = dropout.forward_propagation(input_data, training=True)
            outputs.append(np.mean(output))
        
        # Média das médias deve estar próxima do valor original
        mean_output = np.mean(outputs)
        self.assertAlmostEqual(mean_output, 10.0, delta=0.5)
    
    def test_inference_equals_training_expectation(self):
        """
        Testa se output em inference é aproximadamente igual
        ao valor esperado em training (graças ao scaling)
        """
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        
        # Input grande
        input_data = np.random.randn(10000, 100)
        
        # Output em inference
        output_inference = dropout.forward_propagation(input_data, training=False)
        mean_inference = np.mean(output_inference)
        
        # Média de múltiplos outputs em training
        training_means = []
        for _ in range(50):
            output_training = dropout.forward_propagation(input_data, training=True)
            training_means.append(np.mean(output_training))
        
        mean_training = np.mean(training_means)
        
        # Devem ser aproximadamente iguais
        self.assertAlmostEqual(mean_inference, mean_training, delta=0.1)


class TestDropoutEdgeCases(unittest.TestCase):
    """Testes para casos extremos"""
    
    def test_single_element_input(self):
        """Testa com input de um único elemento"""
        dropout = Dropout(probability=0.5)
        input_data = np.array([5.0])
        
        output = dropout.forward_propagation(input_data, training=True)
        
        self.assertEqual(output.shape, input_data.shape)
    
    def test_zero_input(self):
        """Testa com input contendo zeros"""
        dropout = Dropout(probability=0.5)
        input_data = np.array([[0.0, 1.0, 0.0, 2.0]])
        
        output = dropout.forward_propagation(input_data, training=True)
        
        self.assertEqual(output.shape, input_data.shape)
    
    def test_negative_input(self):
        """Testa com valores negativos"""
        dropout = Dropout(probability=0.5)
        input_data = np.array([[-1.0, -2.0, -3.0, -4.0]])
        
        output = dropout.forward_propagation(input_data, training=True)
        
        # Valores negativos mascarados devem permanecer negativos (escalados)
        non_zero_output = output[output != 0]
        if len(non_zero_output) > 0:
            self.assertTrue(np.any(non_zero_output < 0))
    
    def test_large_input(self):
        """Testa com input muito grande"""
        dropout = Dropout(probability=0.5)
        input_data = np.random.randn(1000, 1000)
        
        output = dropout.forward_propagation(input_data, training=True)
        
        self.assertEqual(output.shape, input_data.shape)


class TestDropoutIntegration(unittest.TestCase):
    """Testes de integração simulando uso em rede neural"""
    
    def test_multiple_forward_backward_cycles(self):
        """Testa múltiplos ciclos de forward/backward"""
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        input_data = np.random.randn(32, 128)
        
        for epoch in range(5):
            # Forward
            output = dropout.forward_propagation(input_data, training=True)
            
            # Backward
            gradient = np.random.randn(*output.shape)
            input_gradient = dropout.backward_propagation(gradient)
            
            # Verificações
            self.assertEqual(output.shape, input_data.shape)
            self.assertEqual(input_gradient.shape, gradient.shape)
    
    def test_training_then_inference(self):
        """Testa transição de training para inference"""
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        input_data = np.random.randn(10, 20)
        
        # Training
        output_train = dropout.forward_propagation(input_data, training=True)
        self.assertIsNotNone(dropout.mask)
        
        # Inference
        output_inference = dropout.forward_propagation(input_data, training=False)
        self.assertIsNone(dropout.mask)
        
        # Outputs devem ser diferentes
        self.assertFalse(np.array_equal(output_train, output_inference))
    
    def test_consistent_mask_during_backward(self):
        """Testa se máscara permanece consistente durante backward"""
        np.random.seed(42)
        dropout = Dropout(probability=0.5)
        input_data = np.random.randn(5, 10)
        
        # Forward
        output = dropout.forward_propagation(input_data, training=True)
        mask_after_forward = dropout.mask.copy()
        
        # Backward
        gradient = np.ones_like(output)
        input_gradient = dropout.backward_propagation(gradient)
        
        # Máscara não deve ter mudado
        np.testing.assert_array_equal(dropout.mask, mask_after_forward)
        
        # Gradiente zero onde máscara é zero
        self.assertTrue(np.all(input_gradient[mask_after_forward == 0] == 0))

if __name__ == '__main__':
    unittest.main()