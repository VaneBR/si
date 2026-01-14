import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from si.model_selection.randomized_search import randomized_search_cv
from si.data.dataset import Dataset
from si.base.model import Model


class MockModel(Model):
    """Modelo mock para testes"""
    
    def __init__(self, param1=1, param2=0.1, param3=100):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self._fitted = False
    
    def _fit(self, dataset):
        self._fitted = True
        return self
    
    def _predict(self, dataset):
        return np.random.randint(0, 2, size=len(dataset.X))
    
    def _score(self, dataset, predictions):
        return np.random.random()


class TestRandomizedSearchCVValidation(unittest.TestCase):
    """Testes para validação de inputs"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(50, 3)
        self.y = np.random.randint(0, 2, 50)
        self.dataset = Dataset(X=self.X, y=self.y)
        self.model = MockModel()
    
    def test_invalid_hyperparameter_raises_error(self):
        """Testa se levanta erro com hiperparâmetro inválido"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3]),
            'invalid_param': np.array([1, 2, 3])  # Não existe no modelo
        }
        
        with self.assertRaises(ValueError) as context:
            randomized_search_cv(
                model=self.model,
                dataset=self.dataset,
                hyperparameter_distributions=hyperparameter_distributions,
                n_iter=2,
                cv=2
            )
        
        self.assertIn("não tem", str(context.exception).lower())
        self.assertIn("invalid_param", str(context.exception))
    
    def test_all_valid_hyperparameters_pass(self):
        """Testa se hiperparâmetros válidos passam a validação"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3]),
            'param2': np.array([0.1, 0.2, 0.3])
        }
        
        # Não deve levantar erro
        try:
            result = randomized_search_cv(
                model=self.model,
                dataset=self.dataset,
                hyperparameter_distributions=hyperparameter_distributions,
                n_iter=2,
                cv=2
            )
            self.assertIsNotNone(result)
        except ValueError:
            self.fail("Não deveria levantar erro com hiperparâmetros válidos")
    
    def test_empty_hyperparameter_distributions(self):
        """Testa com dicionário vazio de hiperparâmetros"""
        hyperparameter_distributions = {}
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=1,
            cv=2
        )
        
        # Deve funcionar mas sem testar nada
        self.assertIsInstance(result, dict)


class TestRandomizedSearchCVOutput(unittest.TestCase):
    """Testes para estrutura do output"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(30, 3)
        self.y = np.random.randint(0, 2, 30)
        self.dataset = Dataset(X=self.X, y=self.y)
        self.model = MockModel()
    
    def test_output_is_dict(self):
        """Testa se output é um dicionário"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=2,
            cv=2
        )
        
        self.assertIsInstance(result, dict)
    
    def test_output_has_required_keys(self):
        """Testa se output tem todas as chaves necessárias"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=2,
            cv=2
        )
        
        required_keys = ['hyperparameters', 'scores', 'best_hyperparameters', 'best_score']
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_hyperparameters_is_list(self):
        """Testa se hyperparameters é uma lista"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=2,
            cv=2
        )
        
        self.assertIsInstance(result['hyperparameters'], list)
    
    def test_scores_is_list(self):
        """Testa se scores é uma lista"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=2,
            cv=2
        )
        
        self.assertIsInstance(result['scores'], list)
    
    def test_best_hyperparameters_is_dict(self):
        """Testa se best_hyperparameters é um dicionário"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=2,
            cv=2
        )
        
        self.assertIsInstance(result['best_hyperparameters'], dict)
    
    def test_best_score_is_numeric(self):
        """Testa se best_score é numérico"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=2,
            cv=2
        )
        
        self.assertIsInstance(result['best_score'], (int, float, np.number))


class TestRandomizedSearchCVIterations(unittest.TestCase):
    """Testes para número de iterações"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(30, 3)
        self.y = np.random.randint(0, 2, 30)
        self.dataset = Dataset(X=self.X, y=self.y)
        self.model = MockModel()
    
    def test_correct_number_of_iterations(self):
        """Testa se executa o número correto de iterações"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3, 4, 5])
        }
        n_iter = 3
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=n_iter,
            cv=2
        )
        
        self.assertEqual(len(result['scores']), n_iter)
        self.assertEqual(len(result['hyperparameters']), n_iter)
    
    def test_n_iter_larger_than_combinations(self):
        """Testa quando n_iter é maior que combinações possíveis"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2])  # Apenas 2 valores
        }
        n_iter = 5  # Pede mais iterações do que possível
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=n_iter,
            cv=2
        )
        
        # Deve ter no máximo 2 combinações únicas
        self.assertLessEqual(len(result['scores']), n_iter)
    
    def test_single_iteration(self):
        """Testa com apenas uma iteração"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=1,
            cv=2
        )
        
        self.assertEqual(len(result['scores']), 1)
        self.assertEqual(len(result['hyperparameters']), 1)


class TestRandomizedSearchCVRandomness(unittest.TestCase):
    """Testes para comportamento aleatório e seed"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(30, 3)
        self.y = np.random.randint(0, 2, 30)
        self.dataset = Dataset(X=self.X, y=self.y)
        self.model = MockModel()
    
    def test_seed_reproducibility(self):
        """Testa se seed garante reprodutibilidade"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3, 4, 5]),
            'param2': np.array([0.1, 0.2, 0.3])
        }
        
        # Primeira execução
        result1 = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=3,
            cv=2,
            seed=42
        )
        
        # Segunda execução com mesma seed
        result2 = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=3,
            cv=2,
            seed=42
        )
        
        # Hiperparâmetros testados devem ser os mesmos
        self.assertEqual(result1['hyperparameters'], result2['hyperparameters'])
    
    def test_different_seeds_different_results(self):
        """Testa se seeds diferentes geram resultados diferentes"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'param2': np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        }
        
        # Primeira execução
        result1 = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=5,
            cv=2,
            seed=42
        )
        
        # Segunda execução com seed diferente
        result2 = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=5,
            cv=2,
            seed=123
        )
        
        # Devem ser diferentes (com alta probabilidade)
        self.assertNotEqual(result1['hyperparameters'], result2['hyperparameters'])
    
    def test_no_duplicate_combinations(self):
        """Testa se não há combinações duplicadas"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3, 4, 5])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=5,
            cv=2,
            seed=42
        )
        
        # Converter para tuples para verificar unicidade
        combinations = [tuple(sorted(params.items())) 
                       for params in result['hyperparameters']]
        
        self.assertEqual(len(combinations), len(set(combinations)))


class TestRandomizedSearchCVBestSelection(unittest.TestCase):
    """Testes para seleção do melhor resultado"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(30, 3)
        self.y = np.random.randint(0, 2, 30)
        self.dataset = Dataset(X=self.X, y=self.y)
    
    @patch('si.model_selection.randomized_search.k_fold_cross_validation')
    def test_best_score_is_maximum(self, mock_cv):
        """Testa se best_score é o máximo dos scores"""
        # Configurar scores fixos
        mock_cv.side_effect = [
            [0.7, 0.75],  # média 0.725
            [0.8, 0.85],  # média 0.825 (melhor)
            [0.6, 0.65]   # média 0.625
        ]
        
        model = MockModel()
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=3,
            cv=2,
            seed=42
        )
        
        # O melhor score deve ser o máximo
        self.assertEqual(result['best_score'], max(result['scores']))
    
    @patch('si.model_selection.randomized_search.k_fold_cross_validation')
    def test_best_hyperparameters_correspond_to_best_score(self, mock_cv):
        """Testa se best_hyperparameters correspondem ao best_score"""
        # Configurar scores fixos
        mock_cv.side_effect = [
            [0.7, 0.75],
            [0.9, 0.95],  # Este é o melhor
            [0.6, 0.65]
        ]
        
        model = MockModel()
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=3,
            cv=2,
            seed=42
        )
        
        # Encontrar índice do melhor score
        best_idx = result['scores'].index(result['best_score'])
        
        # Best hyperparameters devem corresponder
        self.assertEqual(
            result['best_hyperparameters'],
            result['hyperparameters'][best_idx]
        )


class TestRandomizedSearchCVMultipleHyperparameters(unittest.TestCase):
    """Testes com múltiplos hiperparâmetros"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(30, 3)
        self.y = np.random.randint(0, 2, 30)
        self.dataset = Dataset(X=self.X, y=self.y)
        self.model = MockModel()
    
    def test_single_hyperparameter(self):
        """Testa com um único hiperparâmetro"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3, 4, 5])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=3,
            cv=2
        )
        
        # Cada combinação deve ter apenas param1
        for params in result['hyperparameters']:
            self.assertEqual(len(params), 1)
            self.assertIn('param1', params)
    
    def test_two_hyperparameters(self):
        """Testa com dois hiperparâmetros"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3]),
            'param2': np.array([0.1, 0.2, 0.3])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=3,
            cv=2
        )
        
        # Cada combinação deve ter param1 e param2
        for params in result['hyperparameters']:
            self.assertEqual(len(params), 2)
            self.assertIn('param1', params)
            self.assertIn('param2', params)
    
    def test_three_hyperparameters(self):
        """Testa com três hiperparâmetros"""
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3]),
            'param2': np.array([0.1, 0.2, 0.3]),
            'param3': np.array([100, 200, 300])
        }
        
        result = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=5,
            cv=2
        )
        
        # Cada combinação deve ter todos os 3 parâmetros
        for params in result['hyperparameters']:
            self.assertEqual(len(params), 3)
            self.assertIn('param1', params)
            self.assertIn('param2', params)
            self.assertIn('param3', params)


class TestRandomizedSearchCVCrossValidation(unittest.TestCase):
    """Testes relacionados com cross-validation"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(50, 3)
        self.y = np.random.randint(0, 2, 50)
        self.dataset = Dataset(X=self.X, y=self.y)
        self.model = MockModel()
    
    @patch('si.model_selection.randomized_search.k_fold_cross_validation')
    def test_cv_called_for_each_iteration(self, mock_cv):
        """Testa se k_fold_cross_validation é chamado para cada iteração"""
        mock_cv.return_value = [0.8, 0.85]
        
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        n_iter = 3
        
        randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=n_iter,
            cv=2
        )
        
        # Deve ser chamado n_iter vezes
        self.assertEqual(mock_cv.call_count, n_iter)
    
    @patch('si.model_selection.randomized_search.k_fold_cross_validation')
    def test_cv_with_custom_scoring(self, mock_cv):
        """Testa se scoring customizado é passado para CV"""
        mock_cv.return_value = [0.8, 0.85]
        
        def custom_scoring(y_true, y_pred):
            return 0.9
        
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            scoring=custom_scoring,
            n_iter=2,
            cv=2
        )
        
        # Verificar que scoring foi passado
        for call in mock_cv.call_args_list:
            self.assertEqual(call[1]['scoring'], custom_scoring)


class TestRandomizedSearchCVErrorHandling(unittest.TestCase):
    """Testes para tratamento de erros"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.random.randn(30, 3)
        self.y = np.random.randint(0, 2, 30)
        self.dataset = Dataset(X=self.X, y=self.y)
    
    @patch('si.model_selection.randomized_search.k_fold_cross_validation')
    def test_handles_cv_errors(self, mock_cv):
        """Testa se trata erros durante cross-validation"""
        # Primeira chamada funciona, segunda falha, terceira funciona
        mock_cv.side_effect = [
            [0.8, 0.85],
            Exception("CV failed"),
            [0.7, 0.75]
        ]
        
        model = MockModel()
        hyperparameter_distributions = {
            'param1': np.array([1, 2, 3])
        }
        
        result = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=3,
            cv=2
        )
        
        # Deve ter 3 resultados (erro retorna -inf)
        self.assertEqual(len(result['scores']), 3)
        
        # Um dos scores deve ser -inf
        self.assertIn(-np.inf, result['scores'])


class TestRandomizedSearchCVProtocol(unittest.TestCase):
    """Teste do protocolo completo do exercício"""
    
    def setUp(self):
        """Setup simulando o protocolo do exercício"""
        # Simular breast-bin.csv
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 2, 100)
        self.dataset = Dataset(X=self.X, y=self.y)
    
    def test_protocol_structure(self):
        """
        Testa o protocolo do exercício 11.2:
        - l2_penalty: 1 a 10 (10 valores)
        - alpha: 0.001 a 0.0001 (100 valores)
        - max_iter: 1000 a 2000 (200 valores)
        - n_iter=10, cv=3
        """
        model = MockModel()
        
        # Adicionar atributos esperados
        model.l2_penalty = 1.0
        model.alpha = 0.001
        model.max_iter = 1000
        
        hyperparameter_distributions = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200).astype(int)
        }
        
        result = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_distributions=hyperparameter_distributions,
            n_iter=10,
            cv=3,
            seed=42
        )
        
        # Verificações
        self.assertEqual(len(result['scores']), 10)
        self.assertEqual(len(result['hyperparameters']), 10)
        self.assertIn('l2_penalty', result['best_hyperparameters'])
        self.assertIn('alpha', result['best_hyperparameters'])
        self.assertIn('max_iter', result['best_hyperparameters'])

if __name__ == '__main__':
    unittest.main()