import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from si.ensemble.stacking_classifier import StackingClassifier
from si.data.dataset import Dataset
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression


class TestStackingClassifierInit(unittest.TestCase):
    """Testes para inicialização do StackingClassifier"""
    
    def test_init_basic(self):
        """Testa inicialização básica"""
        model1 = Mock()
        model2 = Mock()
        final_model = Mock()
        
        stacking = StackingClassifier(
            models=[model1, model2],
            final_model=final_model
        )
        
        self.assertEqual(len(stacking.models), 2)
        self.assertEqual(stacking.final_model, final_model)
    
    def test_init_empty_models(self):
        """Testa inicialização com lista vazia de modelos"""
        final_model = Mock()
        
        stacking = StackingClassifier(
            models=[],
            final_model=final_model
        )
        
        self.assertEqual(len(stacking.models), 0)
    
    def test_init_with_kwargs(self):
        """Testa inicialização com kwargs adicionais"""
        model1 = Mock()
        final_model = Mock()
        
        stacking = StackingClassifier(
            models=[model1],
            final_model=final_model,
            random_state=42
        )
        
        self.assertIsNotNone(stacking)


class TestPredictionsToFeatures(unittest.TestCase):
    """Testes para conversão de predições em features"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.stacking = StackingClassifier(models=[], final_model=Mock())
    
    def test_numeric_predictions(self):
        """Testa conversão de predições numéricas"""
        predictions_list = [
            np.array([0, 1, 0, 1]),
            np.array([1, 1, 0, 0]),
            np.array([0, 0, 1, 1])
        ]
        
        X_meta = self.stacking._predictions_to_features(predictions_list)
        
        self.assertEqual(X_meta.shape, (4, 3))
        self.assertTrue(np.all(X_meta >= 0))
        self.assertTrue(np.all(X_meta <= 1))
    
    def test_categorical_predictions(self):
        """Testa conversão de predições categóricas"""
        predictions_list = [
            np.array(['A', 'B', 'A', 'B']),
            np.array(['X', 'Y', 'X', 'Y'])
        ]
        
        X_meta = self.stacking._predictions_to_features(predictions_list)
        
        self.assertEqual(X_meta.shape, (4, 2))
        # Verifica que são números
        self.assertTrue(np.issubdtype(X_meta.dtype, np.number))
    
    def test_mixed_predictions(self):
        """Testa conversão de predições mistas (numéricas e categóricas)"""
        predictions_list = [
            np.array([0, 1, 0, 1]),
            np.array(['A', 'B', 'A', 'B'])
        ]
        
        X_meta = self.stacking._predictions_to_features(predictions_list)
        
        self.assertEqual(X_meta.shape, (4, 2))
        self.assertTrue(np.issubdtype(X_meta.dtype, np.number))
    
    def test_2d_predictions_flatten(self):
        """Testa flatten de predições 2D"""
        predictions_list = [
            np.array([[0], [1], [0], [1]]),
            np.array([[1], [1], [0], [0]])
        ]
        
        X_meta = self.stacking._predictions_to_features(predictions_list)
        
        self.assertEqual(X_meta.shape, (4, 2))
    
    def test_single_model_predictions(self):
        """Testa conversão com um único modelo"""
        predictions_list = [
            np.array([0, 1, 0, 1, 0])
        ]
        
        X_meta = self.stacking._predictions_to_features(predictions_list)
        
        self.assertEqual(X_meta.shape, (5, 1))


class TestStackingClassifierFit(unittest.TestCase):
    """Testes para o método _fit"""
    
    def setUp(self):
        """Setup para cada teste"""
        # Criar dataset sintético
        self.X = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6]
        ])
        self.y = np.array([0, 0, 1, 1, 1])
        self.dataset = Dataset(X=self.X, y=self.y, features=['f1', 'f2'], label='target')
    
    def test_fit_no_labels(self):
        """Testa fit com dataset sem labels"""
        dataset_no_y = Dataset(X=self.X, y=None)
        
        model1 = Mock()
        final_model = Mock()
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        with self.assertRaises(ValueError) as context:
            stacking.fit(dataset_no_y)
        
        self.assertIn("labels", str(context.exception).lower())
    
    def test_fit_trains_base_models(self):
        """Testa se todos os modelos base são treinados"""
        model1 = Mock()
        model2 = Mock()
        model3 = Mock()
        final_model = Mock()
        
        # Configurar predições mock
        model1.predict.return_value = np.array([0, 0, 1, 1, 1])
        model2.predict.return_value = np.array([0, 1, 1, 1, 0])
        model3.predict.return_value = np.array([1, 0, 1, 0, 1])
        
        stacking = StackingClassifier(
            models=[model1, model2, model3],
            final_model=final_model
        )
        
        stacking.fit(self.dataset)
        
        # Verifica se fit foi chamado para cada modelo base
        model1.fit.assert_called_once()
        model2.fit.assert_called_once()
        model3.fit.assert_called_once()
    
    def test_fit_trains_final_model(self):
        """Testa se o modelo final é treinado"""
        model1 = Mock()
        model1.predict.return_value = np.array([0, 0, 1, 1, 1])
        
        final_model = Mock()
        
        stacking = StackingClassifier(
            models=[model1],
            final_model=final_model
        )
        
        stacking.fit(self.dataset)
        
        # Verifica se final_model.fit foi chamado
        final_model.fit.assert_called_once()
    
    def test_fit_returns_self(self):
        """Testa se fit retorna self"""
        model1 = Mock()
        model1.predict.return_value = np.array([0, 0, 1, 1, 1])
        final_model = Mock()
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        result = stacking.fit(self.dataset)
        
        self.assertEqual(result, stacking)
    
    def test_fit_meta_dataset_shape(self):
        """Testa se o meta dataset tem o shape correto"""
        model1 = Mock()
        model2 = Mock()
        model1.predict.return_value = np.array([0, 0, 1, 1, 1])
        model2.predict.return_value = np.array([0, 1, 1, 1, 0])
        
        final_model = Mock()
        
        stacking = StackingClassifier(
            models=[model1, model2],
            final_model=final_model
        )
        
        stacking.fit(self.dataset)
        
        # Obter o dataset passado para final_model.fit
        call_args = final_model.fit.call_args
        meta_dataset = call_args[0][0]
        
        # Verificar shape: (n_samples, n_models)
        self.assertEqual(meta_dataset.X.shape, (5, 2))
        self.assertEqual(len(meta_dataset.features), 2)


class TestStackingClassifierPredict(unittest.TestCase):
    """Testes para o método _predict"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.array([[1, 2], [2, 3], [3, 4]])
        self.y = np.array([0, 1, 1])
        self.dataset = Dataset(X=self.X, y=self.y)
    
    def test_predict_without_fit(self):
        """Testa predição sem treinar o modelo"""
        model1 = Mock()
        final_model = None  # Modelo não treinado
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        with self.assertRaises(ValueError):
            stacking.predict(self.dataset)
    
    def test_predict_calls_base_models(self):
        """Testa se predict chama todos os modelos base"""
        model1 = Mock()
        model2 = Mock()
        model1.predict.return_value = np.array([0, 1, 1])
        model2.predict.return_value = np.array([1, 1, 0])
        
        final_model = Mock()
        final_model.predict.return_value = np.array([0, 1, 1])
        
        stacking = StackingClassifier(
            models=[model1, model2],
            final_model=final_model
        )
        
        stacking.predict(self.dataset)
        
        model1.predict.assert_called_once()
        model2.predict.assert_called_once()
    
    def test_predict_calls_final_model(self):
        """Testa se predict chama o modelo final"""
        model1 = Mock()
        model1.predict.return_value = np.array([0, 1, 1])
        
        final_model = Mock()
        final_model.predict.return_value = np.array([0, 1, 1])
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        stacking.predict(self.dataset)
        
        final_model.predict.assert_called_once()
    
    def test_predict_returns_correct_shape(self):
        """Testa se predict retorna o shape correto"""
        model1 = Mock()
        model1.predict.return_value = np.array([0, 1, 1])
        
        final_model = Mock()
        final_model.predict.return_value = np.array([0, 1, 1])
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        predictions = stacking.predict(self.dataset)
        
        self.assertEqual(predictions.shape[0], 3)
    
    def test_predict_with_no_y(self):
        """Testa predict com dataset sem y (casos de produção)"""
        dataset_no_y = Dataset(X=self.X, y=None)
        
        model1 = Mock()
        model1.predict.return_value = np.array([0, 1, 1])
        
        final_model = Mock()
        final_model.predict.return_value = np.array([0, 1, 1])
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        # Não deve dar erro mesmo sem y
        predictions = stacking.predict(dataset_no_y)
        
        self.assertIsNotNone(predictions)


class TestStackingClassifierScore(unittest.TestCase):
    """Testes para o método _score"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 0, 1, 1])
        self.dataset = Dataset(X=self.X, y=self.y)
    
    def test_score_perfect_predictions(self):
        """Testa score com predições perfeitas"""
        predictions = np.array([0, 0, 1, 1])
        
        stacking = StackingClassifier(models=[], final_model=Mock())
        score = stacking._score(self.dataset, predictions)
        
        self.assertEqual(score, 1.0)
    
    def test_score_half_correct(self):
        """Testa score com 50% de acerto"""
        predictions = np.array([0, 1, 1, 0])
        
        stacking = StackingClassifier(models=[], final_model=Mock())
        score = stacking._score(self.dataset, predictions)
        
        self.assertEqual(score, 0.5)
    
    def test_score_all_wrong(self):
        """Testa score com todas as predições erradas"""
        predictions = np.array([1, 1, 0, 0])
        
        stacking = StackingClassifier(models=[], final_model=Mock())
        score = stacking._score(self.dataset, predictions)
        
        self.assertEqual(score, 0.0)


class TestGetBaseScores(unittest.TestCase):
    """Testes para o método get_base_scores"""
    
    def setUp(self):
        """Setup para cada teste"""
        self.X = np.array([[1, 2], [2, 3], [3, 4]])
        self.y = np.array([0, 1, 1])
        self.dataset = Dataset(X=self.X, y=self.y)
    
    def test_get_base_scores_returns_dict(self):
        """Testa se retorna um dicionário"""
        model1 = Mock()
        model1.__class__.__name__ = "KNN"
        model1.predict.return_value = np.array([0, 1, 1])
        
        final_model = Mock()
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        scores = stacking.get_base_scores(self.dataset)
        
        self.assertIsInstance(scores, dict)
    
    def test_get_base_scores_correct_keys(self):
        """Testa se as chaves do dicionário estão corretas"""
        model1 = Mock()
        model1.__class__.__name__ = "KNN"
        model1.predict.return_value = np.array([0, 1, 1])
        
        model2 = Mock()
        model2.__class__.__name__ = "LogisticRegression"
        model2.predict.return_value = np.array([0, 1, 1])
        
        final_model = Mock()
        
        stacking = StackingClassifier(
            models=[model1, model2],
            final_model=final_model
        )
        
        scores = stacking.get_base_scores(self.dataset)
        
        self.assertIn("KNN_1", scores)
        self.assertIn("LogisticRegression_2", scores)
    
    def test_get_base_scores_values_are_floats(self):
        """Testa se os scores são floats"""
        model1 = Mock()
        model1.__class__.__name__ = "KNN"
        model1.predict.return_value = np.array([0, 1, 1])
        
        final_model = Mock()
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        scores = stacking.get_base_scores(self.dataset)
        
        for score in scores.values():
            self.assertIsInstance(score, (float, np.floating))


class TestStackingClassifierIntegration(unittest.TestCase):
    """Testes de integração completos"""
    
    def setUp(self):
        """Setup para testes de integração"""
        np.random.seed(42)
        
        # Dataset sintético maior
        self.X_train = np.random.randn(100, 5)
        self.y_train = (self.X_train[:, 0] + self.X_train[:, 1] > 0).astype(int)
        self.train_dataset = Dataset(X=self.X_train, y=self.y_train)
        
        self.X_test = np.random.randn(30, 5)
        self.y_test = (self.X_test[:, 0] + self.X_test[:, 1] > 0).astype(int)
        self.test_dataset = Dataset(X=self.X_test, y=self.y_test)
    
    def test_full_pipeline_with_mocks(self):
        """Testa pipeline completo com mocks"""
        # Criar modelos mock
        model1 = Mock()
        model2 = Mock()
        final_model = Mock()
        
        # Configurar retornos
        model1.predict.return_value = np.random.randint(0, 2, 100)
        model2.predict.return_value = np.random.randint(0, 2, 100)
        final_model.predict.return_value = np.random.randint(0, 2, 100)
        
        # Criar e treinar stacking
        stacking = StackingClassifier(
            models=[model1, model2],
            final_model=final_model
        )
        
        # Fit
        result = stacking.fit(self.train_dataset)
        self.assertEqual(result, stacking)
        
        # Predict
        model1.predict.return_value = np.random.randint(0, 2, 30)
        model2.predict.return_value = np.random.randint(0, 2, 30)
        final_model.predict.return_value = np.random.randint(0, 2, 30)
        
        predictions = stacking.predict(self.test_dataset)
        self.assertEqual(len(predictions), 30)
    
    def test_stacking_better_than_single_model(self):
        """
        Testa conceito: stacking geralmente tem performance igual ou melhor
        que o melhor modelo individual
        """
        # Este teste é mais conceitual - com mocks apenas verifica estrutura
        model1 = Mock()
        model2 = Mock()
        final_model = Mock()
        
        model1.predict.return_value = np.random.randint(0, 2, 100)
        model2.predict.return_value = np.random.randint(0, 2, 100)
        final_model.predict.return_value = self.y_train  # Perfect predictions
        
        stacking = StackingClassifier(
            models=[model1, model2],
            final_model=final_model
        )
        
        stacking.fit(self.train_dataset)
        
        # Verifica que o pipeline funciona
        self.assertIsNotNone(stacking.models)
        self.assertIsNotNone(stacking.final_model)


class TestStackingClassifierEdgeCases(unittest.TestCase):
    """Testes para casos extremos"""
    
    def test_single_base_model(self):
        """Testa com apenas um modelo base"""
        X = np.array([[1, 2], [2, 3]])
        y = np.array([0, 1])
        dataset = Dataset(X=X, y=y)
        
        model1 = Mock()
        model1.predict.return_value = np.array([0, 1])
        final_model = Mock()
        final_model.predict.return_value = np.array([0, 1])
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        stacking.fit(dataset)
        predictions = stacking.predict(dataset)
        
        self.assertIsNotNone(predictions)
    
    def test_many_base_models(self):
        """Testa com muitos modelos base"""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1, 1])
        dataset = Dataset(X=X, y=y)
        
        # 10 modelos base
        models = []
        for i in range(10):
            model = Mock()
            model.predict.return_value = np.array([0, 1, 1])
            models.append(model)
        
        final_model = Mock()
        final_model.predict.return_value = np.array([0, 1, 1])
        
        stacking = StackingClassifier(models=models, final_model=final_model)
        
        stacking.fit(dataset)
        predictions = stacking.predict(dataset)
        
        self.assertEqual(len(predictions), 3)
    
    def test_binary_classification(self):
        """Testa especificamente classificação binária"""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        dataset = Dataset(X=X, y=y)
        
        model1 = Mock()
        model1.predict.return_value = np.array([0, 0, 1, 1])
        
        final_model = Mock()
        final_model.predict.return_value = np.array([0, 0, 1, 1])
        
        stacking = StackingClassifier(models=[model1], final_model=final_model)
        
        stacking.fit(dataset)
        predictions = stacking.predict(dataset)
        
        # Verificar que só tem 0s e 1s
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))


if __name__ == '__main__':
    unittest.main()