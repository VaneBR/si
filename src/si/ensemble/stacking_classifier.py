import numpy as np
from si import models
from si.base.model import Model
from si.data import dataset
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from typing import List

class StackingClassifier(Model):
    """Stacking Classifier - Ensemle de dois níveis.
    O StackingClassifier usa predições de múltiplos modelos (base models)
    como features para treinar um modelo final (meta-learner) que faz
    a predição final.
    
    Architecture:
    1. Base models fazem predições no treino
    2. Predições servem como features para o modelo final
    3. Modelo final é treinado nesssas features
    4. Para predição: base models -> features -> final model -> predição
    
    """

    def __init__(self, models: List[Model], final_model: Model):
        """Inicializa o StackingClassifier."""
        super().__init__()
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """Treina o ensemble em duas etapas.
        Etapa 1: Treina os modelos base
        1. Para cada modelo base, treina no dataset original

        Etapa 2: Treina o modelo final
        2. Obter predições dos modelos base no dataset de treino
        3. Usa essas predições como features para treinar o modelo final
        4. Treina o modelo final com essas novas features
        """
        if dataset.y is None:
            raise ValueError("Dataset precisa de ter labels (y) para treino.")
    
        # Etapa 1: Treina os modelos base
        print(f"Treinando modelo {i+1}/{len(self.models)}: {model.__class__.__name__}")

        for i, model in enumerate(self.models):
            print(f"Treinando modelo {i+1}/{len(self.models)}: {model.__class__.__name__}")
            model.fit(dataset)

        # Etapa 2: Obter predições dos modelos base
        print(f"\nObtendo predições dos modelos base para treino do modelo final...")

        base_predictions = []
        for i, model in enumerate(self.models):
            predictions = model.predict(dataset)
            base_predictions.append(predictions)
        
        #Converter predições para features numericas
        #Cada modelo contribui com uma coluna de predições
        X_meta=self._predictions_to_features(base_predictions)

        #Criar dataset para o modelo final
        meta_dataset=Dataset(
            X=X_meta,
            y=dataset.y,
            features=[f"model_{i+1}_pred" for i in range(len(self.models))],
            label=dataset.label
        )

        #Etapa 3: Treinar o modelo final
        print(f"Treinando o modelo final: {self.final_model.__class__.__name__}")
        self.final_model.fit(meta_dataset)

        print("StackingClassifier treinado com sucesso.")
        return self

    def _predictions_to_features(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Converte predições dos modelos base em features numericas
        Se as predições forem categóricas, converte para numerico usando label enconding"""

        n_samples=len(predictions_list[0])
        n_models=len(predictions_list)

        X_meta=np.zeros((n_samples,n_models))

        for i, predictions in enumerate(predictions_list):
            #Se as predições são categoricas, fazer label encoding
            if predictions.dtype==object or not np.issubdtype(predictions.dtype, np.numer):
                unique_labels=np.unique(predictions)
                label_map={label:idx for idx, label in enumerate(unique_labels)}
                predictions_numeric=np.array([label_map[pred] for pred in predictions])
                X_meta[:,i]=predictions_numeric
            else:
                X_meta[:,i]=predictions
        return X_meta
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """Faz predições usando o ensemble
        1. Obter predições dos modelos base 
        2. Usar essas predições como features para o modelo final
        3. Usar modelo final para predição final
        """

        if not self.models or self.final_model is None: 
            raise ValueError("Modelo precisa de ser treinado antes de fazer predições.")
        
        #1. Obter predições dos modelos base
        base_predictions = []
        for model in self.models:
            predictions = model.predict(dataset)
            base_predictions.append(predictions)
        
        #2. Converter predições para features 
        X_meta=self._predictions_to_features(base_predictions)

        #3. Criar dataset meta e fazer predição final
        meta_dataset = Dataset (
            X=X_meta,
            y=dataset.y
            features=[f"model_{i+1}_pred" for i in range(len(self.models))],
            label=dataset.label
        )

        final_predictions = self.final_model.predict(meta_dataset)

        return final_predictions
    
    def _score(self, dataset: Dataset) -> float:
        """Calcula a accuracy das predições"""
        
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)
    
    def get_base_scores(self, dataset:Dataset) -> dict: 
        """Retorna accuracy de cada modelo base individualmente"""

        scores = {}

        for i, model in enumerate(self.models):
            model_name = f"{model.__class__.__name__}_{i+1}"
            score= model._score(dataset)
            scores[model_name]=score
        return scores
    