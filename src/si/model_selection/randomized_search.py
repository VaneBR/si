import numpy as np
from si.data.dataset import Dataset
from si.base.model import Model
from si.model_selection.cross_validate import k_fold_cross_validation
from typing import Dict, Callable, List, Any

def randomized_search_cv(model:Model, 
                         dataset:Dataset,
                         hyperparameter_distributions:Dict[str, np.ndarray],
                         scoring:Callable=None, 
                         cv:int=5, 
                         n_iter:int=10,
                         seed:int=42) -> Dict: 
    """
    Realiza Randomized Search com Cross Validation para otimização de hiperparametros
    Testa n_iter combinações aleatorias de hiperparametros de distribuição fornecidas
    usando cross-validation e retorna a melhor combinação encontrada

    Vantagens em relação ao Grid Search:
    - Mais rapido (testa apenas n_iter combinações ao invés de todas)
    - Explora melhor o espaço de hiperparametros
    -Útil quando há muitos hiperparametros ou valores possíveis
    """

    #Definir seed
    np.random.seed(seed)

    #1. Validar se os hiperparametros existem no modelo
    print("="*80)
    print("Randomized Search com Cross Validation")
    print("="*80)

    for param_name in hyperparameter_distributions.keys():
        if not hasattr(model, param_name):
            raise ValueError(f"Modelo {model.__class__.__name__} não tem "
                             f"o hiperparâmetro '  {param_name}'")
        
    print(f"\nModelo: {model.__class__.__name__}")
    print(f"Dataset: {dataset.X.shape()}")
    print(f"CV Folds: {cv}")
    print(f"Iterações (combinações aleatorias): {n_iter}")
    print(f"Random seed: {seed}")

    print(f"\nDistribuições de hiperparâmetros:")
    for param, distribution in hyperparameter_distributions.items():
        print(f" {param}: {len(distribution)} valores"
              f"[{np.min(distribution):.4f} - {np.max(distribution):.4f}]")

    #Calcular total de combinações possíveis
    total_possible = np.prod([len(v) for v in hyperparameter_distributions.values()])
    print(f"\nTotal de combinações possíveis: {total_possible}")
    print(f"Testando {n_iter} combinações aleatórias"
          f"({(n_iter/total_possible)*100:.2f}% do espaço)")
    print(f"Total de treinos: {n_iter * cv}")
    print("\n"+"-"*80)

    #2. Gerar n_iter combinações aleatorias
    param_names=list(hyperparameter_distributions.keys())
    param_distributions=list(hyperparameter_distributions.values())

    #Listas para armazenar resultados
    all_hyperparameters = []
    all_scores = []

    #Gerar combinações únicas
    tested_combinations = set()

    iteration=0
    while iteration < n_iter:
        #Seleciona um valor aleatorio para cada distribuição 
        current_combination = tuple(np.random.choice(distribution)
                                    for distribution in param_distributions)
        
        #Verifica se a combinação já foi testada (evitar duplicatas)
        if current_combination in tested_combinations:
            continue

        tested_combinations.add(current_combination)

        #Criar dicionario com a combinação atual
        current_params=dict(zip(param_names, current_combination))

        print(f"\nIteração {iteration + 1}/{n_iter}:")
        print(f"Params: {current_params}")

        #3. Definir os hiperparametros no modelo
        for param_name, param_value in current_params.items():
            setattr(model, param_name, param_value)

        #4. Cross-validate
        try: 
            scores = k_fold_cross_validation(model=model,
                                             dataset=dataset,
                                             scoring=scoring,
                                             cv=cv, 
                                             seed=seed)
            
            #5. Calcular média dos scores
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"Score: {mean_score:.4f} +/- {std_score:.4f}")

            #Armazenar resultados
            all_hyperparameters.append(current_params.copy())
            all_scores.append(-np.inf)
            iteration += 1

        except Exception as e:
            print(f"Erro: {str(e)}")
            #Em caso de erro, adicionar score -inf
            all_hyperparameters.append(current_params.copy())
            all_scores.append(-np.inf)
            iteration += 1

    #6. Encontrar melhor score e hiperparametros
    best_index = np.argmax(all_scores)
    best_score=all_scores[best_index]
    best_hyperparameters=all_hyperparameters[best_index]

    #7. Preparar resultado
    results = {
        'hyperparameters': all_hyperparameters,
        'scores': all_scores,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }

    print("\n"+"="*80)
    print("Resultados do Randomized Search")
    print("="*80)

    print(f"\n{'#':<5} {'Score':<12} {'Hyperparameters'}")
    print("-"*80)

    #Ordenar por score (melhor primeiro)
    sorted_indices = np.argsort(all_scores)[::-1]

    for ranks, idx in enumerate(sorted_indices, 1):
        score = all_scores[idx]
        params = all_hyperparameters[idx]
        marker="Best if idx==best_index else "
        print(f"{ranks:<5} {score:<12.4f} {params} {marker}")    

    print("\n" + "="*80)
    print("Melhor configuração")
    print("="*80)
    print(f"\nBest score: {best_score:.4f}")
    print(f"Best hyperparameters:")
    for param, value in best_hyperparameters.items():
        print(f" - {param}: {value}")
    
    return results