import numpy as np
import pandas as pd
from si.io.csv_file import read_csv


# Exercício 1: NumPy array Indexing/Slicing

# 1.1) Carregar o dataset iris.csv
print("=== 1.1) Carregar iris.csv ===")
iris_dataset = read_csv(filename='../datasets/iris/iris.csv', sep=',', features=True, label=True)
print(f"Dataset carregado com shape: {iris_dataset.shape()}")
print(f"Features: {iris_dataset.features}")
print(f"Label: {iris_dataset.label}")

# 1.2) Selecionar a penúltima variável independente
print("\n=== 1.2) Penúltima variável independente ===")
penultimate_feature = iris_dataset.X[:, -2]
print(f"Dimensão do array resultante: {penultimate_feature.shape}")
print(f"Primeiros 5 valores: {penultimate_feature[:5]}")

# 1.3) Selecionar as últimas 10 amostras
print("\n=== 1.3) Últimas 10 amostras ===")
last_10_samples = iris_dataset.X[-10:, :]
mean_last_10 = np.mean(last_10_samples, axis=0)
print(f"Shape das últimas 10 amostras: {last_10_samples.shape}")
print(f"Média das últimas 10 amostras para cada feature:")
for i, feature_name in enumerate(iris_dataset.features):
    print(f"  {feature_name}: {mean_last_10[i]:.4f}")

# 1.4) Selecionar amostras com valores <= 6 em todas as features
print("\n=== 1.4) Amostras com valores <= 6 em todas as features ===")
mask = np.all(iris_dataset.X <= 6, axis=1)
samples_le_6 = iris_dataset.X[mask]
print(f"Número de amostras obtidas: {samples_le_6.shape[0]}")

# 1.5) Selecionar amostras com classe diferente de 'Iris-setosa'
print("\n=== 1.5) Amostras com classe != 'Iris-setosa' ===")
mask = iris_dataset.y != 'Iris-setosa'
samples_not_setosa = iris_dataset.X[mask]
y_not_setosa = iris_dataset.y[mask]
print(f"Número de amostras obtidas: {samples_not_setosa.shape[0]}")
print(f"Classes únicas nas amostras selecionadas: {np.unique(y_not_setosa)}")