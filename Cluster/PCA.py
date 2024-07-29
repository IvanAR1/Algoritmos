import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import joblib

def pca(n_components: int = None):
    return PCA(n_components) # Inicializa el modelo PCA

def explained_variance(pca: PCA):
    return pca.explained_variance_ratio_ # Devuelve la varianza explicada por cada componente principal

def n_components_variance(explained_variance: list):
    return np.argmax(np.cumsum(explained_variance) >= 0.95) + 1 # Devuelve el número de componentes principales que explican el 95% de la varianza

def graph_variance(explained_variance: list):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--') # Grafica la varianza explicada por cada componente principal
    plt.title('Varianza explicada por cada componente principal') # Añade título al gráfico
    plt.xlabel('Componente Principal') # Añade etiqueta al eje x
    plt.ylabel('Varianza Explicada') # Añade etiqueta al eje y
    plt.show()

def save_pca_model(model, file_path):
    joblib.dump(model, file_path) # Guarda el modelo en un archivo

def charge_pca_model(file_path):
    print(joblib.load(file_path)) # Carga el modelo desde un archivo