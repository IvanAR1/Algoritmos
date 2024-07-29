from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

def clusters_selects(pca_data):
    wcss = [] # Inicializa la lista de WCSS
    for i in range(1, 11): # Itera sobre el rango de 1 a 10
        kmeans = KMeans(n_clusters=i, random_state=42) # Inicializa el modelo KMeans
        kmeans.fit(pca_data) # Ajusta el modelo
        wcss.append(kmeans.inertia_) # Añade el valor de WCSS a la lista
    return wcss # Devuelve la lista de WCSS

def select_clusters(pca_data, optimal_clusters: int = 3):
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42) # Inicializa el modelo KMeans
    return kmeans.fit_predict(pca_data) # Ajusta y predice los clústers

def evaluate_clusters(pca_data:PCA, clusters, optimal_clusters: int = 3):
    silhouette_avg = silhouette_score(pca_data, clusters) # Calcula el Silhouette Score
    print('Silhouette Score para %s clusters: %s' % (optimal_clusters, silhouette_avg)) # Imprime el Silhouette Score

def graph_elbow(wcss):
    plt.figure(figsize=(10, 6))  # Inicializa el gráfico
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--') # Grafica el número de clústers vs. WCSS
    plt.title('Método del codo') # Añade título al gráfico
    plt.xlabel('Número de clústers') # Añade etiqueta al eje x
    plt.ylabel('WCSS') # Añade etiqueta al eje y
    plt.show() # Muestra el gráfico

def graph_clusters(data, clusters):
    pca = PCA(n_components=2) # Inicializa el modelo PCA
    pca_data = pca.fit_transform(data) # Ajusta y transforma los datos
    plt.figure(figsize=(10, 6)) # Inicializa el gráfico
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters) # Grafica los clústers
    plt.legend(*scatter.legend_elements(), title='Clusters') # Añade la leyenda
    plt.title('Clústers') # Añade título al gráfico
    plt.xlabel('Componente Principal 1') # Añade etiqueta al eje x
    plt.ylabel('Componente Principal 2') # Añade etiqueta al eje y
    plt.show() # Muestra el gráfico

def save_kmeans_model(model, file_path):
    joblib.dump(model, file_path) # Guarda el modelo en un archivo

def charge_kmeans_model(file_path):
    print(joblib.load(file_path)) # Carga el modelo desde un archivo