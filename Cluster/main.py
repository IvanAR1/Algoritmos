from ImportData import import_data, data_preprocessing
from PCA import pca, explained_variance, n_components_variance, graph_variance, save_pca_model, charge_pca_model
from K_Means import clusters_selects, graph_elbow, select_clusters, evaluate_clusters, graph_clusters, save_kmeans_model, charge_kmeans_model
import os

def main():
    file_path = os.path.dirname(__file__) + "/files/data.csv"
    data = import_data(file_path)
    # pca_example(data_preprocessing(data))
    kmeans_example(data_preprocessing(data))

def pca_example(data_processed):
    pca_data = pca()
    pca_data.fit_transform(data_processed)
    variance = explained_variance(pca_data)
    n_components = n_components_variance(variance)
    print("Varianza explicada por cada componente principal: ", variance)
    print("Número de componentes principales que explican el 95% de la varianza: ", n_components)
    graph_variance(variance)
    file_path = os.path.dirname(__file__) + "/files/pca_model.joblib"
    save_pca_model(pca_data.fit_transform(data_processed), file_path)
    charge_pca_model(file_path)

def kmeans_example(data_processed):
    wcss = clusters_selects(data_processed)
    graph_elbow(wcss)
    clusters = select_clusters(data_processed)
    evaluate_clusters(data_processed, clusters)
    graph_clusters(data_processed, clusters)
    file_path = os.path.dirname(__file__) + "/files/kmeans_model.joblib"
    save_kmeans_model(clusters, file_path)
    charge_kmeans_model(file_path)

main()