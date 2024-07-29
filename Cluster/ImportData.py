import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def import_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna() # Elimina las filas con valores nulos
    data = data.drop_duplicates() # Elimina las filas duplicadas
    data = data.select_dtypes(include=[np.number]) # Selecciona las columnas numéricas
    scaler = StandardScaler() # Estandariza los datos
    data_scaled = scaler.fit_transform(data) # Ajusta y transforma los datos
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns) # Convierte los datos a un DataFrame
    return data_scaled