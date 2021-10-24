import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class TransformerFechas(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        columna_fecha = pd.to_datetime(X["pickup_datetime"])
        fecha_df = pd.DataFrame()
        fecha_df["weekday"] = columna_fecha.dt.weekday
        fecha_df["hour"] = columna_fecha.dt.hour
        return fecha_df

class TransformerDistancia(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_init = X[["pickup_latitude", "pickup_longitude"]].to_numpy()
        X_final = X[["dropoff_latitude", "dropoff_longitude"]].to_numpy()

        # Distancia de Haversine
        distancia = self.distancia_haversine(X_init=X_init, X_final=X_final)
        distancia_df = pd.DataFrame()
        distancia_df["distancia"] = distancia
        return distancia_df
    
    def distancia_haversine(self, X_init, X_final):
        # Convertir de decimal a radianes
        X_init = np.radians(X_init)
        X_final = np.radians(X_final)

        # haversine formula 
        dlat = X_final[:, 0] - X_init[:, 0] 
        dlon = X_final[:, 1] - X_init[:, 1]
        a = np.sin(dlat / 2) ** 2 + np.cos(X_init[:, 0]) * np.cos(X_final[:, 0]) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r
    
class TransformerVelocidad(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        X_init = X[["pickup_latitude", "pickup_longitude"]].to_numpy()
        X_final = X[["dropoff_latitude", "dropoff_longitude"]].to_numpy()

        # Distancia de Haversine
        distancia = self.distancia_haversine(X_init=X_init, X_final=X_final)
        
        velocidad_df = pd.DataFrame()
        tiempo_en_horas = y.to_numpy() / 3600
        velocidad_df["velocidad"] = distancia / tiempo_en_horas
        velocidad_df["pickup_borough"] = X["pickup_borough"]
        velocidad_borough = velocidad_df.groupby("pickup_borough")["velocidad"].mean()
        self.velocidad_borough = velocidad_borough.to_dict()
        return self

    def transform(self, X, y=None):
        velocidad_df = pd.DataFrame()
        velocidad_df["velocidad"] = X["pickup_borough"].map(self.velocidad_borough)
        return velocidad_df
    
    def distancia_haversine(self, X_init, X_final):
        # Convertir de decimal a radianes
        X_init = np.radians(X_init)
        X_final = np.radians(X_final)

        # Haversine formula 
        dlat = X_final[:, 0] - X_init[:, 0] 
        dlon = X_final[:, 1] - X_init[:, 1]
        a = (np.sin(dlat / 2) ** 2) + np.cos(X_init[:, 0]) * np.cos(X_final[:, 0]) * (np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371 # Radio de la tierra en kilómetros
        return c * r