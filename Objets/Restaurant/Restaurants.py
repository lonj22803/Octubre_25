"""
Restaurants class that inherits from CityBase for generalized cluster operations.
"""

from Objets.City.City import CityBase
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import numpy as np
import pandas as pd
import random


class Restaurants(CityBase):
    """
    Clase para representar restaurantes en la ciudad.
    Hereda de CityBase para utilizar métodos generalizados de clustering.
    """

    def __init__(self, lat: float, lon: float, num_restaurantes: int, 
                max_distancia_km: float, mean_distancia_km: float,
                lambda_poisson: float, alpha: float, beta: float, 
                mean_tiempo: float, std_tiempo: float):
        """
        Inicializa los restaurantes.
        
        Args:
            lat: Centro en latitud
            lon: Centro en longitud
            num_restaurantes: Número de restaurantes a generar
            max_distancia_km: Distancia máxima desde el centro en km
            mean_distancia_km: Media de la distribución exponencial en km
            lambda_poisson: Parámetro lambda para distribución Poisson (precio)
            alpha: Parámetro alpha para distribución Beta (calificación)
            beta: Parámetro beta para distribución Beta (calificación)
            mean_tiempo: Media para tiempo promedio (minutos)
            std_tiempo: Desviación estándar para tiempo promedio
        """
        self.city_center = (lat, lon)
        self.num_restaurantes = num_restaurantes
        self.max_distancia_km = max_distancia_km
        self.mean_distancia_km = mean_distancia_km
        self.lambda_poisson = lambda_poisson
        self.alpha = alpha
        self.beta = beta
        self.mean_tiempo = mean_tiempo
        self.std_tiempo = std_tiempo
        self.restaurantes = pd.DataFrame()
        self.clusters_info = {}
        self.cluster_weights = {}

    def restaurant_generation_points(self, system, stations, station_coordinates):
        """
        Genera puntos de restaurantes usando clustering generalizado.
        Utiliza métodos heredados de CityBase.
        
        Args:
            system: Sistema de metro
            stations: Lista de estaciones
            station_coordinates: Diccionario de coordenadas
            
        Returns:
            DataFrame con los restaurantes generados
        """
        # Validar parámetros
        validated = self.validate_parameters(
            system=system,
            stations=stations,
            station_coordinates=station_coordinates
        )
        
        # Identificar estaciones con múltiples líneas
        repeat_stations, freq = self.identify_repeat_stations(system)
        
        # Ordenar por frecuencia
        repeat_stations_sorted = sorted(repeat_stations, key=lambda x: freq[x], reverse=True)
        
        # Obtener coordenadas
        coords_repeat_stations = [
            station_coordinates[station]
            for station in repeat_stations_sorted
            if station in station_coordinates
        ]
        
        if not coords_repeat_stations:
            raise ValueError("No se encontraron estaciones con más de una línea de metro.")
        
        print(f"Estaciones con cruces (ordenadas por frecuencia): {repeat_stations_sorted}")
        
        # Generar clusters
        self.clusters_info = self.generate_clusters(
            coords_repeat_stations,
            n_clusters=min(4, len(coords_repeat_stations)),
            method='kmeans'
        )
        
        # Calcular pesos con mayor influencia del centro
        self.cluster_weights = self.calculate_cluster_weights(
            self.clusters_info,
            self.city_center,
            center_weight_multiplier=3.0  # Mayor influencia para restaurantes
        )
        
        # Asignar pesos a estaciones
        station_weights = self.assign_weights_to_entities(
            repeat_stations_sorted,
            coords_repeat_stations,
            self.clusters_info,
            self.cluster_weights
        )
        
        # Distribuir restaurantes
        restaurant_distribution = self.distribute_entities_by_weights(
            self.num_restaurantes,
            repeat_stations_sorted,
            station_weights
        )
        
        print(f"Distribución de restaurantes por estación: {restaurant_distribution}")
        
        # Generar restaurantes
        nombre_restaurantes = [f"RST{i + 1}" for i in range(self.num_restaurantes)]
        latitudes = []
        longitudes = []
        precios_promedio = []
        calificaciones = []
        tipos_comida = []
        tiempos_promedio = []

        for station in repeat_stations_sorted:
            num_rest_for_station = restaurant_distribution.get(station, 0)
            if station not in station_coordinates:
                continue
                
            lat_est, lon_est = station_coordinates[station]
            
            for _ in range(num_rest_for_station):
                if len(latitudes) >= self.num_restaurantes:
                    break

                # Generar distancia y ángulo
                distancia = np.random.exponential(self.mean_distancia_km)
                while distancia > self.max_distancia_km:
                    distancia = np.random.exponential(self.mean_distancia_km)
                angulo = random.uniform(0, 2 * np.pi)

                # Calcular coordenadas
                delta_lat = (distancia / 111) * np.cos(angulo)
                delta_lon = (distancia / (111 * np.cos(np.radians(lat_est)))) * np.sin(angulo)
                nueva_lat = lat_est + delta_lat
                nueva_lon = lon_est + delta_lon

                latitudes.append(round(nueva_lat, 6))
                longitudes.append(round(nueva_lon, 6))

                # Generar atributos
                precio_promedio = np.random.poisson(self.lambda_poisson)
                precios_promedio.append(max(5, precio_promedio))
                
                calificacion = np.random.beta(self.alpha, self.beta) * 4 + 1
                calificaciones.append(round(calificacion, 1))
                
                tipo_comida = np.random.choice(
                    ['comida típica', 'comida rápida', 'comida extranjera']
                )
                tipos_comida.append(tipo_comida)
                
                tiempo_promedio = np.random.normal(self.mean_tiempo, self.std_tiempo)
                tiempos_promedio.append(max(0, round(tiempo_promedio, 1)))

        self.restaurantes = pd.DataFrame({
            'nombre_restaurante': nombre_restaurantes[:len(latitudes)],
            'latitud': latitudes,
            'longitud': longitudes,
            'precio_promedio': precios_promedio,
            'calificacion': calificaciones,
            'tipo_comida': tipos_comida,
            'tiempo_promedio': tiempos_promedio
        })

        return self.restaurantes

    def save_restaurantes_csv(self, filename: str):
        """
        Guarda los datos de los restaurantes en un CSV.
        
        Args:
            filename: Nombre del archivo de salida
        """
        if self.restaurantes.empty:
            raise ValueError(
                "No hay datos de restaurantes para guardar. "
                "Genera los restaurantes primero."
            )
        self.restaurantes.to_csv(filename, index=False)
        print(f"Datos de restaurantes guardados en {filename}")

    def plot_restaurantes(self, lat_city=None, lon_city=None):
        """
        Grafica los restaurantes generados en un mapa.
        
        Args:
            lat_city: Latitud del centro de la ciudad (opcional)
            lon_city: Longitud del centro de la ciudad (opcional)
        """
        if self.restaurantes.empty:
            raise ValueError(
                "Primero debes generar los restaurantes con restaurant_generation_points()"
            )

        gdf_restaurantes = gpd.GeoDataFrame(
            self.restaurantes,
            geometry=gpd.points_from_xy(
                self.restaurantes['longitud'], 
                self.restaurantes['latitud']
            ),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        centro_restaurantes = gpd.GeoSeries(
            [Point(self.city_center[1], self.city_center[0])], 
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        if lat_city is not None and lon_city is not None:
            centro_ciudad = gpd.GeoSeries(
                [Point(lon_city, lat_city)], 
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
        else:
            centro_ciudad = None

        fig, ax = plt.subplots(figsize=(12, 10))

        gdf_restaurantes.plot(ax=ax, color='red', markersize=150, alpha=0.7, label='Restaurantes')
        centro_restaurantes.plot(ax=ax, color='blue', markersize=50, label='Centro Restaurantes', zorder=3)

        if centro_ciudad is not None:
            centro_ciudad.plot(ax=ax, color='purple', markersize=50, label='Centro Ciudad', zorder=3)

        buffer = 5000  # metros
        cx, cy = centro_restaurantes.geometry[0].x, centro_restaurantes.geometry[0].y
        ax.set_xlim(cx - buffer, cx + buffer)
        ax.set_ylim(cy - buffer, cy + buffer)

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        ax.set_title("Distribución de Restaurantes en la Ciudad", fontsize=16, fontweight='bold')
        ax.legend()
        ax.set_axis_off()
        plt.tight_layout()
        
        ruta_salida = os.path.join(os.path.dirname(__file__), "restaurantes_mapa.png")
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
