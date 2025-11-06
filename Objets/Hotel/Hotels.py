"""
Hotels class that inherits from CityBase for generalized cluster operations.
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


class Hotels(CityBase):
    """
    Clase para representar hoteles en la ciudad.
    Hereda de CityBase para utilizar métodos generalizados de clustering.
    """

    def __init__(self, lat: float, lon: float, num_hoteles: int, 
                max_distancia_km: float, mean_distancia_km: float,
                mean_precio: float, std_precio: float):
        """
        Inicializa un sistema de hoteles.
        
        Args:
            lat: Centro en latitud
            lon: Centro en longitud
            num_hoteles: Número de hoteles a generar
            max_distancia_km: Distancia máxima desde el centro en km
            mean_distancia_km: Media de la distribución exponencial en km
            mean_precio: Precio medio de los hoteles
            std_precio: Desviación estándar del precio
        """
        self.city_center = (lat, lon)
        self.num_hoteles = num_hoteles
        self.max_distancia_km = max_distancia_km
        self.mean_distancia_km = mean_distancia_km
        self.mean_precio = mean_precio
        self.std_precio = std_precio
        self.hoteles = pd.DataFrame()
        self.clusters_info = {}
        self.cluster_weights = {}

    def hotel_generation_points(self, system, stations, station_coordinates):
        """
        Genera puntos de hoteles usando clustering generalizado.
        Utiliza métodos heredados de CityBase para clustering y asignación de pesos.
        
        Args:
            system: Sistema de metro
            stations: Lista de estaciones
            station_coordinates: Diccionario de coordenadas de estaciones
            
        Returns:
            DataFrame con los hoteles generados
        """
        # Validar parámetros usando método heredado
        validated = self.validate_parameters(
            system=system,
            stations=stations,
            station_coordinates=station_coordinates
        )
        
        # Identificar estaciones con múltiples líneas usando método heredado
        repeat_stations, freq = self.identify_repeat_stations(system)
        
        # Obtener coordenadas de estaciones repetidas
        coords_repeat_stations = [
            station_coordinates[station] 
            for station in repeat_stations 
            if station in station_coordinates
        ]
        
        if not coords_repeat_stations:
            raise ValueError(
                "No se encontraron estaciones con más de una línea de metro "
                "o no tienen coordenadas asignadas."
            )
        
        print(f"Estaciones con múltiples líneas: {repeat_stations}")
        
        # Generar clusters usando método heredado
        self.clusters_info = self.generate_clusters(
            coords_repeat_stations,
            n_clusters=min(3, len(coords_repeat_stations)),
            method='kmeans'
        )
        
        # Calcular pesos de clusters con mayor influencia del centro
        self.cluster_weights = self.calculate_cluster_weights(
            self.clusters_info,
            self.city_center,
            center_weight_multiplier=2.5  # Aumentar influencia del centro
        )
        
        # Asignar pesos a las estaciones
        station_weights = self.assign_weights_to_entities(
            repeat_stations,
            coords_repeat_stations,
            self.clusters_info,
            self.cluster_weights
        )
        
        # Distribuir hoteles por estaciones
        hotel_distribution = self.distribute_entities_by_weights(
            self.num_hoteles,
            repeat_stations,
            station_weights
        )
        
        # Generar hoteles
        nombre_hoteles = [f"Hotel_{i+1}" for i in range(self.num_hoteles)]
        latitudes = []
        longitudes = []
        precios = []
        precios_todo_incluido = []
        calificacion_hotel = []

        for station in repeat_stations:
            num_hotels_for_station = hotel_distribution.get(station, 0)
            if station not in station_coordinates:
                continue
                
            lat_est, lon_est = station_coordinates[station]
            
            for _ in range(num_hotels_for_station):
                if len(latitudes) >= self.num_hoteles:
                    break
                
                # Generar distancia y ángulo aleatorio
                distancia = np.random.exponential(self.mean_distancia_km)
                while distancia > self.max_distancia_km:
                    distancia = np.random.exponential(self.mean_distancia_km)
                angulo = random.uniform(0, 2 * np.pi)

                # Calcular nuevas coordenadas usando método heredado
                nueva_lat, nueva_lon = self.calculate_new_coordinates(
                    lat_est, lon_est, distancia, angulo
                )
                
                latitudes.append(round(nueva_lat, 6))
                longitudes.append(round(nueva_lon, 6))
                
                precio = max(20, np.random.normal(self.mean_precio, self.std_precio))
                precios.append(round(precio, 2))
                precios_todo_incluido.append(round(precio * 1.3, 2))
                calificacion_hotel.append(round(random.uniform(1, 5), 1))

        self.hoteles = pd.DataFrame({
            'nombre_hotel': nombre_hoteles[:len(latitudes)],
            'latitud': latitudes,
            'longitud': longitudes,
            'precio': precios,
            'precio_todo_incluido': precios_todo_incluido,
            'calificacion': calificacion_hotel
        })

        return self.hoteles

    def save_hotels_csv(self, filename: str):
        """
        Guarda los datos de los hoteles en un archivo CSV.
        
        Args:
            filename: Nombre del archivo de salida
        """
        if self.hoteles.empty:
            raise ValueError("No hay datos de hoteles para guardar. Genera los hoteles primero.")
        self.hoteles.to_csv(filename, index=False)
        print(f"Datos de hoteles guardados en {filename}")

    def plot_hotels(self, lat_city=None, lon_city=None):
        """
        Grafica los hoteles generados en un mapa.
        
        Args:
            lat_city: Latitud del centro de la ciudad (opcional)
            lon_city: Longitud del centro de la ciudad (opcional)
        """
        if self.hoteles.empty:
            raise ValueError("Primero debes generar los hoteles con hotel_generation_points()")

        # Convertir hoteles a GeoDataFrame
        gdf_hoteles = gpd.GeoDataFrame(
            self.hoteles,
            geometry=gpd.points_from_xy(self.hoteles['longitud'], self.hoteles['latitud']),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Centro del sistema de hoteles
        centro_hoteles = gpd.GeoSeries(
            [Point(self.city_center[1], self.city_center[0])], 
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Centro de la ciudad si se proporciona
        if lat_city is not None and lon_city is not None:
            centro_ciudad = gpd.GeoSeries([Point(lon_city, lat_city)], crs="EPSG:4326").to_crs(epsg=3857)
        else:
            centro_ciudad = None

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dibujar hoteles
        gdf_hoteles.plot(ax=ax, color='green', markersize=150, alpha=0.7, label='Hoteles')

        # Dibujar centros
        centro_hoteles.plot(ax=ax, color='blue', markersize=50, label='Centro Hoteles', zorder=3)

        # Ajustar límites
        buffer = 5000  # metros
        cx, cy = centro_hoteles.geometry[0].x, centro_hoteles.geometry[0].y
        ax.set_xlim(cx - buffer, cx + buffer)
        ax.set_ylim(cy - buffer, cy + buffer)

        # Añadir mapa base
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Configuración final
        ax.set_title("Distribución de Hoteles en la Ciudad", fontsize=16, fontweight='bold')
        ax.legend()
        ax.set_axis_off()
        plt.tight_layout()
        
        ruta_salida = os.path.join(os.path.dirname(__file__), "hoteles_mapa.png")
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
