from funciones_propias import *
import os
import  networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from pyproj import Transformer
import numpy as np
import pandas as pd
import random

class Hotel:
    """
    Clase para representar un hotel en la ciudad, que hereda del sistema de metro.
    El centro en latitud y longitud del hotel se utiliza para centrar el sistema de metro.
    Hereda de la clase TrainSystem.
    El centro en longitud y latitud para generar de manera automatica los hoteles con una
    distribucion de probabilidad de exponbencial respecto al centro de la ciudad.
    """
    def __init__(self, lat: float, lon: float, num_hoteles: int, max_distancia_km: float, mean_distancia_km: float,
                 mean_precio: float, std_precio: float):
        """
        Inicializa un hoteles con su ubicación y parámetros de distribución.
        :param lat: centro en latitud
        :param lon: cemtro en longitud
        :param num_hoteles: número de hoteles a generar
        :param max_distancia_km: distancia máxima desde el centro en km
        :param mean_distancia_km: media de la distribución exponencial en km
        :param mean_precio: precio medio de los hoteles
        :param std_precio: desviación estándar del precio de los hoteles
        """
        self.city_center = (lat, lon)
        self.num_hoteles = num_hoteles
        self.max_distancia_km = max_distancia_km
        self.mean_distancia_km = mean_distancia_km
        self.mean_precio = mean_precio
        self.std_precio = std_precio
        self.hoteles = pd.DataFrame()

    def hotel_generation_points(self, system, stations,station_coordinates):
        """
        Genera puntos de hoteles alrededor de las estaciones de metro donde se cuentran mas de
        una linea de metro y a partir de ellos crea una distribución de probabilidad exponencial
        para generar los hoteles, al rededor de esos puntos, toma todos los toteles que se desean
        generar y los distribuye en la ciudad.
        :return: DataFrame con las coordenadas y precios de los hoteles generados.
        """
        nombre_hoteles = [f"Hotel_{i+1}" for i in range(self.num_hoteles)]
        latitudes = []
        longitudes = []
        precios = []
        precios_todo_incluido = []
        calificacion_hotel = []

        #Identificamos las estaciones con mas de una linea de metro
        stations_all_lines =[]
        for line, date in system.items():
            stations_all_lines.extend(date["estaciones"])

        repeat_stations = [station for station in set(stations_all_lines) if stations_all_lines.count(station) > 1]

        #Obtenemos las coordenadas de esas estaciones
        coords_repeat_stations = [station_coordinates[station] for station in repeat_stations if station in station_coordinates]
        if not coords_repeat_stations:
            raise ValueError("No se encontraron estaciones con más de una línea de metro o no tienen coordenadas asignadas.")
        print("====Las estaciones que tienen mas de una linea de metro =====\n ", repeat_stations)

        #Generamos los hoteles alrededor de esas estaciones
        hotel_for_station = max(1, self.num_hoteles // len(coords_repeat_stations))
        for (lat_est, lon_est) in coords_repeat_stations:
            for _ in range(hotel_for_station):
                if len(latitudes) >= self.num_hoteles:
                    break
                # Generar distancia y ángulo aleatorio
                distancia = np.random.exponential(self.mean_distancia_km)
                while distancia > self.max_distancia_km:
                    distancia = np.random.exponential(self.mean_distancia_km)
                angulo = random.uniform(0, 2 * np.pi)

                # Calcular nuevas coordenadas
                delta_lat = (distancia / 111) * np.cos(angulo)
                delta_lon = (distancia / (111 * np.cos(np.radians(lat_est)))) * np.sin(angulo)
                nueva_lat = lat_est + delta_lat
                nueva_lon = lon_est + delta_lon
                latitudes.append(round(nueva_lat, 6))
                longitudes.append(round(nueva_lon, 6))
                precio = max(20, np.random.normal(self.mean_precio, self.std_precio))
                precios.append(round(precio, 2))
                precios_todo_incluido.append(round(precio * 1.3, 2))  # Suponiendo que todo incluido es un 30% más caro
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
        :param filename: Nombre del archivo donde se guardarán los datos.
        :return:
        """
        if self.hoteles.empty:
            raise ValueError("No hay datos de hoteles para guardar. Genera los hoteles primero.")
        self.hoteles.to_csv(filename, index=False)
        print(f"Datos de hoteles guardados en {filename}")
        return

    def plot_hotels(self, lat_city=None, lon_city=None):
        """
        Grafica los hoteles generados en un mapa con los centros de la ciudad y del sistema de hoteles.
        :param lat_city: latitud del centro de la ciudad (opcional)
        :param lon_city: longitud del centro de la ciudad (opcional)
        """
        if self.hoteles.empty:
            raise ValueError("Primero debes generar los hoteles con hotel_generation_points()")

        # -----------------------------
        # Convertir hoteles a GeoDataFrame
        # -----------------------------
        gdf_hoteles = gpd.GeoDataFrame(
            self.hoteles,
            geometry=gpd.points_from_xy(self.hoteles['longitud'], self.hoteles['latitud']),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # Centro del sistema de hoteles
        centro_hoteles = gpd.GeoSeries([Point(self.city_center[1], self.city_center[0])], crs="EPSG:4326").to_crs(
            epsg=3857)

        # Centro de la ciudad si se proporciona
        if lat_city is not None and lon_city is not None:
            centro_ciudad = gpd.GeoSeries([Point(lon_city, lat_city)], crs="EPSG:4326").to_crs(epsg=3857)
        else:
            centro_ciudad = None

        # -----------------------------
        # Crear figura
        # -----------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dibujar hoteles
        gdf_hoteles.plot(ax=ax, color='green', markersize=150, alpha=0.7, label='Hoteles')

        # Dibujar centros
        centro_hoteles.plot(ax=ax, color='blue', markersize=50, label='Centro Hoteles', zorder=3)

        # -----------------------------
        # Ajustar límites
        # -----------------------------
        buffer = 5000  # metros alrededor del centro de los hoteles
        cx, cy = centro_hoteles.geometry[0].x, centro_hoteles.geometry[0].y
        ax.set_xlim(cx - buffer, cx + buffer)
        ax.set_ylim(cy - buffer, cy + buffer)

        # Añadir mapa base
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # -----------------------------
        # Configuración final
        # -----------------------------
        ax.set_title("Distribución de Hoteles en la Ciudad", fontsize=16, fontweight='bold')
        ax.legend()
        ax.set_axis_off()
        plt.tight_layout()
        #guardamos la imagen
        ruta_salida = os.path.join(os.path.dirname(__file__), "hoteles_mapa.png")
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

