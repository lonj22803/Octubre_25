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

class Airport:
    """
    Clase para representar un aeropuerto en la ciudad, que hereda del sistema de metro.
    El centro en latitud y longitud del aeropuerto se utiliza para centrar el sistema de metro.
    Hereda de la clase TrainSystem.
    El centro en longitud y latitud para generar de manera automatica los aeropuertos con una
    distribucion de probabilidad exponbencial respecto al centro de la ciudad.
    """
    def __init__(self, lat: float=None, lon: float=None):
        """
        Inicializa un aeropuerto con su ubicación y parámetros de distribución.
        :param lat: centro en latitud
        :param lon: centro en longitud
        """
        self.airport_center = [lat, lon]

    def adq_ubication(self, station_coordinates: dict, centroide_latlon: tuple):
        """
        Adquiere la ubicación del aeropuerto a partir de las estaciones del sistema de metro
        partiendo del centroide del sistema de metro.
        A partir de la distancia minima entre estaciones y el centroide del sistema de metro
        busca el punto mas lejano al centroide y lo utiliza para ubicar el aeropuerto.
        """
        if not station_coordinates:
            raise ValueError("No se han añadido las coordenadas geográficas de las estaciones. Use el método add_station primero.")
        if not centroide_latlon:
            raise ValueError("No se ha calculado el centroide del sistema de metro. Use el método geographics_dates primero.")

        min_distancia = float('inf')
        estacion_mas_cercana = None

        for estacion, (lat, lon) in station_coordinates.items():
            distancia = calcular_distancia_haversine(lat, lon, centroide_latlon[0], centroide_latlon[1])
            if distancia < min_distancia:
                min_distancia = distancia
                estacion_mas_cercana = (lat, lon)

        if estacion_mas_cercana is None:
            raise ValueError("No se pudo determinar la estación más cercana al centroide.")

        # Ahora buscamos la estación más lejana al centroide
        max_distancia = 0
        estacion_mas_lejana = None

        for estacion, (lat, lon) in station_coordinates.items():
            distancia = calcular_distancia_haversine(lat, lon, centroide_latlon[0], centroide_latlon[1])
            if distancia > max_distancia:
                max_distancia = distancia
                estacion_mas_lejana = (lat, lon)

        if estacion_mas_lejana is None:
            raise ValueError("No se pudo determinar la estación más lejana al centroide.")

        self.airport_center = estacion_mas_lejana
        return self.airport_center

    def creator_itinerary(self, cities_to:list, cities_from:list, airline_code:list, num_flights:int):
        """
        Crea un itinerario de vuelos entre ciudades.
        :param cities_to: Lista de ciudades de destino.
        :param cities_from: Lista de ciudades de origen.
        :return: Lista de tuplas representando los itinerarios (origen, destino).
        """
        if not cities_to or not cities_from:
            raise ValueError("Las listas de ciudades no pueden estar vacías.")

        """
        Generamos, nombres de vuelod de la forma AA### donde:
        AA son las letras codido de la aerolinea
        ### son numeros aleatorios entre 100 y 999
        """

    def plot_airport(self, lat_city=None, lon_city=None):
        """
        Grafica el centro del aeropuerto en un mapa con el centro de la ciudad si se proporciona.
        :param lat_city: latitud del centro de la ciudad (opcional)
        :param lon_city: longitud del centro de la ciudad (opcional)
        """
        if self.airport_center is None:
            raise ValueError("El centro del aeropuerto no ha sido definido. Usa el método adq_ubication primero.")

        # Crear GeoSeries para el aeropuerto
        punto_aeropuerto = gpd.GeoSeries([Point(self.airport_center[1], self.airport_center[0])], crs="EPSG:4326")
        gdf_aeropuerto = gpd.GeoDataFrame(geometry=punto_aeropuerto).to_crs(epsg=3857)

        # Centro de la ciudad si se proporciona
        if lat_city is not None and lon_city is not None:
            centro_ciudad = gpd.GeoSeries([Point(lon_city, lat_city)], crs="EPSG:4326").to_crs(epsg=3857)
        else:
            centro_ciudad = None

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dibujar el aeropuerto
        gdf_aeropuerto.plot(ax=ax, color='red', markersize=200, alpha=0.7, label='Aeropuerto')

        # Dibujar el centro de la ciudad si se proporciona
        if centro_ciudad is not None:
            centro_ciudad.plot(ax=ax, color='blue', markersize=50, label='Centro Ciudad')

        # Ajustar límites alrededor del centro del aeropuerto
        cx, cy = gdf_aeropuerto.geometry[0].x, gdf_aeropuerto.geometry[0].y
        buffer = 5000  # metros alrededor del centro del aeropuerto
        ax.set_xlim(cx - buffer, cx + buffer)
        ax.set_ylim(cy - buffer, cy + buffer)

        # Añadir mapa base
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Configuración final
        ax.set_title("Ubicación del Aeropuerto", fontsize=16, fontweight='bold')
        ax.legend()
        ax.set_axis_off()
        plt.tight_layout()

        # Guardar la imagen
        ruta_salida = os.path.join(os.path.dirname(__file__), "aeropuerto_mapa.png")
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
