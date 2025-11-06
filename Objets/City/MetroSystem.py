"""
MetroSystem class that inherits from CityBase for generalized cluster operations.
"""

from Objets.City.City import CityBase
import os
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from pyproj import Transformer
import numpy as np
import pandas as pd
import math


def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos en la Tierra usando la fórmula de Haversine.
    
    Args:
        lat1, lon1: Coordenadas del primer punto
        lat2, lon2: Coordenadas del segundo punto
        
    Returns:
        Distancia en kilómetros
    """
    R = 6371  # Radio de la Tierra en km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


class MetroSystem(CityBase):
    """
    Clase para representar un sistema de metro genérico que hereda de CityBase
    para utilizar métodos generalizados de clustering y gestión de pesos.
    """

    def __init__(self):
        self.system = {}
        self.lines = []
        self.stations = []
        self.station_coordinates = {}
        self.centroide_m = [0.0, 0.0]
        self.centroide_latlon = [0.0, 0.0]
        self.radio_km = 0.0

    def load_system(self, system_file: dict):
        """
        Cargar sistema de metro desde un diccionario.
        
        Args:
            system_file: Diccionario con la estructura del sistema de metro
        """
        self.system = system_file
        self.lines = list(system_file.keys())
        
        estaciones = set()
        for linea in self.lines:
            estaciones.update(system_file[linea]["estaciones"])
        self.stations = list(estaciones)

    def add_station(self, coordinates_cartesian_stations: dict, 
                   center_city_base_point: tuple, 
                   scale: float = 0.0001):
        """
        Convierte coordenadas cartesianas a geográficas.
        
        Args:
            coordinates_cartesian_stations: Diccionario con coordenadas cartesianas
            center_city_base_point: Punto base (lat, lon) de la ciudad
            scale: Escala de conversión
            
        Returns:
            Diccionario con coordenadas geográficas
        """
        pos_geo = {}
        for key, (x, y) in coordinates_cartesian_stations.items():
            lat = round((center_city_base_point[0] + (y * scale)), 10)
            lon = round((center_city_base_point[1] + (x * scale)), 10)
            pos_geo[key] = (lat, lon)

        self.station_coordinates = pos_geo
        return pos_geo

    def geographics_dates(self):
        """
        Calcula análisis geográficos de las estaciones.
        
        Returns:
            Diccionario con resultados del análisis geográfico
        """
        if not self.station_coordinates:
            raise ValueError(
                "No se han añadido las coordenadas geográficas de las estaciones. "
                "Use el método add_station primero."
            )

        # Transformar coordenadas a proyección plana
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        coords_mercator = {
            est: transformer.transform(lon, lat)
            for est, (lat, lon) in self.station_coordinates.items()
        }

        # Calcular centroide en proyección plana
        xs = [x for x, _ in coords_mercator.values()]
        ys = [y for _, y in coords_mercator.values()]
        centro_x = np.mean(xs)
        centro_y = np.mean(ys)

        # Convertir centroide a coordenadas geográficas
        transformer_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        centroide_lon, centroide_lat = transformer_inv.transform(centro_x, centro_y)

        # Calcular distancias entre estaciones
        estaciones = list(self.station_coordinates.keys())
        distancias = []
        pares_distancias = {}

        for i in range(len(estaciones)):
            for j in range(i + 1, len(estaciones)):
                est1, est2 = estaciones[i], estaciones[j]
                lat1, lon1 = self.station_coordinates[est1]
                lat2, lon2 = self.station_coordinates[est2]
                dist = calcular_distancia_haversine(lat1, lon1, lat2, lon2)
                distancias.append(dist)
                pares_distancias[(est1, est2)] = dist

        # Estadísticas básicas
        if distancias:
            distancia_promedio = sum(distancias) / len(distancias)
            distancia_maxima = max(distancias)
            distancia_minima = min(distancias)
            total_pares = len(distancias)

            min_dist = min(pares_distancias.values())
            par_cercano = next(k for k, v in pares_distancias.items() if v == min_dist)
            max_dist = max(pares_distancias.values())
            par_lejano = next(k for k, v in pares_distancias.items() if v == max_dist)
        else:
            distancia_promedio = distancia_maxima = distancia_minima = total_pares = 0
            par_cercano = par_lejano = None
            min_dist = max_dist = 0

        # Calcular distancia al centroide
        distancias_al_centro = {
            est: math.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)
            for est, (x, y) in coords_mercator.items()
        }

        estacion_mas_lejana = max(distancias_al_centro, key=distancias_al_centro.get)
        radio_sistema = distancias_al_centro[estacion_mas_lejana] / 1000.0

        # Guardar en la instancia
        self.centroide_latlon = [centroide_lat, centroide_lon]
        self.centroide_m = [centro_x, centro_y]
        self.radio_km = radio_sistema

        return {
            'centroide_lat': round(centroide_lat, 6),
            'centroide_lon': round(centroide_lon, 6),
            'radio_sistema': round(radio_sistema, 2),
            'estacion_mas_lejana': estacion_mas_lejana,
            'distancia_promedio': round(distancia_promedio, 2),
            'distancia_maxima': round(distancia_maxima, 2),
            'distancia_minima': round(distancia_minima, 2),
            'total_pares': total_pares,
            'par_cercano': par_cercano,
            'min_dist': round(min_dist, 2),
            'par_lejano': par_lejano,
            'max_dist': round(max_dist, 2)
        }

    def graphics_trian_city(self, color_map: dict, lat_city: float = 40.4168, 
                           lon_city: float = -3.7038):
        """
        Representa gráficamente el sistema de metro sobre un mapa.
        
        Args:
            color_map: Diccionario con colores para cada línea
            lat_city: Latitud del centro de la ciudad
            lon_city: Longitud del centro de la ciudad
        """
        if not color_map:
            raise ValueError("No se ha proporcionado un mapa de colores para las líneas del metro.")
        elif set(color_map.keys()) != set(self.lines):
            raise ValueError("Las líneas en el mapa de colores no coinciden con las líneas del sistema.")

        # Crear grafo
        G = nx.Graph()
        for color, datos_linea in self.system.items():
            estaciones = datos_linea["sentido_ida"]
            for i in range(len(estaciones) - 1):
                G.add_edge(estaciones[i], estaciones[i + 1], color=color_map[color])

        # Transformar coordenadas
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        pos_web = {
            estacion: transformer.transform(lon, lat)
            for estacion, (lat, lon) in self.station_coordinates.items()
        }

        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dibujar grafo
        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]
        nx.draw(
            G, pos_web, with_labels=True, node_size=650, node_color="white",
            edge_color=colors, width=4, font_size=6, font_weight="bold", ax=ax
        )

        # Calcular centro y radio
        centro_sistema = gpd.GeoSeries(
            [Point(self.centroide_latlon[1], self.centroide_latlon[0])],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        radio_metros = self.radio_km * 1000
        cx, cy = centro_sistema.geometry[0].x, centro_sistema.geometry[0].y

        ax.set_xlim(cx - radio_metros * 1.2, cx + radio_metros * 1.2)
        ax.set_ylim(cy - radio_metros * 1.2, cy + radio_metros * 1.2)

        # Añadir mapa base
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Dibujar centros
        centro_sistema.plot(ax=ax, color='blue', markersize=100, zorder=4, 
                          label='Centro del Sistema de Metro')

        circulo_sistema = centro_sistema.buffer(radio_metros)
        circulo_sistema.plot(ax=ax, facecolor='none', edgecolor='blue', 
                           linestyle='--', linewidth=2, zorder=2)

        ax.set_title('Sistema de Transporte Genérico', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_axis_off()
        plt.tight_layout()

        ruta_salida = os.path.join(os.path.dirname(__file__), "sistema_generico_mapa.png")
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def guardar_sistema(self, filename: str):
        """
        Guarda el sistema de metro en un archivo JSON.
        
        Args:
            filename: Nombre del archivo de salida
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.system, f, indent=4)
        print(f"Sistema guardado en {filename}")

    def guardar_estaciones_csv(self, filename: str):
        """
        Guarda las coordenadas de las estaciones en un CSV.
        
        Args:
            filename: Nombre del archivo de salida
        """
        df = pd.DataFrame.from_dict(
            self.station_coordinates, 
            orient='index', 
            columns=['latitud', 'longitud']
        )
        df.index.name = 'estacion'
        df.to_csv(filename)
        print(f"Coordenadas de estaciones guardadas en {filename}")
