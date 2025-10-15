from IPython.core.pylabtools import figsize

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

class TrainSystem:
    """
    Clase para representar un sistema de metro generico a partir del cual se contruye el benchmark o Testbed para
    evaluar la capacidad de un LLM para resolver problemas de rutas en sistemas de transporte urbano ademas de cruzar
    información con datos externos que se crean a partir de este
    """

    def __init__(self):
        self.system = {}
        self.lines = []
        self.stations = []
        self.station_coordinates = {}  # Diccionario para almacenar las coordenadas de las estaciones
        self.centroide_m = [0.0, 0.0]
        self.centroide_latlon = [0.0, 0.0]
        self.radio_km = 0.0
        # Coordenadas del centroide del sistema de metro (latitud, longitud)

    def load_system(self,system_file: dict):
        """
        Cargar sistema de metro desde un diccionario, como
        :param self:
        :param basic_system:
        return:
        """
        self.system = system_file
        #Las líneas son las llaves del diccionario de dentro del diccionario principal
        self.lines = list(system_file.keys())
        #Las estaciones son la union de todas las estaciones de todas las lineas
        estaciones = set()
        for linea in self.lines:
            estaciones.update(system_file[linea]["estaciones"])
        self.stations = list(estaciones)

        return

    def add_station(self, coordinates_cartesian_stations:dict, center_city_base_point:tuple, scale:float=0.0001):
        """
        Las coordenadas estan dada en cartesianas por lo tanto se deben convertir a latitud y longitud, con el fin
        de poder representar que un sistema de metro sea realista y el LLM lo pueda interpretar como un dato real.
        :param coordinates_cartesian_stations: Diccionario con las coordenadas cartesianas de las estaciones que contiene
        una tupla (latitud, longitud).
        :param center_city_base_point: Punto base de la ciudad (latitud, longitud) para centrar las coordenadas, en la
        ciudad deseada
        :param scale: Escala para convertir las coordenadas cartesianas a latitud y longitud, dada una transformación a fin
        de que las estaciones no queden muy juntas.
        Es 0.0001 por defecto, que equivale a 11.13 metros por cada 0.0001 en latitud y longitud.
        :return:
        """
        pos_geo = {}
        for key, (x, y) in coordinates_cartesian_stations.items():
            lat = round((center_city_base_point[0] + (y * scale)),10)
            lon = round((center_city_base_point[1] + (x * scale)),10)
            pos_geo[key] = (lat, lon)

        self.station_coordinates = pos_geo
        return pos_geo

    import math  # Asegúrate de importar math si no está en tu archivo

    def geographics_dates(self):
        """
        Calcula análisis geográficos de las estaciones, incluyendo:
        - Centroide (calculado en proyección plana EPSG:3857)
        - Radio del sistema (distancia máxima desde el centroide)
        - Estación más lejana al centroide
        - Estadísticas generales de distancias entre estaciones (en km)

        Requiere que se haya llamado a add_station primero.

        Returns:
            Un diccionario con los resultados del análisis.
        """

        global par_cercano, par_lejano, min_dist, max_dist  # variables globales opcionales

        if not self.station_coordinates:
            raise ValueError(
                "No se han añadido las coordenadas geográficas de las estaciones. Use el método add_station primero."
            )

        # ------------------------------
        # 1️⃣ Transformar coordenadas a proyección plana (EPSG:3857)
        # ------------------------------
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        coords_mercator = {
            est: transformer.transform(lon, lat)
            for est, (lat, lon) in self.station_coordinates.items()
        }

        # ------------------------------
        # 2️⃣ Calcular centroide en proyección plana (metros)
        # ------------------------------
        xs = [x for x, _ in coords_mercator.values()]
        ys = [y for _, y in coords_mercator.values()]
        centro_x = np.mean(xs)
        centro_y = np.mean(ys)

        # Convertir centroide nuevamente a coordenadas geográficas (EPSG:4326)
        transformer_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        centroide_lon, centroide_lat = transformer_inv.transform(centro_x, centro_y)

        # ------------------------------
        # 3️⃣ Calcular distancias entre estaciones (usando Haversine, en km)
        # ------------------------------
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

        # ------------------------------
        # 4️⃣ Estadísticas básicas de distancias
        # ------------------------------
        if distancias:
            distancia_promedio = sum(distancias) / len(distancias)
            distancia_maxima = max(distancias)
            distancia_minima = min(distancias)
            total_pares = len(distancias)

            # Pares más cercanos y más lejanos
            min_dist = min(pares_distancias.values())
            par_cercano = next(k for k, v in pares_distancias.items() if v == min_dist)
            max_dist = max(pares_distancias.values())
            par_lejano = next(k for k, v in pares_distancias.items() if v == max_dist)
        else:
            distancia_promedio = distancia_maxima = distancia_minima = total_pares = 0
            par_cercano = par_lejano = None
            min_dist = max_dist = 0

        # ------------------------------
        # 5️⃣ Calcular distancia al centroide (en proyección plana)
        # ------------------------------
        distancias_al_centro = {
            est: math.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)
            for est, (x, y) in coords_mercator.items()
        }

        # Punto más lejano y radio del sistema (convertido a km)
        estacion_mas_lejana = max(distancias_al_centro, key=distancias_al_centro.get)
        radio_sistema = distancias_al_centro[estacion_mas_lejana] / 1000.0

        # ------------------------------
        # 6️⃣ Guardar en la instancia
        # ------------------------------
        self.centroide_latlon = [centroide_lat, centroide_lon]  # Coordenadas geográficas reales
        self.centroide_m = [centro_x, centro_y]  # Coordenadas planas en metros
        self.radio_km = radio_sistema

        # ------------------------------
        # 7️⃣ Retornar resultados
        # ------------------------------
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

    def graphics_trian_city(self, color_map: dict, lat_city: float = 40.429800, lon_city: float = -3.705770):
        """
        Representa gráficamente el sistema de metro sobre un mapa de la ciudad.
        Permite visualizar el trazado de líneas, estaciones y su relación con el centro urbano.

        Parámetros:
            color_map (dict): Diccionario con los colores de cada línea del sistema.
            lat_city (float): Latitud del centro de la ciudad.
            lon_city (float): Longitud del centro de la ciudad.
        """

        # -----------------------------
        # Validaciones iniciales
        # -----------------------------
        if not color_map:
            raise ValueError("No se ha proporcionado un mapa de colores para las líneas del metro.")
        elif set(color_map.keys()) != set(self.lines):
            raise ValueError("Las líneas en el mapa de colores no coinciden con las líneas del sistema de metro.")

        # -----------------------------
        # Crear grafo y posiciones
        # -----------------------------
        G = nx.Graph()
        for color, datos_linea in self.system.items():
            estaciones = datos_linea["sentido_ida"]
            for i in range(len(estaciones) - 1):
                G.add_edge(estaciones[i], estaciones[i + 1], color=color_map[color])

        # Transformador de coordenadas
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Convertir coordenadas de estaciones
        pos_web = {
            estacion: transformer.transform(lon, lat)
            for estacion, (lat, lon) in self.station_coordinates.items()
        }

        # -----------------------------
        # Crear figura
        # -----------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dibujar el grafo
        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]
        nx.draw(
            G,
            pos_web,
            with_labels=True,
            node_size=650,
            node_color="white",
            edge_color=colors,
            width=4,
            font_size=6,
            font_weight="bold",
            ax=ax
        )

        # -----------------------------
        # Calcular centro y radio del sistema
        # -----------------------------
        centro_sistema = gpd.GeoSeries(
            [Point(self.centroide_latlon[1], self.centroide_latlon[0])],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        radio_metros = self.radio_km * 1000
        cx, cy = centro_sistema.geometry[0].x, centro_sistema.geometry[0].y

        # Ajustar límites para no cortar el mapa
        ax.set_xlim(cx - radio_metros * 1.2, cx + radio_metros * 1.2)
        ax.set_ylim(cy - radio_metros * 1.2, cy + radio_metros * 1.2)

        # -----------------------------
        # Añadir mapa base (después de fijar límites)
        # -----------------------------
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # -----------------------------
        # Dibujar centros y círculos
        # -----------------------------
        centro_ciudad = gpd.GeoSeries([Point(lon_city, lat_city)], crs="EPSG:4326").to_crs(epsg=3857)
        centro_ciudad.plot(ax=ax, color='red', markersize=100, zorder=4, label='Centro de la Ciudad')

        centro_sistema.plot(ax=ax, color='blue', markersize=100, zorder=4, label='Centro del Sistema de Metro')

        circulo_sistema = centro_sistema.buffer(radio_metros)
        circulo_sistema.plot(ax=ax, facecolor='none', edgecolor='blue', linestyle='--', linewidth=2, zorder=2)

        # -----------------------------
        # Configuración final
        # -----------------------------
        ax.set_title('Sistema de Transporte Genérico', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_axis_off()
        plt.tight_layout()

        # Guardar imagen
        ruta_salida = os.path.join(os.path.dirname(__file__), "sistema_generico_mapa.png")
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

    def guardar_sistema(self, filename: str):
        """
        Guarda el sistema de metro en un archivo JSON.
        :param filename: Nombre del archivo donde se guardará el sistema.
        :return:
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.system, f, indent=4)
        print(f"Sistema guardado en {filename}")
        return

    def guardar_estaciones_csv(self, filename: str):
        """
        Guarda las coordenadas de las estaciones en un archivo csv.
        :param filename: Nombre del archivo donde se guardarán las coordenadas.
        :return:
        """
        df = pd.DataFrame.from_dict(self.station_coordinates, orient='index', columns=['latitud', 'longitud'])
        df.index.name = 'estacion'
        df.to_csv(filename)
        print(f"Coordenadas de estaciones guardadas en {filename}")
        return








if __name__ == "__main__":
    sistema_basico_prueba = {
            "amarilla": {
                "estaciones": ["AA1SC", "AB2SC", "AC3SC", "AD4RF", "AE5VE", "AF6SC", "AG7BH"],
                "sentido_ida": ["AA1SC", "AB2SC", "AC3SC", "AD4RF", "AE5VE", "AF6SC", "AG7BH"],
                "sentido_vuelta": ["AG7BH", "AF6SC", "AE5VE", "AD4RF", "AC3SC", "AB2SC", "AA1SC"]
            },
            "azul": {
                "estaciones": ["BA1SC", "BB2OC", "BC3SC", "BD2VB", "BE4RC", "BF5SC", "BG6SC", "AG7BH"],
                "sentido_ida": ["BA1SC", "BB2OC", "BC3SC", "BD2VB", "BE4RC", "BF5SC", "BG6SC", "AG7BH"],
                "sentido_vuelta": ["AG7BH", "BG6SC", "BF5SC", "BE4RC", "BD2VB", "BC3SC", "BB2OC", "BA1SC"]
            },
            "roja": {
                "estaciones": ["RA1SC", "RB2SC", "BE4RC", "RD3VC", "RE5SC", "AD4RF", "RG6SC"],
                "sentido_ida": ["RA1SC", "RB2SC", "BE4RC", "RD3VC", "RE5SC", "AD4RF", "RG6SC"],
                "sentido_vuelta": ["RG6SC", "AD4RF", "RE5SC", "RD3VC", "BE4RC", "RB2SC", "RA1SC"]
            },
            "verde": {
                "estaciones": ["VA1SC", "BD2VB", "RD3VC", "VD4SC", "AE5VE", "VF6SC"],
                "sentido_ida": ["VA1SC", "BD2VB", "RD3VC", "VD4SC", "AE5VE", "VF6SC"],
                "sentido_vuelta": ["VF6SC", "AE5VE", "VD4SC", "RD3VC", "BD2VB", "VA1SC"]
            },
            "naranja": {
                "estaciones": ["OA1SC", "OB2SC", "BB2OC", "OC3SC", "RD3VC"],
                "sentido_ida": ["OA1SC", "OB2SC", "BB2OC", "OC3SC", "RD3VC"],
                "sentido_vuelta": ["RD3VC", "OC3SC", "BB2OC", "OB2SC", "OA1SC"]
            }

        }

    posiciones_estaciones = {
            "AA1SC": (-40, 20),
            "AB2SC": (50, -40),
            "AC3SC": (120, 0),
            "AD4RF": (180, 40),
            "AE5VE": (240, 60),
            "AF6SC": (320, 100),
            "AG7BH": (400, 120),
            "BA1SC": (-120, 180),
            "BB2OC": (-60, 200),
            "BC3SC": (0, 220),
            "BD2VB": (60, 240),
            "BE4RC": (120, 220),
            "BF5SC": (200, 180),
            "BG6SC": (290, 160),
            "RA1SC": (240, 260),
            "RB2SC": (180, 300),
            "RD3VC": (100, 160),
            "RE5SC": (120, 80),
            "RG6SC": (120, -120),
            "VA1SC": (0, 360),
            "VD4SC": (180, 120),
            "VF6SC": (260, 0),
            "OA1SC": (-160, 380),
            "OB2SC": (-120, 280),
            "OC3SC": (0, 80)
        }



    sistema_metro = TrainSystem()
    sistema_metro.load_system(sistema_basico_prueba)
    print("Líneas del sistema de metro de prueba:", sistema_metro.lines, "\nTotal son:", len(sistema_metro.lines), " líneas")
    print("Estaciones del sistema de metro de prueba:", sistema_metro.stations,"\nTotal son:", len(sistema_metro.stations), " estaciones")
    sistema_metro.add_station(posiciones_estaciones, (40.4168, -3.7038))
    print("Coordenadas geográficas de las estaciones:", sistema_metro.station_coordinates)
    analisis_geo = sistema_metro.geographics_dates()
    print("Centro del sistema de metro (latitud, longitud):", (analisis_geo['centroide_lat'], analisis_geo['centroide_lon']))
    print ("Centro del sistema de metro (metros):", sistema_metro.centroide_m)
    print("Centroide del sistema de metro (latitud, longitud):", sistema_metro.centroide_latlon)
    print("Radio del sistema de metro (km):", analisis_geo['radio_sistema'])
    color_map = {
        "amarilla": "yellow",
        "azul": "blue",
        "roja": "red",
        "verde": "green",
        "naranja": "orange"
    }
    sistema_metro.graphics_trian_city(color_map=color_map)
    sistema_metro.guardar_sistema("sistema_generico.json")




