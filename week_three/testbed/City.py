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
from collections import Counter

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

    def graphics_trian_city(self, color_map: dict, lat_city: float = 40.4168, lon_city: float = 3.7038):
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

 # Para contar frecuencias

import pandas as pd
import numpy as np
import random
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import os
from collections import Counter  # Para contar frecuencias


class Restaurantes(TrainSystem):  # Hereda de TrainSystem
    """
    Clase para representar restaurantes en la ciudad, que hereda del sistema de metro.
    El centro en latitud y longitud se utiliza para centrar el sistema.
    Hereda de la clase TrainSystem.
    Genera restaurantes alrededor de estaciones con cruces, priorizando aquellas con más líneas.
    """

    def __init__(self, lat: float, lon: float, num_restaurantes: int, max_distancia_km: float, mean_distancia_km: float,
                 lambda_poisson: float, alpha: float, beta: float, mean_tiempo: float, std_tiempo: float):
        """
        Inicializa los restaurantes con su ubicación y parámetros de distribución.
        :param lat: Centro en latitud.
        :param lon: Centro en longitud.
        :param num_restaurantes: Número de restaurantes a generar.
        :param max_distancia_km: Distancia máxima desde el centro en km.
        :param mean_distancia_km: Media de la distribución exponencial en km.
        :param lambda_poisson: Parámetro lambda para la distribución Poisson (precio promedio).
        :param alpha: Parámetro alpha para la distribución Beta (calificación).
        :param beta: Parámetro beta para la distribución Beta (calificación).
        :param mean_tiempo: Media para la distribución normal del tiempo promedio (en minutos).
        :param std_tiempo: Desviación estándar para la distribución normal del tiempo promedio.
        """
        self.city_center = (lat, lon)
        self.num_restaurantes = num_restaurantes
        self.max_distancia_km = max_distancia_km
        self.mean_distancia_km = mean_distancia_km
        self.lambda_poisson = lambda_poisson  # Para precios
        self.alpha = alpha  # Para calificación
        self.beta = beta  # Para calificación
        self.mean_tiempo = mean_tiempo  # Para tiempo promedio
        self.std_tiempo = std_tiempo  # Para tiempo promedio
        self.restaurantes = pd.DataFrame()  # DataFrame para almacenar los restaurantes

    def restaurant_generation_points(self, system, stations, station_coordinates):
        """
        Genera puntos de restaurantes alrededor de las estaciones de metro con cruces.
        Prioriza estaciones con más líneas: aquellas con mayor repetición tendrán más restaurantes.
        Crea una distribución proporcional basada en la frecuencia de las estaciones.
        :return: DataFrame con las coordenadas, precios, calificaciones, tipos de comida y tiempo promedio de los restaurantes.
        """
        nombre_restaurantes = [f"RST{i + 1}" for i in range(self.num_restaurantes)]
        latitudes = []
        longitudes = []
        precios_promedio = []
        calificaciones = []
        tipos_comida = []
        tiempos_promedio = []  # Nueva lista para tiempo promedio

        # Identificar estaciones y contar frecuencias
        stations_all_lines = []
        for line, data in system.items():
            stations_all_lines.extend(data["estaciones"])

        freq = Counter(stations_all_lines)  # Contador de frecuencias
        repeat_stations = [station for station in freq if freq[station] > 1]  # Solo estaciones con >1 línea

        if not repeat_stations:
            raise ValueError("No se encontraron estaciones con más de una línea de metro.")

        # Ordenar estaciones por frecuencia descendente (prioridad)
        repeat_stations_sorted = sorted(repeat_stations, key=lambda x: freq[x], reverse=True)
        weights = [freq[station] for station in repeat_stations_sorted]  # Pesos basados en frecuencia

        # Distribuir el número de restaurantes proporcionalmente
        num_rest_por_estacion = np.random.multinomial(self.num_restaurantes, pvals=[w / sum(weights) for w in weights])

        coords_repeat_stations = [station_coordinates[station] for station in repeat_stations_sorted if
                                  station in station_coordinates]

        print("Estaciones con cruces (ordenadas por frecuencia):", repeat_stations_sorted)
        print("Número de restaurantes por estación:", num_rest_por_estacion)

        for i, (lat_est, lon_est) in enumerate(coords_repeat_stations):
            for _ in range(num_rest_por_estacion[i]):  # Asigna según la distribución
                if len(latitudes) >= self.num_restaurantes:
                    break

                # Generar distancia y ángulo aleatorio (distribución exponencial)
                distancia = np.random.exponential(self.mean_distancia_km)
                while distancia > self.max_distancia_km:
                    distancia = np.random.exponential(self.mean_distancia_km)
                angulo = random.uniform(0, 2 * np.pi)

                # Calcular nuevas coordenadas
                delta_lat = (distancia / 111) * np.cos(angulo)  # 111 km por grado de latitud
                delta_lon = (distancia / (111 * np.cos(np.radians(lat_est)))) * np.sin(angulo)
                nueva_lat = lat_est + delta_lat
                nueva_lon = lon_est + delta_lon

                latitudes.append(round(nueva_lat, 6))
                longitudes.append(round(nueva_lon, 6))

                # Generar atributos
                precio_promedio = np.random.poisson(self.lambda_poisson)  # Distribución Poisson
                precios_promedio.append(max(5, precio_promedio))  # Asegurar un mínimo realista
                calificacion = np.random.beta(self.alpha, self.beta) * 4 + 1  # Escalar a 1-5
                calificaciones.append(round(calificacion, 1))
                tipo_comida = np.random.choice(
                    ['comida típica', 'comida rápida', 'comida extranjera'])  # Distribución uniforme
                tipos_comida.append(tipo_comida)
                tiempo_promedio = np.random.normal(self.mean_tiempo, self.std_tiempo)  # Distribución normal
                tiempos_promedio.append(max(0, round(tiempo_promedio, 1)))  # Asegurar no negativo y redondear

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
        Guarda los datos de los restaurantes en un archivo CSV.
        :param filename: Nombre del archivo donde se guardarán los datos.
        """
        if self.restaurantes.empty:
            raise ValueError("No hay datos de restaurantes para guardar. Genera los restaurantes primero.")
        self.restaurantes.to_csv(filename, index=False)
        print(f"Datos de restaurantes guardados en {filename}")

    def plot_restaurantes(self, lat_city=None, lon_city=None):
        """
        Grafica los restaurantes generados en un mapa con los centros de la ciudad y del sistema de restaurantes.
        :param lat_city: Latitud del centro de la ciudad (opcional).
        :param lon_city: Longitud del centro de la ciudad (opcional).
        """
        if self.restaurantes.empty:
            raise ValueError("Primero debes generar los restaurantes con restaurant_generation_points()")

        gdf_restaurantes = gpd.GeoDataFrame(
            self.restaurantes,
            geometry=gpd.points_from_xy(self.restaurantes['longitud'], self.restaurantes['latitud']),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        centro_restaurantes = gpd.GeoSeries([Point(self.city_center[1], self.city_center[0])], crs="EPSG:4326").to_crs(
            epsg=3857)

        if lat_city is not None and lon_city is not None:
            centro_ciudad = gpd.GeoSeries([Point(lon_city, lat_city)], crs="EPSG:4326").to_crs(epsg=3857)
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

    hoteles = Hotel(lat=sistema_metro.centroide_latlon[0], lon=sistema_metro.centroide_latlon[1], num_hoteles=60, max_distancia_km=1, mean_distancia_km=0.5, mean_precio=100, std_precio=20)
    df_hoteles = hoteles.hotel_generation_points(system=sistema_metro.system, stations=sistema_metro.stations, station_coordinates=sistema_metro.station_coordinates)
    print("Hoteles generados:\n", df_hoteles)
    hoteles.save_hotels_csv("hoteles_generados.csv")

    hoteles.plot_hotels(lat_city=40.4168, lon_city=-3.7038)

    restaurantes = Restaurantes(lat=sistema_metro.centroide_latlon[0], lon=sistema_metro.centroide_latlon[1], num_restaurantes=150, max_distancia_km=0.5,
                                mean_distancia_km=1, lambda_poisson=15, alpha=4, beta=2, mean_tiempo=30, std_tiempo=5)
    restaurantes.restaurant_generation_points(system=sistema_metro.system, stations=sistema_metro.stations, station_coordinates=sistema_metro.station_coordinates)
    print("Restaurantes generados:\n", restaurantes.restaurantes)
    restaurantes.save_restaurantes_csv("restaurantes.csv")
    restaurantes.plot_restaurantes(lat_city=40.4168, lon_city=-3.7038)






