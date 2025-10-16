import pandas as pd
import numpy as np
import random
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import os
from collections import Counter  # Para contar frecuencias


class Restaurantes:
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