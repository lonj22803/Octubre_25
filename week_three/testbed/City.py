from funciones_propias import *
import json
import  networkx as nx


class TrainSystem():
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
        self.centroide = [0.0, 0.0]
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
            lat = round((center_city_base_point[0] + (y * scale)),6)
            lon = round((center_city_base_point[1] + (x * scale)),6)
            pos_geo[key] = (lat, lon)

        self.station_coordinates = pos_geo
        return pos_geo

    import math  # Asegúrate de importar math si no está en tu archivo

    def geographics_dates(self):
        """
        Calcula análisis geográficos de las estaciones, incluyendo centroide, distancias y estadísticas.
        Requiere que se haya llamado a add_station primero.

        Returns:
            Un diccionario con los resultados del análisis.
        """
        global par_cercano, par_lejano, min_dist, max_dist  # Variables globales para almacenar los pares y distancias
        if not self.station_coordinates:  # Verifica si el diccionario está vacío
            raise ValueError(
                "No se han añadido las coordenadas geográficas de las estaciones. Use el método add_station primero.")

        # Calcular el centroide
        lats = [lat for lat, lon in self.station_coordinates.values()]
        lons = [lon for lat, lon in self.station_coordinates.values()]
        centroide_lat = sum(lats) / len(lats)
        centroide_lon = sum(lons) / len(lons)

        estaciones = list(self.station_coordinates.keys())  # Lista de nombres de estaciones
        distancias = []  # Lista para almacenar todas las distancias
        pares_distancias = {}  # Diccionario para pares y sus distancias (para reutilizar)

        for i in range(len(estaciones)):
            for j in range(i + 1, len(estaciones)):
                est1 = estaciones[i]
                est2 = estaciones[j]
                lat1, lon1 = self.station_coordinates[est1]
                lat2, lon2 = self.station_coordinates[est2]
                dist = calcular_distancia_haversine(lat1, lon1, lat2, lon2)
                distancias.append(dist)
                pares_distancias[(est1, est2)] = dist  # Almacena la distancia para el par

        if distancias:  # Solo calcula estadísticas si hay distancias
            distancia_promedio = sum(distancias) / len(distancias)
            distancia_maxima = max(distancias)
            distancia_minima = min(distancias)
            total_pares = len(distancias)

            # Encontrar par más cercano y más lejano usando el diccionario
            if pares_distancias:
                min_dist = min(pares_distancias.values())
                par_cercano = next(key for key, value in pares_distancias.items() if
                                   value == min_dist)  # Primer par con distancia mínima

                max_dist = max(pares_distancias.values())
                par_lejano = next(key for key, value in pares_distancias.items() if
                                  value == max_dist)  # Primer par con distancia máxima

            # Calcular radio del sistema (distancia máxima desde el centroide)
            distancias_desde_centro = [
                calcular_distancia_haversine(centroide_lat, centroide_lon, lat, lon)
                for lat, lon in self.station_coordinates.values()
            ]
            radio_sistema = max(distancias_desde_centro) if distancias_desde_centro else 0

            self.centroide=[round(centroide_lat,6),round(centroide_lon,6)]

            return {
                'centroide_lat': round(centroide_lat, 6),
                'centroide_lon': round(centroide_lon, 6),
                'radio_sistema': round(radio_sistema, 2),  # En km
                'distancia_promedio': round(distancia_promedio, 2),
                'distancia_maxima': round(distancia_maxima, 2),
                'distancia_minima': round(distancia_minima, 2),
                'total_pares': total_pares,
                'par_cercano': par_cercano,  # Tupla (est1, est2)
                'min_dist': round(min_dist, 2),
                'par_lejano': par_lejano,  # Tupla (est1, est2)
                'max_dist': round(max_dist, 2)
            }
        else:
            return {}  # Retorna un diccionario vacío si no hay pares (pocas estaciones)


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
    #sistema_metro.add_station(posiciones_estaciones, (40.4168, -3.7038))
    #print("Coordenadas geográficas de las estaciones:", sistema_metro.station_coordinates)
    analisis_geo = sistema_metro.geographics_dates()
    #print("Centro del sistema de metro (latitud, longitud):", (analisis_geo['centroide_lat'], analisis_geo['centroide_lon']))
    #print ("Centro del sistema de metro (latitud, longitud):", sistema_metro.centroide)
    #print("Radio del sistema de metro (km):", analisis_geo['radio_sistema'])



