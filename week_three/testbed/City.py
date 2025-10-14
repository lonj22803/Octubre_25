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

    def geographics_dates(self):
        """
        Si ya se tienen las coordenadas geográficas de las estaciones, se puede calcular el centroide del sistema
        de metro, radio del sistema, y la distancia entre estaciones, etc. esto es de interes para creart las demas
        instancias del benchmark
        """

        if self.station_coordinates == {}:
            raise ValueError("No se han añadido las coordenadas geográficas de las estaciones. Use el método add_station primero, para añadir las estaciones primero.")
        # Calcular el centroide del sistema de metro
        lats = [lat for lat, lon in self.station_coordinates.values()]
        lons = [lon for lat, lon in self.station_coordinates.values()]
        self.centroide = [round(sum(lats) / len(lats),6), round(sum(lons) / len(lons),6)]

        distancias=[]
        #Calculamos la distancia entre todas las estaciones:
        for i in range(len(self.stations)):
            for j in range(i + 1, len(self.stations)):
                est1 = self.stations[i]
                est2 = self.stations[j]
                lat1,lon1 = self.station_coordinates[est1]
                lat2,lon2 = self.station_coordinates[est2]
                distancia = calcular_distancia_haversine(lat1, lon1, lat2, lon2)
                distancias.append(distancias)
                






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

