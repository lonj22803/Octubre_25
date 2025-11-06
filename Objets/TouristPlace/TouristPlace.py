"""
TouristPlace class that inherits from CityBase for generalized cluster operations.
"""

from Objets.City.City import CityBase
import numpy as np
import pandas as pd


class TouristPlace(CityBase):
    """
    Clase para representar lugares turísticos en la ciudad.
    Hereda de CityBase para utilizar métodos generalizados de clustering.
    Esta clase recibe estaciones, hoteles y restaurantes para generar lugares turísticos.
    """

    def __init__(self, lat: float, lon: float, num_tourist_places: int):
        """
        Inicializa lugares turísticos.
        
        Args:
            lat: Centro en latitud
            lon: Centro en longitud
            num_tourist_places: Número de lugares turísticos a generar
        """
        self.city_center = (lat, lon)
        self.num_tourist_places = num_tourist_places
        self.tourist_places = pd.DataFrame()
        self.clusters_info = {}
        self.cluster_weights = {}

    def generate_tourist_places(self, stations, hotels, restaurants, 
                                station_coordinates, hotel_coordinates, 
                                restaurant_coordinates):
        """
        Genera lugares turísticos considerando estaciones, hoteles y restaurantes.
        Utiliza métodos heredados de CityBase para clustering.
        
        Args:
            stations: Lista de estaciones
            hotels: Lista de hoteles
            restaurants: Lista de restaurantes
            station_coordinates: Coordenadas de estaciones
            hotel_coordinates: Coordenadas de hoteles
            restaurant_coordinates: Coordenadas de restaurantes
            
        Returns:
            DataFrame con lugares turísticos generados
        """
        # Validar parámetros
        validated = self.validate_parameters(
            stations=stations,
            hotels=hotels,
            restaurants=restaurants,
            station_coordinates=station_coordinates
        )
        
        # Combinar todas las coordenadas para generar clusters
        all_coords = []
        all_coords.extend(list(station_coordinates.values()))
        all_coords.extend(hotel_coordinates)
        all_coords.extend(restaurant_coordinates)
        
        if len(all_coords) < 2:
            raise ValueError("Se necesitan al menos 2 coordenadas para generar lugares turísticos")
        
        # Generar clusters
        self.clusters_info = self.generate_clusters(
            all_coords,
            n_clusters=min(5, len(all_coords) // 10),
            method='kmeans'
        )
        
        # Calcular pesos con fuerte influencia del centro
        self.cluster_weights = self.calculate_cluster_weights(
            self.clusters_info,
            self.city_center,
            center_weight_multiplier=4.0  # Mayor influencia para lugares turísticos
        )
        
        # TODO: Implementar lógica específica de generación de lugares turísticos
        print("Generación de lugares turísticos - funcionalidad base implementada")
        
        return self.tourist_places
