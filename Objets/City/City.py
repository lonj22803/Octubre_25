"""
Base class with static methods for cluster generation and management.
This module provides generalized functionality for clustering, weight calculation,
and parameter validation that can be used by MetroSystem, Hotels, Restaurant, and TouristPlace classes.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from scipy.stats import multivariate_normal
import warnings


class CityBase:
    """
    Base class with static methods for cluster generation and management operations.
    All methods are static to allow inheritance and use without instantiation.
    """

    @staticmethod
    def validate_parameters(**kwargs) -> Dict[str, Any]:
        """
        Validates input parameters for different entity types (stations, hotels, restaurants, tourist zones).
        
        Args:
            **kwargs: Variable keyword arguments that can include:
                - stations: list of station objects or coordinates
                - hotels: list of hotel objects or coordinates
                - restaurants: list of restaurant objects or coordinates
                - tourist_zones: list of tourist zone objects or coordinates
                - system: system dictionary
                - station_coordinates: dictionary of station coordinates
                
        Returns:
            Dict with validated parameters
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        validated = {}
        
        # Validate stations if provided
        if 'stations' in kwargs:
            stations = kwargs['stations']
            if stations is None or (isinstance(stations, list) and len(stations) == 0):
                raise ValueError("Stations parameter is empty or None")
            validated['stations'] = stations
            
        # Validate hotels if provided
        if 'hotels' in kwargs:
            hotels = kwargs['hotels']
            if hotels is None or (isinstance(hotels, list) and len(hotels) == 0):
                raise ValueError("Hotels parameter is empty or None")
            validated['hotels'] = hotels
            
        # Validate restaurants if provided
        if 'restaurants' in kwargs:
            restaurants = kwargs['restaurants']
            if restaurants is None or (isinstance(restaurants, list) and len(restaurants) == 0):
                raise ValueError("Restaurants parameter is empty or None")
            validated['restaurants'] = restaurants
            
        # Validate tourist zones if provided
        if 'tourist_zones' in kwargs:
            tourist_zones = kwargs['tourist_zones']
            if tourist_zones is None:
                raise ValueError("Tourist zones parameter is None")
            validated['tourist_zones'] = tourist_zones
            
        # Validate system if provided
        if 'system' in kwargs:
            system = kwargs['system']
            if not isinstance(system, dict) or len(system) == 0:
                raise ValueError("System must be a non-empty dictionary")
            validated['system'] = system
            
        # Validate station_coordinates if provided
        if 'station_coordinates' in kwargs:
            coords = kwargs['station_coordinates']
            if not isinstance(coords, dict) or len(coords) == 0:
                raise ValueError("Station coordinates must be a non-empty dictionary")
            validated['station_coordinates'] = coords
            
        return validated

    @staticmethod
    def identify_repeat_stations(system: Dict, min_lines: int = 2) -> Tuple[List[str], Counter]:
        """
        Identifies stations that appear in multiple lines (junction stations).
        
        Args:
            system: Dictionary containing metro system information
            min_lines: Minimum number of lines for a station to be considered (default: 2)
            
        Returns:
            Tuple of (list of repeat stations, Counter object with frequencies)
        """
        stations_all_lines = []
        for line, data in system.items():
            stations_all_lines.extend(data["estaciones"])
        
        freq = Counter(stations_all_lines)
        repeat_stations = [station for station in freq if freq[station] >= min_lines]
        
        if not repeat_stations:
            raise ValueError(f"No stations found with at least {min_lines} lines")
        
        return repeat_stations, freq

    @staticmethod
    def generate_clusters(coordinates: List[Tuple[float, float]], 
                         n_clusters: Optional[int] = None,
                         method: str = 'kmeans') -> Dict[int, Dict[str, Any]]:
        """
        Generates clusters from a list of coordinates and stores cluster information.
        
        Args:
            coordinates: List of (lat, lon) tuples
            n_clusters: Number of clusters to generate. If None, uses heuristic.
            method: Clustering method ('kmeans', 'dbscan', etc.)
            
        Returns:
            Dictionary with cluster information:
            {
                cluster_id: {
                    'mean': (lat, lon),
                    'cov': 2x2 covariance matrix,
                    'weight': cluster weight,
                    'size': number of points in cluster,
                    'pdf_values': probability density function values
                }
            }
        """
        if len(coordinates) < 2:
            raise ValueError("Need at least 2 coordinates to generate clusters")
        
        # Convert to numpy array
        coords_array = np.array(coordinates)
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = max(2, min(5, len(coordinates) // 10))
        
        # Perform clustering based on method
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords_array)
            
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            
            dbscan = DBSCAN(eps=0.01, min_samples=2)
            labels = dbscan.fit_predict(coords_array)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Generate cluster information
        clusters_info = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = coords_array[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            # Calculate cluster statistics
            mean = np.mean(cluster_points, axis=0)
            
            # Calculate covariance with regularization to avoid singular matrices
            if len(cluster_points) > 1:
                cov = np.cov(cluster_points.T)
                # Add small regularization to diagonal
                cov += np.eye(2) * 1e-6
            else:
                # For single point clusters, use small isotropic covariance
                cov = np.eye(2) * 1e-4
            
            # Calculate weight based on cluster size
            weight = len(cluster_points) / len(coordinates)
            
            # Store cluster information
            clusters_info[cluster_id] = {
                'mean': tuple(mean),
                'cov': cov,
                'weight': weight,
                'size': len(cluster_points),
                'points': cluster_points.tolist()
            }
        
        return clusters_info

    @staticmethod
    def calculate_cluster_weights(clusters_info: Dict[int, Dict[str, Any]], 
                                  center_coords: Tuple[float, float],
                                  center_weight_multiplier: float = 2.0) -> Dict[int, float]:
        """
        Calculates weights for clusters considering distance to center.
        The center receives a higher weight multiplier to increase its influence.
        
        Args:
            clusters_info: Dictionary with cluster information
            center_coords: (lat, lon) of the city center
            center_weight_multiplier: Multiplier for center weight to increase influence (default: 2.0)
            
        Returns:
            Dictionary mapping cluster_id to adjusted weight
        """
        weights = {}
        center_array = np.array(center_coords)
        
        # Calculate distances from center to each cluster
        distances = {}
        for cluster_id, info in clusters_info.items():
            cluster_mean = np.array(info['mean'])
            distance = np.linalg.norm(cluster_mean - center_array)
            distances[cluster_id] = distance
        
        # Find the cluster closest to center
        if distances:
            center_cluster_id = min(distances, key=distances.get)
        else:
            center_cluster_id = None
        
        # Calculate weights with center boost
        for cluster_id, info in clusters_info.items():
            base_weight = info['weight']
            
            # Apply multiplier to center cluster
            if cluster_id == center_cluster_id:
                weights[cluster_id] = base_weight * center_weight_multiplier
            else:
                weights[cluster_id] = base_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights

    @staticmethod
    def assign_weights_to_entities(entities: List[Any],
                                   entity_coords: List[Tuple[float, float]],
                                   clusters_info: Dict[int, Dict[str, Any]],
                                   cluster_weights: Dict[int, float]) -> List[float]:
        """
        Assigns weights to individual entities (stations, hotels, restaurants) 
        based on their cluster membership and cluster weights.
        
        Args:
            entities: List of entity objects or identifiers
            entity_coords: List of (lat, lon) coordinates for each entity
            clusters_info: Dictionary with cluster information
            cluster_weights: Dictionary with cluster weights
            
        Returns:
            List of weights corresponding to each entity
        """
        if len(entities) != len(entity_coords):
            raise ValueError("Number of entities must match number of coordinates")
        
        entity_weights = []
        
        # Convert coordinates to array
        coords_array = np.array(entity_coords)
        
        # Assign each entity to nearest cluster
        for coord in coords_array:
            min_distance = float('inf')
            assigned_cluster = 0
            
            for cluster_id, info in clusters_info.items():
                cluster_mean = np.array(info['mean'])
                distance = np.linalg.norm(coord - cluster_mean)
                
                if distance < min_distance:
                    min_distance = distance
                    assigned_cluster = cluster_id
            
            # Get weight from cluster
            weight = cluster_weights.get(assigned_cluster, 1.0 / len(entities))
            entity_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(entity_weights)
        if total_weight > 0:
            entity_weights = [w / total_weight for w in entity_weights]
        
        return entity_weights

    @staticmethod
    def calculate_pdf_values(clusters_info: Dict[int, Dict[str, Any]],
                           grid_points: np.ndarray,
                           center_coords: Tuple[float, float],
                           center_influence_factor: float = 3.0) -> np.ndarray:
        """
        Calculates probability density function values on a grid considering both
        cluster distributions and center influence.
        
        Args:
            clusters_info: Dictionary with cluster information including mean, cov
            grid_points: Array of shape (n, 2) with grid coordinates
            center_coords: (lat, lon) of the city center
            center_influence_factor: Factor to boost center PDF influence (default: 3.0)
            
        Returns:
            Array of PDF values at each grid point
        """
        pdf_values = np.zeros(len(grid_points))
        
        # Calculate center PDF with stronger influence
        center_cov = np.eye(2) * 0.0001  # Tight covariance for center
        center_pdf_obj = multivariate_normal(mean=center_coords, cov=center_cov)
        center_pdf = center_pdf_obj.pdf(grid_points)
        
        # Calculate cluster PDFs
        for cluster_id, info in clusters_info.items():
            try:
                cluster_pdf_obj = multivariate_normal(mean=info['mean'], cov=info['cov'])
                cluster_pdf = cluster_pdf_obj.pdf(grid_points)
                cluster_weight = info['weight']
                
                # Add weighted cluster contribution
                pdf_values += cluster_pdf * cluster_weight
            except np.linalg.LinAlgError:
                warnings.warn(f"Singular covariance matrix for cluster {cluster_id}, skipping")
                continue
        
        # Add center contribution with boosted weight
        # This addresses the issue mentioned in lines 453 and 474
        pdf_values += center_pdf * center_influence_factor
        
        # Normalize
        max_pdf = np.max(pdf_values)
        if max_pdf > 0:
            pdf_values = pdf_values / max_pdf
        
        return pdf_values

    @staticmethod
    def distribute_entities_by_weights(total_entities: int,
                                      stations: List[str],
                                      station_weights: List[float]) -> Dict[str, int]:
        """
        Distributes a total number of entities across stations based on weights.
        Uses multinomial distribution for proportional allocation.
        
        Args:
            total_entities: Total number of entities to distribute
            stations: List of station identifiers
            station_weights: List of weights for each station
            
        Returns:
            Dictionary mapping station to number of entities
        """
        if len(stations) != len(station_weights):
            raise ValueError("Number of stations must match number of weights")
        
        # Normalize weights to probabilities
        total_weight = sum(station_weights)
        if total_weight == 0:
            # Equal distribution if all weights are zero
            probs = [1.0 / len(stations)] * len(stations)
        else:
            probs = [w / total_weight for w in station_weights]
        
        # Use multinomial to distribute
        distribution = np.random.multinomial(total_entities, probs)
        
        # Create mapping
        entity_distribution = {station: count for station, count in zip(stations, distribution)}
        
        return entity_distribution
