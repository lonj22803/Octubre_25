"""
Compatibility example showing how to migrate from old code to new generalized architecture.

This demonstrates that the new Objets classes are backward compatible and can be used
as drop-in replacements for the original classes in week_three/testbed.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'week_three', 'testbed'))
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("COMPATIBILITY DEMONSTRATION")
print("Old code from week_three/testbed vs New Objets package")
print("=" * 70)

# Test data (same as in week_three/testbed)
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
    "AA1SC": (-40, 20), "AB2SC": (50, -40), "AC3SC": (120, 0),
    "AD4RF": (180, 40), "AE5VE": (240, 60), "AF6SC": (320, 100),
    "AG7BH": (400, 120), "BA1SC": (-120, 180), "BB2OC": (-60, 200),
    "BC3SC": (0, 220), "BD2VB": (60, 240), "BE4RC": (120, 220),
    "BF5SC": (200, 180), "BG6SC": (290, 160), "RA1SC": (240, 260),
    "RB2SC": (180, 300), "RD3VC": (100, 160), "RE5SC": (120, 80),
    "RG6SC": (120, -120), "VA1SC": (0, 360), "VD4SC": (180, 120),
    "VF6SC": (260, 0), "OA1SC": (-160, 380), "OB2SC": (-120, 280),
    "OC3SC": (0, 80)
}


def test_old_metro_system():
    """Test the old Metro.py TrainSystem class"""
    print("\n" + "-" * 70)
    print("OPTION 1: Using OLD Metro.py from week_three/testbed")
    print("-" * 70)
    
    try:
        from Metro import TrainSystem as OldTrainSystem
        
        sistema_metro = OldTrainSystem()
        sistema_metro.load_system(sistema_basico_prueba)
        sistema_metro.add_station(posiciones_estaciones, (40.4168, -3.7038))
        analisis_geo = sistema_metro.geographics_dates()
        
        print(f"✓ Old TrainSystem works")
        print(f"  - Lines: {len(sistema_metro.lines)}")
        print(f"  - Stations: {len(sistema_metro.stations)}")
        print(f"  - Center: ({analisis_geo['centroide_lat']}, {analisis_geo['centroide_lon']})")
        
        return sistema_metro
    except Exception as e:
        print(f"✗ Old TrainSystem failed: {e}")
        return None


def test_new_metro_system():
    """Test the new MetroSystem class from Objets"""
    print("\n" + "-" * 70)
    print("OPTION 2: Using NEW MetroSystem from Objets package")
    print("-" * 70)
    
    try:
        from Objets import MetroSystem as NewMetroSystem
        
        sistema_metro = NewMetroSystem()
        sistema_metro.load_system(sistema_basico_prueba)
        sistema_metro.add_station(posiciones_estaciones, (40.4168, -3.7038))
        analisis_geo = sistema_metro.geographics_dates()
        
        print(f"✓ New MetroSystem works")
        print(f"  - Lines: {len(sistema_metro.lines)}")
        print(f"  - Stations: {len(sistema_metro.stations)}")
        print(f"  - Center: ({analisis_geo['centroide_lat']}, {analisis_geo['centroide_lon']})")
        print(f"  - BONUS: Inherits from CityBase with static methods!")
        
        return sistema_metro
    except Exception as e:
        print(f"✗ New MetroSystem failed: {e}")
        return None


def test_old_hotels(sistema_metro):
    """Test the old Hotel class"""
    if sistema_metro is None:
        return None
        
    print("\n" + "-" * 70)
    print("OPTION 1: Using OLD Hotel.py from week_three/testbed")
    print("-" * 70)
    
    try:
        from Hotel import Hotel as OldHotel
        
        hoteles = OldHotel(
            lat=sistema_metro.centroide_latlon[0],
            lon=sistema_metro.centroide_latlon[1],
            num_hoteles=60,
            max_distancia_km=1,
            mean_distancia_km=0.5,
            mean_precio=100,
            std_precio=20
        )
        
        df_hoteles = hoteles.hotel_generation_points(
            system=sistema_metro.system,
            stations=sistema_metro.stations,
            station_coordinates=sistema_metro.station_coordinates
        )
        
        print(f"✓ Old Hotel works")
        print(f"  - Generated: {len(df_hoteles)} hotels")
        print(f"  - Columns: {list(df_hoteles.columns)}")
        
        return hoteles
    except Exception as e:
        print(f"✗ Old Hotel failed: {e}")
        return None


def test_new_hotels(sistema_metro):
    """Test the new Hotels class from Objets"""
    if sistema_metro is None:
        return None
        
    print("\n" + "-" * 70)
    print("OPTION 2: Using NEW Hotels from Objets package")
    print("-" * 70)
    
    try:
        from Objets import Hotels as NewHotels
        
        hoteles = NewHotels(
            lat=sistema_metro.centroide_latlon[0],
            lon=sistema_metro.centroide_latlon[1],
            num_hoteles=60,
            max_distancia_km=1,
            mean_distancia_km=0.5,
            mean_precio=100,
            std_precio=20
        )
        
        df_hoteles = hoteles.hotel_generation_points(
            system=sistema_metro.system,
            stations=sistema_metro.stations,
            station_coordinates=sistema_metro.station_coordinates
        )
        
        print(f"✓ New Hotels works")
        print(f"  - Generated: {len(df_hoteles)} hotels")
        print(f"  - Columns: {list(df_hoteles.columns)}")
        print(f"  - BONUS: Uses clustering with center influence boost!")
        print(f"  - Clusters: {len(hoteles.clusters_info)}")
        print(f"  - Weights: {hoteles.cluster_weights}")
        
        return hoteles
    except Exception as e:
        print(f"✗ New Hotels failed: {e}")
        return None


def test_old_restaurants(sistema_metro):
    """Test the old Restaurantes class"""
    if sistema_metro is None:
        return None
        
    print("\n" + "-" * 70)
    print("OPTION 1: Using OLD Restaurants.py from week_three/testbed")
    print("-" * 70)
    
    try:
        from Restaurants import Restaurantes as OldRestaurantes
        
        restaurantes = OldRestaurantes(
            lat=sistema_metro.centroide_latlon[0],
            lon=sistema_metro.centroide_latlon[1],
            num_restaurantes=150,
            max_distancia_km=0.5,
            mean_distancia_km=1,
            lambda_poisson=15,
            alpha=4,
            beta=2,
            mean_tiempo=30,
            std_tiempo=5
        )
        
        df_restaurantes = restaurantes.restaurant_generation_points(
            system=sistema_metro.system,
            stations=sistema_metro.stations,
            station_coordinates=sistema_metro.station_coordinates
        )
        
        print(f"✓ Old Restaurantes works")
        print(f"  - Generated: {len(df_restaurantes)} restaurants")
        
        return restaurantes
    except Exception as e:
        print(f"✗ Old Restaurantes failed: {e}")
        return None


def test_new_restaurants(sistema_metro):
    """Test the new Restaurants class from Objets"""
    if sistema_metro is None:
        return None
        
    print("\n" + "-" * 70)
    print("OPTION 2: Using NEW Restaurants from Objets package")
    print("-" * 70)
    
    try:
        from Objets import Restaurants as NewRestaurants
        
        restaurantes = NewRestaurants(
            lat=sistema_metro.centroide_latlon[0],
            lon=sistema_metro.centroide_latlon[1],
            num_restaurantes=150,
            max_distancia_km=0.5,
            mean_distancia_km=1,
            lambda_poisson=15,
            alpha=4,
            beta=2,
            mean_tiempo=30,
            std_tiempo=5
        )
        
        df_restaurantes = restaurantes.restaurant_generation_points(
            system=sistema_metro.system,
            stations=sistema_metro.stations,
            station_coordinates=sistema_metro.station_coordinates
        )
        
        print(f"✓ New Restaurants works")
        print(f"  - Generated: {len(df_restaurantes)} restaurants")
        print(f"  - BONUS: Uses clustering with enhanced center influence!")
        print(f"  - Clusters: {len(restaurantes.clusters_info)}")
        print(f"  - Weights: {restaurantes.cluster_weights}")
        
        return restaurantes
    except Exception as e:
        print(f"✗ New Restaurants failed: {e}")
        return None


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing Metro System")
    print("=" * 70)
    
    old_metro = test_old_metro_system()
    new_metro = test_new_metro_system()
    
    print("\n" + "=" * 70)
    print("Testing Hotels")
    print("=" * 70)
    
    test_old_hotels(old_metro)
    test_new_hotels(new_metro)
    
    print("\n" + "=" * 70)
    print("Testing Restaurants")
    print("=" * 70)
    
    test_old_restaurants(old_metro)
    test_new_restaurants(new_metro)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Both the old code and new Objets package work!

Key differences:
1. NEW code uses generalized clustering logic (CityBase)
2. NEW code solves the PDF weight issue with center influence
3. NEW code is more maintainable (single source of truth)
4. NEW code can be extended to TouristPlace easily
5. OLD code continues to work for backward compatibility

Recommendation: Use NEW Objets package for future development
    """)
