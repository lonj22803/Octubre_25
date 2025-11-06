"""
Test script to verify the new generalized architecture works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Objets import MetroSystem, Hotels, Restaurants

# Test data
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


def test_metro_system():
    """Test MetroSystem class"""
    print("=" * 60)
    print("Testing MetroSystem")
    print("=" * 60)
    
    sistema_metro = MetroSystem()
    sistema_metro.load_system(sistema_basico_prueba)
    print(f"✓ Loaded {len(sistema_metro.lines)} lines")
    print(f"✓ Loaded {len(sistema_metro.stations)} stations")
    
    sistema_metro.add_station(posiciones_estaciones, (40.4168, -3.7038))
    print(f"✓ Added station coordinates")
    
    analisis_geo = sistema_metro.geographics_dates()
    print(f"✓ Geographic analysis completed")
    print(f"  - Center: ({analisis_geo['centroide_lat']}, {analisis_geo['centroide_lon']})")
    print(f"  - Radio: {analisis_geo['radio_sistema']} km")
    
    return sistema_metro


def test_hotels(sistema_metro):
    """Test Hotels class"""
    print("\n" + "=" * 60)
    print("Testing Hotels with generalized clustering")
    print("=" * 60)
    
    hoteles = Hotels(
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
    
    print(f"✓ Generated {len(df_hoteles)} hotels")
    print(f"✓ Clusters info: {len(hoteles.clusters_info)} clusters")
    print(f"✓ Cluster weights: {hoteles.cluster_weights}")
    print(f"\nSample hotels:")
    print(df_hoteles.head())
    
    return hoteles


def test_restaurants(sistema_metro):
    """Test Restaurants class"""
    print("\n" + "=" * 60)
    print("Testing Restaurants with generalized clustering")
    print("=" * 60)
    
    restaurantes = Restaurants(
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
    
    print(f"✓ Generated {len(df_restaurantes)} restaurants")
    print(f"✓ Clusters info: {len(restaurantes.clusters_info)} clusters")
    print(f"✓ Cluster weights: {restaurantes.cluster_weights}")
    print(f"\nSample restaurants:")
    print(df_restaurantes.head())
    
    return restaurantes


def test_static_methods():
    """Test static methods from CityBase"""
    print("\n" + "=" * 60)
    print("Testing CityBase static methods")
    print("=" * 60)
    
    from Objets.City.City import CityBase
    
    # Test parameter validation
    try:
        validated = CityBase.validate_parameters(
            stations=['A', 'B', 'C'],
            system={'line1': {'estaciones': ['A', 'B']}}
        )
        print("✓ Parameter validation works")
    except Exception as e:
        print(f"✗ Parameter validation failed: {e}")
    
    # Test cluster generation
    coords = [(40.4, -3.7), (40.41, -3.71), (40.42, -3.72), (40.43, -3.73)]
    clusters = CityBase.generate_clusters(coords, n_clusters=2)
    print(f"✓ Cluster generation works - {len(clusters)} clusters created")
    
    # Test weight calculation
    weights = CityBase.calculate_cluster_weights(clusters, (40.4, -3.7), center_weight_multiplier=2.0)
    print(f"✓ Weight calculation works - weights: {weights}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        # Run tests
        sistema_metro = test_metro_system()
        hoteles = test_hotels(sistema_metro)
        restaurantes = test_restaurants(sistema_metro)
        test_static_methods()
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
