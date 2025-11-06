# Objets Package - Sistema Generalizado de Clustering

Este paquete proporciona una arquitectura generalizada para la generación y gestión de entidades urbanas (metro, hoteles, restaurantes, lugares turísticos) utilizando técnicas de clustering y distribución de pesos.

## Estructura del Paquete

```
Objets/
├── City/
│   ├── City.py              # Clase base CityBase con métodos estáticos
│   ├── MetroSystem.py       # Sistema de metro que hereda de CityBase
│   └── __init__.py
├── Hotel/
│   ├── Hotels.py            # Generación de hoteles con clustering
│   └── __init__.py
├── Restaurant/
│   ├── Restaurants.py       # Generación de restaurantes con clustering
│   └── __init__.py
├── TouristPlace/
│   ├── TouristPlace.py      # Generación de lugares turísticos
│   └── __init__.py
├── __init__.py
├── SOLUCION_PDF_WEIGHTS.md  # Documentación del problema y solución
└── pdf_comparison.png       # Visualización comparativa
```

## Características Principales

### 1. Arquitectura Basada en Herencia

Todas las clases heredan de `CityBase`, que proporciona métodos estáticos para:
- Validación de parámetros
- Generación de clusters
- Cálculo de pesos con influencia del centro
- Asignación de pesos a entidades
- Distribución de entidades por pesos

### 2. Solución al Problema de Influencia de Clusters

El paquete resuelve el problema donde los clusters periféricos tenían más influencia que el centro de la ciudad mediante:

- **`center_weight_multiplier`**: Multiplica el peso del cluster más cercano al centro
- **`center_influence_factor`**: Aumenta la influencia del centro en cálculos de PDF

Ver `SOLUCION_PDF_WEIGHTS.md` para detalles completos.

### 3. Código Reutilizable

Los métodos estáticos permiten:
- Usar la misma lógica en múltiples clases
- Personalizar parámetros por tipo de entidad
- Mantener un único punto de actualización

## Uso Básico

### Importar Clases

```python
from Objets import MetroSystem, Hotels, Restaurants, TouristPlace
```

### Crear Sistema de Metro

```python
sistema_metro = MetroSystem()
sistema_metro.load_system(sistema_basico_prueba)
sistema_metro.add_station(posiciones_estaciones, (40.4168, -3.7038))
analisis_geo = sistema_metro.geographics_dates()
```

### Generar Hoteles

```python
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

# Inspeccionar clusters y pesos
print(f"Clusters: {len(hoteles.clusters_info)}")
print(f"Pesos: {hoteles.cluster_weights}")
```

### Generar Restaurantes

```python
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
```

### Usar Métodos Estáticos Directamente

```python
from Objets.City.City import CityBase

# Generar clusters desde coordenadas
coords = [(40.4, -3.7), (40.41, -3.71), (40.42, -3.72)]
clusters = CityBase.generate_clusters(coords, n_clusters=2)

# Calcular pesos con boost del centro
weights = CityBase.calculate_cluster_weights(
    clusters, 
    center_coords=(40.4, -3.7),
    center_weight_multiplier=2.5
)

# Distribuir entidades
distribution = CityBase.distribute_entities_by_weights(
    total_entities=100,
    stations=['A', 'B', 'C'],
    station_weights=[0.5, 0.3, 0.2]
)
```

## Parámetros Importantes

### center_weight_multiplier
- **Propósito**: Aumenta el peso del cluster más cercano al centro
- **Rango recomendado**: 1.5 - 5.0
- **Default**: 2.0
- **Uso en clases**:
  - Hotels: 2.5
  - Restaurants: 3.0
  - TouristPlace: 4.0

### center_influence_factor
- **Propósito**: Aumenta la influencia del centro en cálculos de PDF
- **Rango recomendado**: 2.0 - 10.0
- **Default**: 3.0
- **Soluciona**: El problema de las líneas 453 y 474 mencionado

## Testing

Ejecutar tests:
```bash
python test_generalized_architecture.py
```

Ver ejemplo de solución PDF:
```bash
python example_pdf_solution.py
```

## Dependencias

- numpy
- scipy
- scikit-learn
- pandas
- geopandas
- matplotlib
- networkx
- contextily
- shapely
- pyproj

## Extensibilidad

Para añadir una nueva clase (ej: Airports):

1. Crear archivo en el directorio apropiado
2. Heredar de `CityBase`
3. Implementar método de generación usando métodos heredados
4. Ajustar `center_weight_multiplier` según necesidades

Ejemplo:

```python
from Objets.City.City import CityBase

class Airports(CityBase):
    def __init__(self, lat, lon, num_airports):
        self.city_center = (lat, lon)
        self.num_airports = num_airports
        self.clusters_info = {}
        self.cluster_weights = {}
    
    def generate_airports(self, ...):
        # Usar métodos heredados
        validated = self.validate_parameters(...)
        clusters = self.generate_clusters(...)
        weights = self.calculate_cluster_weights(
            clusters, 
            self.city_center,
            center_weight_multiplier=1.5  # Aeropuertos más dispersos
        )
        # ... resto de lógica
```

## Contribución

Para contribuir al paquete:
1. Mantener métodos estáticos en `CityBase`
2. Documentar parámetros con docstrings
3. Añadir tests en `test_generalized_architecture.py`
4. Actualizar este README

## Referencias

- Ver `SOLUCION_PDF_WEIGHTS.md` para detalles del problema de influencia
- Ver `example_pdf_solution.py` para ejemplos de uso avanzado
- Ver `test_generalized_architecture.py` para casos de prueba
