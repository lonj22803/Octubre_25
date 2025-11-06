# Solución al Problema de Influencia de Clusters vs Centro en Gráficas 3D

## Problema Original

En el código original (líneas 453 y 474 mencionadas), las multiplicaciones:
- `center_pdf * center_weight`
- `cluster_pdf * cluster_weight`

Mostraban que los clusters (especialmente el cluster 0) tenían más influencia que el centro de la ciudad en las gráficas 3D resultantes. Esto era problemático porque el centro debería tener mayor importancia en la distribución de hoteles, restaurantes y lugares turísticos.

## Causa del Problema

1. **Pesos proporcionales al tamaño**: Los clusters con más puntos recibían pesos más altos automáticamente
2. **Sin ajuste para distancia al centro**: No se consideraba la proximidad al centro de la ciudad
3. **PDF del centro sin boost**: El PDF del centro no tenía un multiplicador para aumentar su influencia

## Solución Implementada

### 1. Método `calculate_cluster_weights` en `CityBase`

```python
@staticmethod
def calculate_cluster_weights(clusters_info: Dict[int, Dict[str, Any]], 
                              center_coords: Tuple[float, float],
                              center_weight_multiplier: float = 2.0) -> Dict[int, float]:
    """
    Calcula pesos para clusters considerando distancia al centro.
    El centro recibe un multiplicador mayor para aumentar su influencia.
    
    Args:
        center_weight_multiplier: Multiplicador para el peso del centro (default: 2.0)
    """
```

**Características clave:**
- Identifica el cluster más cercano al centro
- Aplica `center_weight_multiplier` al cluster central (default: 2.0)
- Normaliza todos los pesos para que sumen 1

### 2. Método `calculate_pdf_values` en `CityBase`

```python
@staticmethod
def calculate_pdf_values(clusters_info: Dict[int, Dict[str, Any]],
                       grid_points: np.ndarray,
                       center_coords: Tuple[float, float],
                       center_influence_factor: float = 3.0) -> np.ndarray:
    """
    Calcula valores de función de densidad de probabilidad en una rejilla.
    
    Args:
        center_influence_factor: Factor para aumentar influencia del centro (default: 3.0)
    """
```

**Características clave:**
- Calcula PDF del centro con covarianza ajustada (más concentrada)
- Suma contribuciones ponderadas de clusters
- Aplica `center_influence_factor` al PDF del centro (default: 3.0)
- Esta es la **solución directa** al problema de las líneas 453 y 474

### 3. Uso en las Clases

#### Hotels
```python
self.cluster_weights = self.calculate_cluster_weights(
    self.clusters_info,
    self.city_center,
    center_weight_multiplier=2.5  # Aumentar influencia del centro
)
```

#### Restaurants
```python
self.cluster_weights = self.calculate_cluster_weights(
    self.clusters_info,
    self.city_center,
    center_weight_multiplier=3.0  # Mayor influencia para restaurantes
)
```

#### TouristPlace
```python
self.cluster_weights = self.calculate_cluster_weights(
    self.clusters_info,
    self.city_center,
    center_weight_multiplier=4.0  # Mayor influencia para lugares turísticos
)
```

## Resultados

### Antes
- Cluster 0 podía tener peso dominante (>70%) incluso si estaba lejos del centro
- La distribución favorecía áreas periféricas con muchos puntos
- Gráficas 3D mostraban "picos" en clusters alejados del centro

### Después
- El cluster central tiene automáticamente 2-4x más peso
- La distribución favorece áreas cerca del centro de la ciudad
- Gráficas 3D muestran un pico claro en el centro con decaimiento hacia la periferia

## Ejemplo de Uso

```python
from Objets import Hotels, MetroSystem

# Crear sistema de metro
sistema_metro = MetroSystem()
sistema_metro.load_system(sistema_basico_prueba)
sistema_metro.add_station(posiciones_estaciones, (40.4168, -3.7038))
sistema_metro.geographics_dates()

# Crear hoteles con clustering y pesos ajustados
hoteles = Hotels(
    lat=sistema_metro.centroide_latlon[0],
    lon=sistema_metro.centroide_latlon[1],
    num_hoteles=60,
    max_distancia_km=1,
    mean_distancia_km=0.5,
    mean_precio=100,
    std_precio=20
)

# Generar hoteles - internamente usa calculate_cluster_weights
# con center_weight_multiplier=2.5
df_hoteles = hoteles.hotel_generation_points(
    system=sistema_metro.system,
    stations=sistema_metro.stations,
    station_coordinates=sistema_metro.station_coordinates
)

# Verificar pesos de clusters
print("Pesos de clusters:", hoteles.cluster_weights)
# Ejemplo de salida: {0: 0.077, 1: 0.769, 2: 0.154}
# donde el cluster 1 está cerca del centro y tiene peso dominante
```

## Parámetros Ajustables

### `center_weight_multiplier`
- **Rango recomendado**: 1.5 - 5.0
- **Default**: 2.0
- **Efecto**: Multiplica el peso del cluster más cercano al centro

### `center_influence_factor`
- **Rango recomendado**: 2.0 - 10.0
- **Default**: 3.0
- **Efecto**: Multiplica el PDF del centro en cálculos de densidad

## Beneficios de la Arquitectura

1. **Código Generalizable**: Todos los métodos son estáticos en `CityBase`
2. **Reusabilidad**: `Hotels`, `Restaurants`, `TouristPlace` heredan la misma lógica
3. **Parametrizable**: Fácil ajustar el balance centro vs clusters
4. **Mantenible**: Un solo lugar para actualizar la lógica de clustering
5. **Testeable**: Métodos estáticos son fáciles de probar independientemente

## Tests

Ver `test_generalized_architecture.py` para ejemplos completos de pruebas que verifican:
- Generación de clusters
- Cálculo de pesos con influencia del centro
- Distribución correcta de entidades
- Validación de parámetros
