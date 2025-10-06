import math

# Coordenadas originales (cartesianas)
pos_original = {
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


def convertir_a_lat_lon(posiciones, centro_lat=40.4168, centro_lon=-3.7038, escala=0.001):
    """
    Convierte coordenadas cartesianas a geográficas

    Args:
        posiciones: Diccionario con coordenadas originales
        centro_lat: Latitud del centro (Madrid por defecto)
        centro_lon: Longitud del centro (Madrid por defecto)
        escala: Factor de conversión (0.001 = aproximadamente 111 metros por unidad)

    Returns:
        Diccionario con coordenadas convertidas (lat, lon)
    """
    pos_geo = {}
    for key, (x, y) in posiciones.items():
        lat = centro_lat + x * escala
        lon = centro_lon + y * escala
        pos_geo[key] = (lat, lon)
    return pos_geo


# Convertir coordenadas
pos_geo = convertir_a_lat_lon(pos_original)

print("=== POSICIONES GEOGRÁFICAS ===\n")
print(pos_geo)

# Mostrar resultados
print("pos_geo = {")
for key, (lat, lon) in pos_geo.items():
    print(f'    "{key}": ({lat:.6f}, {lon:.6f}),')
print("}")

# Mostrar resumen
print(f"\n--- RESUMEN ---")
print(f"Total de puntos: {len(pos_geo)}")
print(f"Ciudad centro: Madrid ({40.4168}, {-3.7038})")
print(f"Escala utilizada: {0.001}")

# Calcular rango geográfico
lats = [lat for lat, lon in pos_geo.values()]
lons = [lon for lat, lon in pos_geo.values()]
print(f"Rango latitud: {min(lats):.6f} a {max(lats):.6f}")
print(f"Rango longitud: {min(lons):.6f} a {max(lons):.6f}")

#Analicemos la distancia máxima entre estaciones

def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en kilómetros entre dos puntos geográficos
    usando la fórmula de Haversine
    """
    # Radio de la Tierra en kilómetros
    R = 6371.0

    # Convertir grados a radianes
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Diferencias
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Fórmula de Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distancia = R * c
    return distancia


# Calcular todas las distancias entre pares de estaciones
distancias = []
estaciones = list(pos_geo.keys())

print("=== ANÁLISIS DE DISTANCIAS ENTRE ESTACIONES ===\n")

# Calcular distancias entre todos los pares únicos
for i in range(len(estaciones)):
    for j in range(i + 1, len(estaciones)):
        estacion1 = estaciones[i]
        estacion2 = estaciones[j]
        lat1, lon1 = pos_geo[estacion1]
        lat2, lon2 = pos_geo[estacion2]

        distancia = calcular_distancia_haversine(lat1, lon1, lat2, lon2)
        distancias.append(distancia)

# Estadísticas
distancia_promedio = sum(distancias) / len(distancias)
distancia_maxima = max(distancias)
distancia_minima = min(distancias)
total_pares = len(distancias)

print(f"ESTADÍSTICAS GENERALES:")
print(f"Total de estaciones: {len(estaciones)}")
print(f"Total de pares únicos: {total_pares}")
print(f"Distancia promedio entre estaciones: {distancia_promedio:.2f} km")
print(f"Distancia mínima entre estaciones: {distancia_minima:.2f} km")
print(f"Distancia máxima entre estaciones: {distancia_maxima:.2f} km")

# Encontrar el par más cercano y más lejano
print(f"\nPARES ESPECIALES:")

# Buscar par más cercano
min_dist = float('inf')
par_cercano = None
for i in range(len(estaciones)):
    for j in range(i + 1, len(estaciones)):
        est1, est2 = estaciones[i], estaciones[j]
        lat1, lon1 = pos_geo[est1]
        lat2, lon2 = pos_geo[est2]
        dist = calcular_distancia_haversine(lat1, lon1, lat2, lon2)
        if dist < min_dist:
            min_dist = dist
            par_cercano = (est1, est2)

# Buscar par más lejano
max_dist = 0
par_lejano = None
for i in range(len(estaciones)):
    for j in range(i + 1, len(estaciones)):
        est1, est2 = estaciones[i], estaciones[j]
        lat1, lon1 = pos_geo[est1]
        lat2, lon2 = pos_geo[est2]
        dist = calcular_distancia_haversine(lat1, lon1, lat2, lon2)
        if dist > max_dist:
            max_dist = dist
            par_lejano = (est1, est2)

print(f"Par más cercano: {par_cercano[0]} - {par_cercano[1]}: {min_dist:.2f} km")
print(f"Par más lejano: {par_lejano[0]} - {par_lejano[1]}: {max_dist:.2f} km")

# Análisis por grupos de distancia
print(f"\nDISTRIBUCIÓN POR RANGOS:")
distancias_0_5 = [d for d in distancias if d <= 5]
distancias_5_10 = [d for d in distancias if 5 < d <= 10]
distancias_10_15 = [d for d in distancias if 10 < d <= 15]
distancias_15_plus = [d for d in distancias if d > 15]

print(f"0-5 km: {len(distancias_0_5)} pares ({len(distancias_0_5) / total_pares * 100:.1f}%)")
print(f"5-10 km: {len(distancias_5_10)} pares ({len(distancias_5_10) / total_pares * 100:.1f}%)")
print(f"10-15 km: {len(distancias_10_15)} pares ({len(distancias_10_15) / total_pares * 100:.1f}%)")
print(f"+15 km: {len(distancias_15_plus)} pares ({len(distancias_15_plus) / total_pares * 100:.1f}%)")

# Radio aproximado del área cubierta
print(f"\nÁREA CUBIERTA:")
lats = [lat for lat, lon in pos_geo.values()]
lons = [lon for lat, lon in pos_geo.values()]
lat_centro = (max(lats) + min(lats)) / 2
lon_centro = (max(lons) + min(lons)) / 2

radio_aproximado = calcular_distancia_haversine(lat_centro, lon_centro, max(lats), max(lons))
print(f"Centro aproximado del sistema: ({lat_centro:.6f}, {lon_centro:.6f})")
print(f"Radio aproximado del área: {radio_aproximado:.2f} km")
print(f"Diámetro aproximado: {radio_aproximado * 2:.2f} km")