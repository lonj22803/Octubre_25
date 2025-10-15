import math

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

def calcular_centroide_esferico(coordenadas):
    """
    Calcula el centroide geográfico esférico de una lista de coordenadas (lat, lon)
    usando coordenadas cartesianas 3D.
    """
    x = y = z = 0.0
    for lat, lon in coordenadas:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x += math.cos(lat_rad) * math.cos(lon_rad)
        y += math.cos(lat_rad) * math.sin(lon_rad)
        z += math.sin(lat_rad)

    total = len(coordenadas)
    x /= total
    y /= total
    z /= total

    lon_centro = math.atan2(y, x)
    hyp = math.sqrt(x * x + y * y)
    lat_centro = math.atan2(z, hyp)

    return math.degrees(lat_centro), math.degrees(lon_centro)