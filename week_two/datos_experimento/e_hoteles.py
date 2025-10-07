import json
import math
import random
import csv
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
from geopy.distance import geodesic  # Para calcular distancias reales en km (instala con: pip install geopy)

# ===== Parámetros de generación =====
center_lat, center_lon = 40.4298, -3.6918  # Centro de Madrid
num_hotels = 35  # Número de hoteles a generar
max_distance_km = 3 # Radio máximo en km desde el centro para colocar hoteles
mean_dist_km = 1.0  # Media de la distribución exponencial para concentrar más en el centro (ajusta para más/menos concentración)
mean_price = 160
std_price = 30


# ===== Función para generar puntos aleatorios con distribución concentrada en el centro =====
def generate_concentrated_point_around_center(lat, lon, max_dist_km, mean_dist_km):
    while True:
        # Generar ángulo aleatorio (0 a 2π)
        angle = random.uniform(0, 2 * math.pi)

        # Generar distancia con distribución exponencial (decae rápidamente hacia el centro)
        # Lambda = 1 / mean_dist_km para que la media sea mean_dist_km
        dist = random.expovariate(1 / mean_dist_km)

        # Si la distancia excede el máximo, regenerar (para mantener dentro del radio)
        if dist > max_dist_km:
            continue

        # Calcular nuevo punto usando aproximación (para distancias pequeñas, es precisa)
        earth_radius_km = 6371
        lat_offset = (dist / earth_radius_km) * (180 / math.pi)
        lon_offset = (dist / earth_radius_km) * (180 / math.pi) / math.cos(math.radians(lat))

        new_lat = lat + lat_offset * math.cos(angle)
        new_lon = lon + lon_offset * math.sin(angle)

        # Verificar distancia real con geodesic y retornar si está bien
        if geodesic((lat, lon), (new_lat, new_lon)).km <= max_dist_km:
            return new_lat, new_lon


# ===== Generar hoteles cerca del centro con mayor concentración =====
hotels = []
generated_distances = []  # Para verificar la distribución (opcional, para debugging)
for i in range(num_hotels):
    lat, lon = generate_concentrated_point_around_center(center_lat, center_lon, max_distance_km, mean_dist_km)

    actual_dist = geodesic((center_lat, center_lon), (lat, lon)).km
    generated_distances.append(actual_dist)

    price = max(50, random.gauss(mean_price, std_price))  # Evita precios negativos
    hotels.append({
        "Hotel": f"Hotel_{i + 1}",
        "Latitud": round(lat, 6),
        "Longitud": round(lon, 6),
        "Precio": round(price, 2),
        "Distancia_al_Centro_km": round(actual_dist, 3)  # Agregado para análisis
    })

# ===== Estadísticas de distancias (para verificar concentración) =====
print(f"Distancia media al centro: {sum(generated_distances) / len(generated_distances):.2f} km")
print(f"Distancia máxima: {max(generated_distances):.2f} km")
print(f"Distancia mínima: {min(generated_distances):.2f} km")

# ===== Guardar en CSV (incluyendo distancia) =====
csv_filename = "hoteles_concentrados_centro.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Hotel", "Latitud", "Longitud", "Precio", "Distancia_al_Centro_km"])
    writer.writeheader()
    writer.writerows(hotels)

print(
    f"Archivo '{csv_filename}' generado con {len(hotels)} hoteles concentrados cerca del centro de Madrid (media: {mean_dist_km} km, máx: {max_distance_km} km).")

# ===== Cargar CSV de hoteles =====
df = pd.read_csv(csv_filename)

# ===== Gráfico 1: Distribución de precios =====
plt.figure(figsize=(8, 5))
plt.hist(df["Precio"], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribución de Precios de Hoteles", fontsize=14)
plt.xlabel("Precio (€)")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("distribucion_precios_hoteles.png", dpi=300, bbox_inches='tight')
plt.show()

# ===== Gráfico Adicional: Distribución de Distancias al Centro (nuevo para verificar concentración) =====
plt.figure(figsize=(8, 5))
plt.hist(df["Distancia_al_Centro_km"], bins=10, color='lightgreen', edgecolor='black')
plt.title("Distribución de Distancias de Hoteles al Centro", fontsize=14)
plt.xlabel("Distancia al Centro (km)")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.axvline(x=mean_dist_km, color='red', linestyle='--', label=f'Media esperada: {mean_dist_km} km')
plt.legend()
plt.tight_layout()
plt.savefig("distribucion_distancias_centro.png", dpi=300, bbox_inches='tight')
plt.show()

# ===== Gráfico 2: Hoteles sobre mapa (enfocado en cercanía al centro) =====
geometry = [Point(xy) for xy in zip(df["Longitud"], df["Latitud"])]
gdf_hoteles = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12, 10))
gdf_hoteles.plot(ax=ax, color='dodgerblue', edgecolor='black', markersize=100, alpha=0.8, zorder=3)

# Centro y círculo de radio (convertir km a metros para buffer)
centro_madrid = gpd.GeoSeries([Point(center_lon, center_lat)], crs="EPSG:4326").to_crs(epsg=3857)
centro_madrid.plot(ax=ax, color='red', markersize=200, marker='x', zorder=4)
circulo = centro_madrid.buffer(max_distance_km * 1000)  # Buffer en metros
circulo.plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', linewidth=2, zorder=2, alpha=0.7)

for x, y, label, precio, dist in zip(gdf_hoteles.geometry.x, gdf_hoteles.geometry.y, gdf_hoteles['Hotel'],
                                     gdf_hoteles['Precio'], df['Distancia_al_Centro_km']):
    ax.text(x + 30, y + 30, f"{label}", fontsize=8, ha='left', va='bottom', zorder=4)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Hoteles Generados Concentrados Cerca del Centro de Madrid", fontsize=16)
ax.set_axis_off()
plt.savefig("hoteles_concentrados_sobre_mapa_centro_madrid.png", dpi=300, bbox_inches='tight')
plt.show()

# ===== Gráfico 3: Hoteles + Estaciones de metro (hoteles independientes de estaciones) =====
# Asegúrate de tener el CSV 'estaciones_sistema.csv' con columnas ['estacion','latitud','longitud']
# Si no lo tienes, comenta esta sección o proporciona el archivo.
try:
    df_metro = pd.read_csv("estaciones_sistema.csv")

    # Crear geometría para estaciones
    geometry_metro = [Point(xy) for xy in zip(df_metro["longitud"], df_metro["latitud"])]
    gdf_metro = gpd.GeoDataFrame(df_metro, geometry=geometry_metro, crs="EPSG:4326").to_crs(epsg=3857)

    # Crear figura combinada
    fig, ax = plt.subplots(figsize=(12, 10))

    # Graficar estaciones y hoteles (hoteles ya concentrados en el centro, no de estaciones)
    gdf_metro.plot(ax=ax, color='orange', edgecolor='black', markersize=80, alpha=0.9, label="Estaciones de Metro",
                   zorder=3)
    gdf_hoteles.plot(ax=ax, color='dodgerblue', edgecolor='black', markersize=120, alpha=0.8,
                     label="Hoteles (concentrados en centro)", zorder=4)

    # Centro y círculo
    centro_madrid.plot(ax=ax, color='red', markersize=200, marker='x', label="Centro", zorder=5)
    circulo.plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', linewidth=2, zorder=2, alpha=0.7)

    # Mapa de fondo y detalles
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title("Hoteles Concentrados en el Centro y Estaciones de Metro en Madrid", fontsize=16)
    ax.set_axis_off()

    plt.savefig("hoteles_concentrados_centro_y_metro_madrid.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Gráfico 3 generado exitosamente (incluye estaciones de metro para comparación).")
except FileNotFoundError:
    print(
        "Archivo 'estaciones_sistema.csv' no encontrado. Gráfico 3 omitido. Los hoteles se generaron independientemente de las estaciones.")
except Exception as e:
    print(f"Error al generar Gráfico 3: {e}. Continuando sin él.")