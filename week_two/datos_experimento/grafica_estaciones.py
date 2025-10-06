import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx

# Cargar CSV
df = pd.read_csv("estaciones_sistema.csv")

# Crear geometría de puntos a partir de longitud y latitud
geometry = [Point(xy) for xy in zip(df['longitud'], df['latitud'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84

# Convertir a proyección web mercator para usar mapa de fondo
gdf = gdf.to_crs(epsg=3857)

# Graficar
fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(ax=ax, color='dodgerblue', edgecolor='black', markersize=100, alpha=0.8, zorder=3)

#Graficamo un circulo al rededor del centro de madrid
centro_madrid = gpd.GeoSeries([Point(-3.691800, 40.429800)], crs="EPSG:4326").to_crs(epsg=3857)
centro_madrid.plot(ax=ax, color='red', markersize=200, marker='x', zorder=4)
circulo = centro_madrid.buffer(5000)  # 5 km de radio
circulo.plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', linewidth=2, zorder=2)

# Añadir etiquetas
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['estacion']):
    ax.text(x + 50, y + 50, label, fontsize=10, ha='left', va='bottom', zorder=4)

# Añadir mapa de fondo
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)


# Título y ejes
ax.set_title('Posiciones de Estaciones sobre Mapa', fontsize=16)
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.set_axis_off()  # Opcional: quita los ejes para ver mejor el mapa
plt.savefig("estaciones_sobre_mapa_madrid.png", dpi=300, bbox_inches='tight')
plt.show()


