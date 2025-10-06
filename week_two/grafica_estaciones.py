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

plt.show()


