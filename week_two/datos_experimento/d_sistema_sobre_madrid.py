import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from pyproj import Transformer

sistema_generico= {
    "lineas":{
        "amarilla": {
            "estaciones": ["AA1SC","AB2SC","AC3SC","AD4RF","AE5VE","AF6SC","AG7BH"],
            "sentido_ida":["AA1SC","AB2SC","AC3SC","AD4RF","AE5VE","AF6SC","AG7BH"],
            "sentido_vuelta":["AG7BH","AF6SC","AE5VE","AD4RF","AC3SC","AB2SC","AA1SC"]
        },
        "azul": {
            "estaciones": ["BA1SC","BB2OC","BC3SC","BD2VB","BE4RC","BF5SC","BG6SC","AG7BH"],
            "sentido_ida":["BA1SC","BB2OC","BC3SC","BD2VB","BE4RC","BF5SC","BG6SC","AG7BH"],
            "sentido_vuelta":["AG7BH","BG6SC","BF5SC","BE4RC","BD2VB","BC3SC","BB2OC","BA1SC"]
        },
        "roja": {
            "estaciones":["RA1SC","RB2SC","BE4RC","RD3VC","RE5SC","AD4RF","RG6SC"],
            "sentido_ida":["RA1SC","RB2SC","BE4RC","RD3VC","RE5SC","AD4RF","RG6SC"],
            "sentido_vuelta":["RG6SC","AD4RF","RE5SC","RD3VC","BE4RC","RB2SC","RA1SC"]
        },
        "verde": {
            "estaciones": ["VA1SC","BD2VB","RD3VC","VD4SC","AE5VE","VF6SC"],
            "sentido_ida":["VA1SC","BD2VB","RD3VC","VD4SC","AE5VE","VF6SC"],
            "sentido_vuelta":["VF6SC","AE5VE","VD4SC","RD3VC","BD2VB","VA1SC"]
        },
        "naranja":{
            "estaciones": ["OA1SC","OB2SC","BB2OC","OC3SC","RD3VC"],
            "sentido_ida":["OA1SC","OB2SC","BB2OC","OC3SC","RD3VC"],
            "sentido_vuelta":["RD3VC","OC3SC","BB2OC","OB2SC","OA1SC"]
        }
    }
}

# Mapeo de colores
color_map = {
    "amarilla": "yellow",
    "azul": "blue",
    "roja": "red",
    "verde": "green",
    "naranja": "orange"
}

# Creamos un grafo
G = nx.Graph()

# Añadimos nodos y aristas por línea
for color, info in sistema_generico["lineas"].items():
    estaciones = info["sentido_ida"]
    for i in range(len(estaciones)-1):
        G.add_edge(estaciones[i], estaciones[i+1], color=color_map[color])

# Coordenadas de las estaciones (lat, lon)
pos = {
    "AA1SC": (40.418800, -3.707800),
    "AB2SC": (40.412800, -3.698800),
    "AC3SC": (40.416800, -3.691800),
    "AD4RF": (40.420800, -3.685800),
    "AE5VE": (40.422800, -3.679800),
    "AF6SC": (40.426800, -3.671800),
    "AG7BH": (40.428800, -3.663800),
    "BA1SC": (40.434800, -3.715800),
    "BB2OC": (40.436800, -3.709800),
    "BC3SC": (40.438800, -3.703800),
    "BD2VB": (40.440800, -3.697800),
    "BE4RC": (40.438800, -3.691800),
    "BF5SC": (40.434800, -3.683800),
    "BG6SC": (40.432800, -3.674800),
    "RA1SC": (40.442800, -3.679800),
    "RB2SC": (40.446800, -3.685800),
    "RD3VC": (40.432800, -3.693800),
    "RE5SC": (40.424800, -3.691800),
    "RG6SC": (40.404800, -3.691800),
    "VA1SC": (40.452800, -3.703800),
    "VD4SC": (40.428800, -3.685800),
    "VF6SC": (40.416800, -3.677800),
    "OA1SC": (40.454800, -3.719800),
    "OB2SC": (40.444800, -3.715800),
    "OC3SC": (40.424800, -3.703800),
}


# Transformador de WGS84 (EPSG:4326) a Web Mercator (EPSG:3857)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Convertir directamente el diccionario original
pos_web = {estacion: transformer.transform(lon, lat) for estacion, (lat, lon) in pos.items()}

# Crear figura
fig, ax = plt.subplots(figsize=(12, 10))

# Dibujar el grafo con las posiciones convertidas
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]

nx.draw(
    G,
    pos_web,
    with_labels=True,
    node_size=650,
    node_color="white",
    edge_color=colors,
    width=4,
    font_size=6,
    font_weight="bold",
    ax=ax
)

# Añadir mapa de fondo
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)


# Añadimos un círculo alrededor del centro de Madrid
centro_madrid = gpd.GeoSeries([Point(-3.6918, 40.429800)], crs="EPSG:4326").to_crs(epsg=3857)
centro_madrid.plot(ax=ax, color='red', markersize=100, zorder=4)
circulo = centro_madrid.buffer(5000)  # 5 km de radio
circulo.plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', linewidth=2, zorder=2)

# Configuración final
ax.set_title('Sistema de Transporte Genérico - Sobre Madrid', fontsize=16, fontweight='bold')
ax.set_axis_off()  # Quitar ejes para mejor visualización

plt.tight_layout()
plt.savefig("sistema_generico_madrid.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()