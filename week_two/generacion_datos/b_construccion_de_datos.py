"""
Version 2.0 del sistema de transporte generico
Autor: Juan Jose Londoño Cardenas
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from fontTools.ttLib.woff2 import bboxFormat
import os

# Carpeta actual
print("Directorio actual:", os.getcwd())

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

#Guardamos el sistema en un archivo JSON
ruta_salida = os.path.join(os.path.dirname(__file__), "sistema_generico.json")
with open(ruta_salida, "w", encoding="utf-8") as f:
    json.dump(sistema_generico, f, ensure_ascii=False, indent=4)

# Mapeo de colores de español a matplotlib
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

# Posiciones manuales para evitar superposición llamamos el json con las posiciones
ruta_pos = os.path.join(os.path.dirname(__file__), "posiciones_geograficas.json")
with open(ruta_pos, "r") as f:
    pos = json.load(f)

# Invertir latitud y longitud (lat -> y, lon -> x)
pos_corrected = {node: (lon, lat) for node, (lat, lon) in pos.items()}

# Dibujamos el grafo
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
nx.draw(
    G,
    pos_corrected,
    with_labels=True,
    node_size=650,
    node_color="lightgray",
    edge_color=colors,
    width=5,
    font_size=6,
    font_weight=600
)
#Guardamos la figura
ruta_salida = os.path.join(os.path.dirname(__file__), "sistema_generico.png")
plt.savefig(ruta_salida, dpi=300,bbox_inches='tight')

plt.show()

# --- Construcción del DataFrame ---

# Diccionario para asociar estación → lista de líneas que pasan por ella
estacion_lineas = {}
for color, datos in sistema_generico["lineas"].items():
    for estacion in datos["estaciones"]:
        estacion_lineas.setdefault(estacion, []).append(color)

# Crear lista con los datos para el DataFrame
data = []
for estacion, (lat, lon) in pos.items():
    lineas = estacion_lineas.get(estacion, [])
    data.append({
        "estacion": estacion,
        "latitud": lat,
        "longitud": lon,
        "lineas": ", ".join(lineas) if lineas else "Ninguna"
    })

# Crear el DataFrame
df = pd.DataFrame(data)

# Mostrarlo (opcional)
print(df)

# Guardar en CSV
ruta_salida = os.path.join(os.path.dirname(__file__), "estaciones_sistema.csv")
df.to_csv(ruta_salida, index=False, encoding='utf-8-sig')

print("\nArchivo 'estaciones_sistema.csv' guardado correctamente.")

