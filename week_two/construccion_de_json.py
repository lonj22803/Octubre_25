"""
Version 2.0 del sistema de transporte generico
Autor: Juan Jose Londoño Cardenas
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from fontTools.ttLib.woff2 import bboxFormat

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
with open("sistema_generico.json", "w", encoding="utf-8") as f:
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

# Posiciones manuales para evitar superposición
"""
pos = {
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
"""
pos = {
    "AA1SC": (40.376800, -3.683800),
    "AB2SC": (40.466800, -3.743800),
    "AC3SC": (40.536800, -3.703800),
    "AD4RF": (40.596800, -3.663800),
    "AE5VE": (40.656800, -3.643800),
    "AF6SC": (40.736800, -3.603800),
    "AG7BH": (40.816800, -3.583800),
    "BA1SC": (40.296800, -3.523800),
    "BB2OC": (40.356800, -3.503800),
    "BC3SC": (40.416800, -3.483800),
    "BD2VB": (40.476800, -3.463800),
    "BE4RC": (40.536800, -3.483800),
    "BF5SC": (40.616800, -3.523800),
    "BG6SC": (40.706800, -3.543800),
    "RA1SC": (40.656800, -3.443800),
    "RB2SC": (40.596800, -3.403800),
    "RD3VC": (40.516800, -3.543800),
    "RE5SC": (40.536800, -3.623800),
    "RG6SC": (40.536800, -3.823800),
    "VA1SC": (40.416800, -3.343800),
    "VD4SC": (40.596800, -3.583800),
    "VF6SC": (40.676800, -3.703800),
    "OA1SC": (40.256800, -3.323800),
    "OB2SC": (40.296800, -3.423800),
    "OC3SC": (40.416800, -3.623800),
}


# Dibujamos el grafo
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=650,
    node_color="lightgray",
    edge_color=colors,
    width=5,
    font_size=6,
    font_weight=600
)
#Guardamos la figura
plt.savefig("sistema_generico.png", dpi=300,bbox_inches='tight')

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
df.to_csv("estaciones_sistema.csv", index=False, encoding='utf-8-sig')

print("\nArchivo 'estaciones_sistema.csv' guardado correctamente.")

