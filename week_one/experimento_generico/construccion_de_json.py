import json
import matplotlib.pyplot as plt
import networkx as nx
from fontTools.ttLib.woff2 import bboxFormat

sistema_generico= {
    "lineas":{
        "amarilla": {
            "estaciones": ["AA1SCI","AB2SC","AC3SC","AD4RF","AE5VE","AF7BH"],
            "sentido_ida":["AA1SCI","AB2SC","AC3SC","AD4RF","AE5VE","AF7BH"],
            "sentido_vuelta":["AF7BH","AE5VE","AD4RF","AC3SC","AB2SC","AA1SCI"]
        },
        "azul": {
            "estaciones": ["BA1SC","BB8SC","BC3SC","BD2VB","BE4RC","BF5SC","BG6SC","AF7BH"],
            "sentido_ida":["BA1SC","BB2SC","BC3SC","BD2VB","BE4RC","BF5SC","BG6SC","AF7BH"],
            "sentido_vuelta":["AF7BH","BG6SC","BF5SC","BE4RC","BD2VB","BC3SC","BB2SC","BA1SC"]
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
        }


    }
}

#Guardamos el sistema en un archivo JSON
#with open("sistema_generico.json", "w", encoding="utf-8") as f:
    #json.dump(sistema_generico, f, ensure_ascii=False, indent=4)

# Mapeo de colores de español a matplotlib
color_map = {
    "amarilla": "yellow",
    "azul": "blue",
    "roja": "red",
    "verde": "green"
}

# Creamos un grafo
G = nx.Graph()

# Añadimos nodos y aristas por línea
for color, info in sistema_generico["lineas"].items():
    estaciones = info["sentido_ida"]
    for i in range(len(estaciones)-1):
        G.add_edge(estaciones[i], estaciones[i+1], color=color_map[color])

# Posiciones manuales para evitar superposición
pos = {
    "AA1SCI": (0,0), "AB2SC": (3,-2), "AC3SC": (6,0), "AD4RF": (9,2), "AE5VE": (12,3), "AF7BH": (15,6),
    "BA1SC": (-6,9), "BB2SC": (-3,10), "BC3SC": (0,11), "BD2VB": (3,12), "BE4RC": (6,11), "BF5SC": (10,9), "BG6SC": (12,8),
    "RA1SC": (12,13), "RB2SC": (9,15), "RD3VC": (5,8), "RE5SC": (6,3), "RG6SC": (6,-6),
    "VA1SC": (0,18), "VD4SC": (9,6), "VF6SC": (13,0)
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

