import json
import matplotlib.pyplot as plt
import networkx as nx
from fontTools.ttLib.woff2 import bboxFormat

sistema_generico= {
    "amarilla": {
        "sentido_uno": [
            "AA1SC1SC",
            "AB2SC2SC",
            "AC3SC3SC",
            "AD4RF4SC",
            "AE5VE5SC",
            "AF6SC6SC",
            "AG7BH7SC"
        ],
        "sentido_dos": [
            "AG7BH7SC",
            "AF6SC6SC",
            "AE5VE5SC",
            "AD4RF4SC",
            "AC3SC3SC",
            "AB2SC2SC",
            "AA1SC1SC"
        ]
    },
    "azul": {
        "sentido_uno": [
            "BA1SC1SC",
            "BB2OC2SC",
            "BC3SC3SC",
            "BD2VB2SC",
            "BE4RC4SC",
            "BF5SC5SC",
            "BG6SC6SC",
            "AG7BH7SC"
        ],
        "sentido_dos": [
            "AG7BH7SC",
            "BG6SC6SC",
            "BF5SC5SC",
            "BE4RC4SC",
            "BD2VB2SC",
            "BC3SC3SC",
            "BB2OC2SC",
            "BA1SC1SC"
        ]
    },
    "roja": {
        "sentido_uno": [
            "RA1SC1SC",
            "RB2SC2SC",
            "BE4RC4SC",
            "RD3VC3OD",
            "RE5SC5SC",
            "AD4RF4SC",
            "RG6SC6SC"
        ],
        "sentido_dos": [
            "RG6SC6SC",
            "AD4RF4SC",
            "RE5SC5SC",
            "RD3VC3OD",
            "BE4RC4SC",
            "RB2SC2SC",
            "RA1SC1SC"
        ]
    },
    "verde": {
        "sentido_uno": [
            "VA1SC1SC",
            "BD2VB2SC",
            "RD3VC3OD",
            "VD4SC4SC",
            "AE5VE5SC",
            "VF6SC6SC"
        ],
        "sentido_dos": [
            "VF6SC6SC",
            "AE5VE5SC",
            "VD4SC4SC",
            "RD3VC3OD",
            "BD2VB2SC",
            "VA1SC1SC"
        ]
    },
    "naranja": {
        "sentido_uno": [
            "OA1SC1SC",
            "OB2SC2SC",
            "BB2OC2SC",
            "OC3SC3SC",
            "RD3VC3OD"
        ],
        "sentido_dos": [
            "RD3VC3OD",
            "OC3SC3SC",
            "BB2OC2SC",
            "OB2SC2SC",
            "OA1SC1SC"
        ]
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
for color, info in sistema_generico.items():
    estaciones = info["sentido_uno"]
    for i in range(len(estaciones)-1):
        G.add_edge(estaciones[i], estaciones[i+1], color=color_map[color])

# Posiciones manuales para evitar superposición
pos = {
    "AA1SC1SC": (-40, 20),
    "AB2SC2SC": (50, -40),
    "AC3SC3SC": (120, 0),
    "AD4RF4SC": (180, 40),
    "AE5VE5SC": (240, 60),
    "AF6SC6SC": (320, 100),
    "AG7BH7SC": (400, 120),
    "BA1SC1SC": (-120, 180),
    "BB2OC2SC": (-60, 200),
    "BC3SC3SC": (0, 220),
    "BD2VB2SC": (60, 240),
    "BE4RC4SC": (120, 220),
    "BF5SC5SC": (200, 180),
    "BG6SC6SC": (290, 160),
    "RA1SC1SC": (240, 260),
    "RB2SC2SC": (180, 300),
    "RD3VC3OD": (100, 160),
    "RE5SC5SC": (120, 80),
    "RG6SC6SC": (120, -120),
    "VA1SC1SC": (0, 360),
    "VD4SC4SC": (180, 120),
    "VF6SC6SC": (260, 0),
    "OA1SC1SC": (-160, 380),
    "OB2SC2SC": (-120, 280),
    "OC3SC3SC": (0, 80)
}

# Dibujamos el grafo
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=700,
    node_color="lightgray",
    edge_color=colors,
    width=3,
    font_size=4.5,
    font_weight=600
)
#Guardamos la figura
plt.savefig("sistema_generico.png", dpi=600,bbox_inches='tight')

plt.show()
