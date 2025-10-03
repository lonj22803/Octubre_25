import json
import pprint
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import os

# Función para cargar el JSON desde un archivo o string
def load_chat_json(json_path: str = None, json_string: str = None) -> Dict[str, Any]:
    """
    Carga el JSON del chat desde un archivo o string.

    Args:
        json_path (str, optional): Ruta al archivo JSON.
        json_string (str, optional): String JSON directamente.

    Returns:
        Dict[str, Any]: El diccionario parseado del chat.
    """
    if json_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif json_string:
        data = json.loads(json_string)
    else:
        raise ValueError("Debe proporcionar json_path o json_string.")
    return data


# Función para imprimir el JSON de manera legible
def pretty_print_chat(data: Dict[str, Any]):
    """
    Imprime el chat de forma legible usando pprint.
    """
    print("=== INFORMACIÓN GENERAL DEL CHAT ===")
    print(f"Chat ID: {data.get('chat_id', 'N/A')}")
    print(f"Fecha de inicio: {data.get('start_time', 'N/A')}")
    print(f"Modelo: {data.get('model_id', 'N/A')}")
    print(f"\nSystem Prompt:\n{data.get('system_prompt', 'N/A')}")
    print("\n" + "=" * 50 + "\n")


# Función para extraer y mostrar el flujo de conversación completo
def display_conversation_flow(data: Dict[str, Any]):
    """
    Muestra el flujo completo de la conversación, reconstruyendo el historial
    para cada interacción.
    """
    print("=== FLUJO DE CONVERSACIÓN ===")
    system_prompt = data.get('system_prompt', '')
    interactions = data.get('interactions', {})

    full_history = [{"role": "system", "content": system_prompt}]

    for inter_key, inter_data in interactions.items():
        print(f"\n--- Interacción: {inter_key} ---")
        user_question = inter_data.get('user_question', '')
        history = inter_data.get('history', [])
        response = inter_data.get('response', '')

        # Mostrar historial de esta interacción
        print("Historial previo:")
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            print(f"  {role.upper()}: {content[:100]}...")  # Truncado para legibilidad

        # Pregunta del usuario
        print(f"\nPregunta del usuario: {user_question[:100]}...")

        # Respuesta del asistente
        print(f"\nRespuesta del asistente: {response[:100]}...")

        # Actualizar historial completo (agregar user y assistant para la siguiente)
        full_history.append({"role": "user", "content": user_question})
        full_history.append({"role": "assistant", "content": response})

    # Guardar historial completo en un archivo para depuración
    with open('full_conversation_history.json', 'w', encoding='utf-8') as f:
        json.dump(full_history, f, ensure_ascii=False, indent=2)
    print("\nHistorial completo guardado en 'full_conversation_history.json'")


# Función para analizar errores o patrones en respuestas (básico)
def analyze_responses(data: Dict[str, Any]):
    """
    Análisis básico de respuestas: longitud, palabras clave, posibles errores.
    En este caso, busca menciones de "error" o "lo siento" para depurar prompting.
    """
    print("=== ANÁLISIS DE RESPUESTAS ===")
    interactions = data.get('interactions', {})

    error_keywords = ['error', 'lo siento', 'disculpa', 'incorrecto']
    response_lengths = []
    error_interactions = []

    for inter_key, inter_data in interactions.items():
        response = inter_data.get('response', '').lower()
        length = len(response.split())
        response_lengths.append(length)

        if any(keyword in response for keyword in error_keywords):
            error_interactions.append(inter_key)
            print(f"Interacción {inter_key} contiene posible error: {response[:200]}...")

    print(f"\nLongitudes de respuestas: {response_lengths}")
    print(
        f"Promedio de palabras por respuesta: {sum(response_lengths) / len(response_lengths) if response_lengths else 0:.2f}")

    if error_interactions:
        print(f"Interacciones con posibles errores: {error_interactions}")
    else:
        print("No se detectaron errores obvios en las respuestas.")


# Función para visualizar el grafo de interacciones (usando networkx y matplotlib)
def visualize_interaction_graph(data: Dict[str, Any]):
    """
    Crea un grafo simple de interacciones: nodos para interacciones, aristas para flujo secuencial.
    Útil para ver el orden y dependencias en el chat.
    """
    print("=== VISUALIZACIÓN DEL GRAFO DE INTERACCIONES ===")
    interactions = list(data.get('interactions', {}).keys())

    G = nx.DiGraph()
    for i, inter in enumerate(interactions):
        G.add_node(inter, label=inter)
        if i > 0:
            prev = interactions[i - 1]
            G.add_edge(prev, inter, label="siguiente")

    # Dibujar el grafo
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrows=True)
    nx.draw_networkx_edge_labels(G, pos)
    plt.title("Grafo de Interacciones del Chat")
    plt.savefig('chat_interaction_graph.png')
    plt.show()
    print("Grafo guardado en 'chat_interaction_graph.png'")


# Función para extraer el esquema de líneas (del JSON inicial del usuario)
def extract_transport_schema(data: Dict[str, Any]):
    """
    Extrae y muestra el esquema de transporte del JSON inicial (primera interacción).
    Útil para depurar si el modelo entiende el esquema.
    """
    print("=== ESQUEMA DE TRANSPORTE EXTRAÍDO ===")
    first_inter = data.get('interactions', {}).get('interaccion_1', {})
    user_question = first_inter.get('user_question', '')

    try:
        # El esquema está en la primera pregunta como string JSON
        schema_start = user_question.find('{')
        schema_end = user_question.rfind('}') + 1
        schema_str = user_question[schema_start:schema_end]
        schema = json.loads(schema_str)

        lines = schema.get('lineas', {})
        for line_name, line_data in lines.items():
            print(f"\nLínea {line_name}:")
            print(f"  Estaciones: {line_data.get('estaciones', [])}")
            print(f"  Sentido Ida: {line_data.get('sentido_ida', [])}")
            print(f"  Sentido Vuelta: {line_data.get('sentido_vuelta', [])}")

        # Guardar esquema limpio
        with open('transport_schema.json', 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print("\nEsquema guardado en 'transport_schema.json'")

    except json.JSONDecodeError:
        print("No se pudo parsear el esquema de transporte.")


# Función principal para ejecutar todo el análisis
def analyze_chat(json_path: str = None, json_string: str = None):
    """
    Función principal: carga y analiza el chat completo.
    """
    data = load_chat_json(json_path, json_string)

    pretty_print_chat(data)
    display_conversation_flow(data)
    #analyze_responses(data)
    #extract_transport_schema(data)
    #visualize_interaction_graph(data)

    print("\nAnálisis completado. Archivos generados para depuración.")


# EJEMPLO DE USO:
# Si tienes el JSON en un archivo 'chat.json', usa:
# analyze_chat(json_path='chat.json')

# Si quieres usar el string directamente (pega el JSON aquí):
if __name__ == "__main__":
    path = "chat_20251003_152530.json"
    print("¿Existe el archivo?:", os.path.exists(path))
    print("Directorio actual:", os.getcwd())
    analyze_chat(json_path=path)
