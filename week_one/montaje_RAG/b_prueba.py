import json
import ollama
import faiss
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import re  # ¡AÑADIDO: Para regex en post-procesamiento y parsing de query!

# Carga el dataset (el mismo que proporcionaste)
dataset = {
    "lineas": {
        "amarilla": {
            "estaciones": ["AA1SC", "AB2SC", "AC3SC", "AD4RF", "AE5VE", "AF6SC", "AG7BH"],
            "sentido_ida": ["AA1SC", "AB2SC", "AC3SC", "AD4RF", "AE5VE", "AF6SC", "AG7BH"],
            "sentido_vuelta": ["AG7BH", "AF6SC", "AE5VE", "AD4RF", "AC3SC", "AB2SC", "AA1SC"]
        },
        "azul": {
            "estaciones": ["BA1SC", "BB2OC", "BC3SC", "BD2VB", "BE4RC", "BF5SC", "BG6SC", "AG7BH"],
            "sentido_ida": ["BA1SC", "BB2OC", "BC3SC", "BD2VB", "BE4RC", "BF5SC", "BG6SC", "AG7BH"],
            "sentido_vuelta": ["AG7BH", "BG6SC", "BF5SC", "BE4RC", "BD2VB", "BC3SC", "BB2OC", "BA1SC"]
        },
        "roja": {
            "estaciones": ["RA1SC", "RB2SC", "BE4RC", "RD3VC", "RE5SC", "AD4RF", "RG6SC"],
            "sentido_ida": ["RA1SC", "RB2SC", "BE4RC", "RD3VC", "RE5SC", "AD4RF", "RG6SC"],
            "sentido_vuelta": ["RG6SC", "AD4RF", "RE5SC", "RD3VC", "BE4RC", "RB2SC", "RA1SC"]
        },
        "verde": {
            "estaciones": ["VA1SC", "BD2VB", "RD3VC", "VD4SC", "AE5VE", "VF6SC"],
            "sentido_ida": ["VA1SC", "BD2VB", "RD3VC", "VD4SC", "AE5VE", "VF6SC"],
            "sentido_vuelta": ["VF6SC", "AE5VE", "VD4SC", "RD3VC", "BD2VB", "VA1SC"]
        },
        "naranja": {
            "estaciones": ["OA1SC", "OB2SC", "BB2OC"],
            "sentido_ida": ["OA1SC", "OB2SC", "BB2OC"],
            "sentido_vuelta": ["BB2OC", "OB2SC", "OA1SC"]
        }
    }
}
print(f'Cargado {len(dataset["lineas"])} líneas desde el dataset')

# Modelos: CAMBIADO a multilingual para mejor manejo de español
EMBEDDING_MODEL = 'nomic-embed-text'  # Multilingual y ligero en Ollama (o 'intfloat/multilingual-e5-large' si disponible)
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Base de datos vectorial
VECTOR_DB: List[Tuple[str, List[float], dict]] = []


def create_granular_chunks(data):
    """Chunking mejorado (mismo que antes)."""
    chunks = []
    all_stations = set()
    for line_name, line_data in data["lineas"].items():
        estaciones = line_data['estaciones']
        all_stations.update(estaciones)

        # Chunk por línea general
        chunk_general = f"Línea {line_name.capitalize()}: Estaciones: {', '.join(estaciones)}. Ida: {', '.join(line_data['sentido_ida'])}. Vuelta: {', '.join(line_data['sentido_vuelta'])}."
        chunks.append((chunk_general, {'type': 'linea_general', 'linea': line_name}))

        # Chunks por segmento adyacente
        for sentido, direction in [('sentido_ida', 'ida'), ('sentido_vuelta', 'vuelta')]:
            stations_order = line_data[sentido]
            for i in range(len(stations_order) - 1):
                from_stat = stations_order[i]
                to_stat = stations_order[i + 1]
                chunk_seg = f"En Línea {line_name} ({direction}): De {from_stat} a {to_stat} (adyacentes)."
                chunks.append((chunk_seg, {'type': 'segmento', 'linea': line_name, 'from': from_stat, 'to': to_stat,
                                           'direction': direction}))

        # Chunks por transbordos
        for stat in estaciones:
            connected_lines = [ln for ln, ln_data in data["lineas"].items() if
                               stat in ln_data['estaciones'] and ln != line_name]
            if connected_lines:
                chunk_trans = f"Transbordo en {stat}: Conecta Línea {line_name} con {', '.join(connected_lines)}."
                chunks.append(
                    (chunk_trans, {'type': 'transbordo', 'station': stat, 'lines': [line_name] + connected_lines}))

    print(f'Generados {len(chunks)} chunks granulares')
    return chunks


# Construir DB (mismo)
chunks_with_meta = create_granular_chunks(dataset)
for chunk_text, metadata in chunks_with_meta:
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk_text)['embeddings'][0]
    VECTOR_DB.append((chunk_text, embedding, metadata))

# FAISS Index (mismo)
embeddings_matrix = np.array([emb for _, emb, _ in VECTOR_DB]).astype('float32')
d = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(embeddings_matrix)
index.add(embeddings_matrix)

# Reranker (mismo)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve(query: str, top_n: int = 7, similarity_threshold: float = 0.5) -> List[
    Tuple[str, float, dict]]:  # Aumentado top_n, bajado umbral
    """Retrieval optimizado (mismo, pero con params ajustados para más cobertura)."""
    query_embedding = np.array([ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]]).astype('float32')
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_n * 2)

    candidates = [(VECTOR_DB[i][0], cosine_similarity(query_embedding[0], VECTOR_DB[i][1]), VECTOR_DB[i][2])
                  for i in indices[0] if i < len(VECTOR_DB)]
    pairs = [(query, cand[0]) for cand in candidates]
    if pairs:
        rerank_scores = reranker.predict(pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        filtered = [(chunk, sim, meta) for (chunk, sim, meta), score in reranked if sim > similarity_threshold][:top_n]
        return filtered
    return []


def extract_stations_from_query(query: str) -> Tuple[str, str]:
    """NUEVO: Extrae origen y destino con regex para robustez."""
    station_pattern = r'\b[A-Z]{2}\d+[A-Z]{2,3}\b'
    stations = re.findall(station_pattern, query)
    if len(stations) >= 2:
        return stations[0], stations[1]  # Asume primero origen, segundo destino
    return "origen desconocido", "destino desconocido"


def post_process_response(response: str, retrieved_knowledge: List[Tuple[str, float, dict]], dataset) -> str:
    """Verificación anti-alucinaciones (mismo, pero ahora con re importado)."""
    all_stations = set()
    for line_data in dataset["lineas"].values():
        all_stations.update(line_data['estaciones'])

    mentioned_stations = set(re.findall(r'\b[A-Z]{2}\d+[A-Z]{2,3}\b', response))
    invalid_stations = mentioned_stations - all_stations
    if invalid_stations:
        return f"Advertencia: Posible alucinación (estaciones inválidas: {invalid_stations}). Respuesta ajustada: {response.split('.')[0]}... Consulta mapa oficial."
    return response


def generate_response(query: str, retrieved_knowledge: List[Tuple[str, float, dict]]) -> str:
    """Genera respuesta (prompt ajustado con parsing robusto + mejor selección self-consistency)."""
    if not retrieved_knowledge:
        return "No hay información suficiente en los chunks para calcular la ruta. Sugiero consultar el mapa oficial del metro."

    origen, destino = extract_stations_from_query(query)
    knowledge_content = '\n'.join([f'Chunk {i + 1} (sim: {sim:.2f}, tipo: {meta.get("type", "general")}): {chunk}'
                                   for i, (chunk, sim, meta) in enumerate(retrieved_knowledge)])

    instruction_prompt = f'''
    Eres un asistente experto en el metro de Madrid. Responde **ÚNICAMENTE** basándote en los chunks proporcionados. No inventes estaciones, sentidos o conexiones. Cita chunks específicos en tu razonamiento (e.g., "Según Chunk 1..."). Si no hay ruta posible, di: "Ruta no posible con info disponible."

    PASOS OBLIGATORIOS (razona internamente paso a paso, pero responde claro al usuario):
    1. Identifica líneas y sentidos para origen ({origen}) y destino ({destino}).
    2. Encuentra transbordos vía estaciones compartidas en chunks.
    3. Construye ruta más corta: Lista estaciones paso a paso, respetando sentidos (ida/vuelta).
    4. Calcula transbordos (cambios de línea) y líneas usadas. Elige ruta eficiente (menos transbordos/estaciones).

    Conocimiento (cita chunks):
    {knowledge_content}

    EJEMPLO 1 (transbordo vía roja-amarilla):
    Pregunta: ¿De RD3VC a AG7BH?
    Razonamiento: Chunk X: RD3VC en Roja (ida a AD4RF). Chunk Y: AD4RF transbordo a Amarilla (ida a AG7BH). Ruta: RD3VC → RE5SC → AD4RF (Roja ida), transbordo → AE5VE → AF6SC → AG7BH (Amarilla ida). 1 transbordo, 2 líneas.
    Respuesta: Toma Roja (ida) de RD3VC a AD4RF. Transborda a Amarilla (ida) hasta AG7BH. Total: 1 transbordo.

    EJEMPLO 2 (misma línea, sin transbordo):
    Pregunta: ¿De BA1SC a AG7BH?
    Razonamiento: Chunk Z: Azul ida incluye BA1SC a AG7BH directamente.
    Respuesta: Directo en Azul (ida): BA1SC → BB2OC → ... → AG7BH. 0 transbordos.

    EJEMPLO 3 (ruta no posible):
    Pregunta: ¿De OA1SC a VF6SC sin conexión?
    Razonamiento: Chunks no muestran transbordos entre Naranja y Verde.
    Respuesta: Ruta no posible con info disponible.

    Pregunta: {query}
    '''

    # Self-consistency: Genera 2 respuestas y elige la que cite más chunks (mejor grounding)
    responses = []
    for i in range(2):
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'system', 'content': instruction_prompt}, {'role': 'user', 'content': query}],
            stream=True,
        )
        full_response = ''.join([chunk['message']['content'] for chunk in stream])
        # Cuenta citas simples (e.g., "Chunk X")
        citation_count = len(re.findall(r'Chunk \d+', full_response))
        responses.append((full_response, citation_count))

    # Selecciona la con más citas (o la más larga si empate)
    best_response = max(responses, key=lambda x: (x[1], len(x[0])))[0]
    return post_process_response(best_response, retrieved_knowledge, dataset)


# Ejemplo de uso
if __name__ == "__main__":
    input_query = "¿Quiero ir de BA1SC a VF6SC? ¿Que líneas debo coger, y cuantos transbordos habrá?"

    retrieved_knowledge = retrieve(input_query, top_n=7)
    print("Chunks relevantes recuperados:")
    for i, (chunk, sim, meta) in enumerate(retrieved_knowledge):
        print(f"Chunk {i + 1} (sim: {sim:.4f}, tipo: {meta.get('type')}): {chunk[:100]}...")

    final_response = generate_response(input_query, retrieved_knowledge)
    print("\nRespuesta del Chatbot (RAG mejorado):")
    print(final_response)

