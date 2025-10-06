import json
import time
import re
from typing import List, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch  # Para HF LLM

# Opcional: Para fallback a Ollama
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama no disponible; usando solo HF para embeddings y LLM.")

# =============================================================================
# CONFIGURACIÓN MODULAR DEL SISTEMA
# =============================================================================

# Configuración de proveedores de embeddings y modelos
EMBEDDING_PROVIDER = 'hf'  # Opciones: 'hf' (HuggingFace) o 'ollama'
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'  # Modelo para embeddings

# Configuración del LLM (Language Model)
USE_HF_FOR_LLM = True  # True: Usa HuggingFace Transformers; False: Usa Ollama
LANGUAGE_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'

# =============================================================================
# INICIALIZACIÓN DE MODELOS (LAZY LOADING)
# =============================================================================

# Variables globales para los modelos (se inicializan bajo demanda)
embedding_model = None
reranker = None
llm_tokenizer = None
llm_model = None
llm_pipeline = None
use_hf_llm_fallback = USE_HF_FOR_LLM  # Variable separada para el fallback


def init_models():
    """
    Inicializa todos los modelos de manera perezosa (lazy loading).
    Gestiona tanto embeddings como modelos de lenguaje grandes.
    """
    global embedding_model, reranker, llm_tokenizer, llm_model, llm_pipeline, use_hf_llm_fallback

    # Inicializar modelo de embeddings
    if embedding_model is None:
        if EMBEDDING_PROVIDER == 'hf':
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"Inicializado embedding HF: {EMBEDDING_MODEL}")
        else:
            if not OLLAMA_AVAILABLE:
                raise ValueError("Ollama no instalado para embeddings.")
            print(f"Inicializado embedding Ollama: {EMBEDDING_MODEL}")

    # Inicializar reranker para reordenar resultados
    if reranker is None:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Inicializado reranker.")

    # Inicializar LLM (HuggingFace u Ollama)
    if use_hf_llm_fallback:
        if llm_pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

                # Cargar tokenizer y modelo
                llm_tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
                llm_model = AutoModelForCausalLM.from_pretrained(
                    LANGUAGE_MODEL,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map='auto' if torch.cuda.is_available() else None
                )

                # Configurar token de padding si no existe
                if llm_tokenizer.pad_token is None:
                    llm_tokenizer.pad_token = llm_tokenizer.eos_token

                # Crear pipeline de generación de texto
                llm_pipeline = pipeline(
                    'text-generation',
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    device_map='auto' if torch.cuda.is_available() else None,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=llm_tokenizer.eos_token_id
                )
                print(f"Inicializado LLM HF: {LANGUAGE_MODEL}")
            except Exception as e:
                print(f"Error cargando HF LLM: {e}. Cambiando a Ollama si disponible.")
                # Cambiar a Ollama como fallback
                use_hf_llm_fallback = False


# =============================================================================
# DATASET DE METRO (DATOS DUROS)
# =============================================================================

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

# =============================================================================
# BASE DE DATOS VECTORIAL
# =============================================================================

VECTOR_DB: List[Tuple[str, List[float], dict]] = []
index = None


def get_embedding(text: str) -> List[float]:
    """
    Genera embeddings para un texto usando el proveedor configurado.

    Args:
        text (str): Texto a convertir en embedding

    Returns:
        List[float]: Vector de embedding
    """
    if EMBEDDING_PROVIDER == 'hf':
        return embedding_model.encode(text).tolist()
    else:
        if not OLLAMA_AVAILABLE:
            raise ValueError("Ollama requerido para embeddings.")
        return ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]


def create_granular_chunks(data):
    """
    Divide el dataset en chunks granulares para mejor recuperación.

    Args:
        data: Diccionario con datos del metro

    Returns:
        list: Lista de tuplas (texto_chunk, metadata)
    """
    chunks = []

    for line_name, line_data in data["lineas"].items():
        estaciones = line_data['estaciones']

        # Chunk general por línea
        chunk_general = f"Línea {line_name.capitalize()}: Estaciones: {', '.join(estaciones)}. Ida: {', '.join(line_data['sentido_ida'])}. Vuelta: {', '.join(line_data['sentido_vuelta'])}."
        chunks.append((chunk_general, {'type': 'linea_general', 'linea': line_name}))

        # Segmentos adyacentes entre estaciones
        for sentido, direction in [('sentido_ida', 'ida'), ('sentido_vuelta', 'vuelta')]:
            stations_order = line_data[sentido]
            for i in range(len(stations_order) - 1):
                from_stat = stations_order[i]
                to_stat = stations_order[i + 1]
                chunk_seg = f"En Línea {line_name} ({direction}): De {from_stat} a {to_stat} (adyacentes)."
                chunks.append((chunk_seg, {
                    'type': 'segmento',
                    'linea': line_name,
                    'from': from_stat,
                    'to': to_stat,
                    'direction': direction
                }))

        # Información de transbordos
        for stat in estaciones:
            connected_lines = [
                ln for ln, ln_data in data["lineas"].items()
                if stat in ln_data['estaciones'] and ln != line_name
            ]
            if connected_lines:
                chunk_trans = f"Transbordo en {stat}: Conecta Línea {line_name} con {', '.join(connected_lines)}."
                chunks.append((chunk_trans, {
                    'type': 'transbordo',
                    'station': stat,
                    'lines': [line_name] + connected_lines
                }))

    print(f'Generados {len(chunks)} chunks granulares')
    return chunks


def build_vector_db():
    """
    Construye la base de datos vectorial con FAISS.
    Se ejecuta una vez al inicio del programa.
    """
    global VECTOR_DB, index
    init_models()  # Asegura que los modelos estén cargados

    chunks_with_meta = create_granular_chunks(dataset)
    VECTOR_DB = []
    embeddings_list = []

    # Procesar cada chunk
    for chunk_text, metadata in chunks_with_meta:
        embedding = get_embedding(chunk_text)
        VECTOR_DB.append((chunk_text, embedding, metadata))
        embeddings_list.append(embedding)

    # Crear índice FAISS
    embeddings_matrix = np.array(embeddings_list).astype('float32')
    d = embeddings_matrix.shape[1]  # Dimensionalidad
    index = faiss.IndexFlatIP(d)  # Índice para similitud coseno
    faiss.normalize_L2(embeddings_matrix)  # Normalizar para cosine similarity
    index.add(embeddings_matrix)

    print(f'DB construida con {len(VECTOR_DB)} vectores (dim: {d}).')


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos vectores.

    Args:
        vec1, vec2: Vectores a comparar

    Returns:
        float: Score de similitud (0-1)
    """
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve(query: str, top_n: int = 7, similarity_threshold: float = 0.5) -> List[Tuple[str, float, dict]]:
    """
    Recupera chunks relevantes usando FAISS + reranking.

    Args:
        query (str): Consulta del usuario
        top_n (int): Número máximo de resultados
        similarity_threshold (float): Umbral mínimo de similitud

    Returns:
        List: Chunks relevantes con sus metadatos
    """
    start_time = time.time()

    # Embedding de la consulta
    query_embedding = np.array([get_embedding(query)]).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Búsqueda en FAISS
    distances, indices = index.search(query_embedding, top_n * 2)

    # Recuperar candidatos
    candidates = []
    for i in indices[0]:
        if i < len(VECTOR_DB):
            chunk, emb, meta = VECTOR_DB[i]
            sim = cosine_similarity(query_embedding[0], np.array(emb))
            candidates.append((chunk, sim, meta))

    # Reranking con cross-encoder
    pairs = [(query, cand[0]) for cand in candidates]
    if pairs:
        rerank_scores = reranker.predict(pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        filtered = [
            (chunk, sim, meta) for (chunk, sim, meta), score in reranked
            if sim > similarity_threshold
        ][:top_n]

        print(f'Retrieval completado en {time.time() - start_time:.2f}s ({len(filtered)} chunks).')
        return filtered

    return []


def extract_stations_from_query(query: str) -> Tuple[str, str]:
    """
    Extrae estaciones de origen y destino de la consulta usando regex.

    Args:
        query (str): Consulta del usuario

    Returns:
        Tuple: (origen, destino)
    """
    station_pattern = r'\b[A-Z]{2}\d+[A-Z]{2,3}\b'
    stations = re.findall(station_pattern, query)
    return (
        stations[0] if stations else "origen desconocido",
        stations[1] if len(stations) > 1 else "destino desconocido"
    )


def post_process_response(response: str, retrieved_knowledge: List[Tuple[str, float, dict]], dataset) -> str:
    """
    Verifica y corrige posibles alucinaciones del modelo.

    Args:
        response (str): Respuesta generada por el LLM
        retrieved_knowledge: Chunks utilizados para la respuesta
        dataset: Dataset completo para verificación

    Returns:
        str: Respuesta verificada y corregida
    """
    # Obtener todas las estaciones válidas
    all_stations = set()
    for line_data in dataset["lineas"].values():
        all_stations.update(line_data['estaciones'])

    # Encontrar estaciones mencionadas en la respuesta
    mentioned_stations = set(re.findall(r'\b[A-Z]{2}\d+[A-Z]{2,3}\b', response))
    invalid_stations = mentioned_stations - all_stations

    # Si hay estaciones inválidas, agregar advertencia
    if invalid_stations:
        return f"Advertencia: Posible alucinación (estaciones inválidas: {invalid_stations}). Respuesta ajustada: {response[:200]}..."

    return response


def generate_with_llm(instruction_prompt: str, user_query: str, num_generations: int = 2) -> str:
    """
    Genera respuestas usando el LLM configurado (HF u Ollama).

    Args:
        instruction_prompt (str): Prompt del sistema
        user_query (str): Consulta del usuario
        num_generations (int): Número de generaciones para self-consistency

    Returns:
        str: Mejor respuesta generada
    """
    init_models()
    responses = []

    if use_hf_llm_fallback:
        # Generación con HuggingFace Transformers
        full_prompt = f"{instruction_prompt}\n\n:User  {user_query}\nAssistant:"

        for _ in range(num_generations):
            gen_start = time.time()
            try:
                output = llm_pipeline(
                    full_prompt,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7
                )[0]['generated_text']
                response = output[len(full_prompt):].strip()  # Extraer solo la respuesta
            except Exception as e:
                response = f"Error en generación HF: {e}"

            citation_count = len(re.findall(r'Chunk \d+', response))
            responses.append((response, citation_count))
            print(f'Generación HF completada en {time.time() - gen_start:.2f}s.')
    else:
        # Generación con Ollama (fallback)
        if not OLLAMA_AVAILABLE:
            raise ValueError("Ollama requerido cuando USE_HF_FOR_LLM=False.")

        for _ in range(num_generations):
            gen_start = time.time()
            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': user_query}
                ],
                stream=True,
            )
            response = ''.join([chunk['message']['content'] for chunk in stream])
            citation_count = len(re.findall(r'Chunk \d+', response))
            responses.append((response, citation_count))
            print(f'Generación Ollama completada en {time.time() - gen_start:.2f}s.')

    # Self-consistency: Elegir la respuesta con más citas (o más larga)
    best_response = max(responses, key=lambda x: (x[1], len(x[0])))[0]
    return best_response


def generate_response(query: str, retrieved_knowledge: List[Tuple[str, float, dict]]) -> str:
    """
    Genera una respuesta usando RAG (Retrieval Augmented Generation).

    Args:
        query (str): Consulta del usuario
        retrieved_knowledge: Chunks recuperados relevantes

    Returns:
        str: Respuesta generada
    """
    if not retrieved_knowledge:
        return "No hay información suficiente en los chunks para calcular la ruta. Sugiero consultar el mapa oficial del metro."

    # Extraer estaciones de origen y destino
    origen, destino = extract_stations_from_query(query)

    # Formatear el conocimiento recuperado
    knowledge_content = '\n'.join([
        f'Chunk {i + 1} (sim: {sim:.2f}, tipo: {meta.get("type", "general")}): {chunk}'
        for i, (chunk, sim, meta) in enumerate(retrieved_knowledge)
    ])

    # Prompt detallado para el LLM
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
    Respuesta: Directo en Azul (ida): BA1SC → BB2OC → BC3SC → BD2VB → BE4RC → BF5SC → BG6SC → AG7BH. 0 transbordos.

    EJEMPLO 3 (ruta no posible):
    Pregunta: ¿De OA1SC a VF6SC sin conexión?
    Razonamiento: Chunks no muestran transbordos entre Naranja y Verde directamente (solo vía Azul o Roja, pero no explícito aquí).
    Respuesta: Ruta no posible con info disponible. Sugiero consultar el mapa oficial para conexiones indirectas.

    Pregunta: {query}
    Responde de forma clara, concisa y útil al usuario.
    '''

    # Generar respuesta usando el LLM
    raw_response = generate_with_llm(instruction_prompt, query)

    # Post-procesar para verificar alucinaciones
    return post_process_response(raw_response, retrieved_knowledge, dataset)


# =============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    # Construir la base de datos vectorial (una vez al inicio)
    build_vector_db()

    # Query de ejemplo
    input_query = "¿Quiero ir de RD3VC a AG7BH? ¿Que líneas debo coger, y cuantos transbordos habrá?"

    # Fase de retrieval: Buscar información relevante
    retrieved_knowledge = retrieve(input_query, top_n=7, similarity_threshold=0.5)

    print("\nChunks relevantes recuperados:")
    for i, (chunk, sim, meta) in enumerate(retrieved_knowledge):
        print(f"Chunk {i + 1} (sim: {sim:.4f}, tipo: {meta.get('type', 'general')}): {chunk[:100]}...")

    # Fase de generación: Crear respuesta usando RAG
    final_response = generate_response(input_query, retrieved_knowledge)

    print("\nRespuesta del Chatbot (RAG mejorado):")
    print(final_response)