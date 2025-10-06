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


def generate_with_llm(instruction_prompt: str, user_query: str, num_generations: int = 1) -> str:
    """
    Genera respuestas usando el LLM configurado con mejor control.
    """
    init_models()
    responses = []

    # PROMPT MEJORADO: Más claro y con instrucciones específicas de formato
    clean_prompt = f"""{instruction_prompt}

INSTRUCCIONES CRÍTICAS:
- Responde ÚNICAMENTE con la ruta específica solicitada
- NO repitas el ejemplo del prompt
- NO agregues notas adicionales o disclaimers
- Si la ruta es posible: indica estaciones, líneas y transbordos claramente
- Si no es posible: di simplemente "Ruta no posible con información disponible"
- FINALIZA después de dar la respuesta principal

Pregunta: {user_query}
Respuesta:"""

    if use_hf_llm_fallback:
        for i in range(num_generations):
            gen_start = time.time()
            try:
                # PARÁMETROS CORREGIDOS para HuggingFace
                output = llm_pipeline(
                    clean_prompt,
                    max_new_tokens=300,  # Reducido para evitar repeticiones
                    do_sample=True,
                    temperature=0.4,  # Temperatura más baja para más consistencia
                    repetition_penalty=1.3,  # Penalizar repeticiones
                    no_repeat_ngram_size=3,  # Evitar repetición de n-gramas
                    pad_token_id=llm_tokenizer.eos_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id
                )[0]['generated_text']

                response = output[len(clean_prompt):].strip()

                # LIMPIAR respuesta inmediatamente
                response = clean_generated_response(response, user_query)

            except Exception as e:
                print(f"Error en generación HF: {e}")
                response = "Error en la generación de respuesta."

            # Calcular calidad de la respuesta
            quality_score = calculate_response_quality(response, user_query)
            responses.append((response, quality_score))
            print(f'Generación HF {i + 1} completada en {time.time() - gen_start:.2f}s. Calidad: {quality_score}')

    else:
        # Código para Ollama
        if not OLLAMA_AVAILABLE:
            return "Error: Ollama no disponible."

        for i in range(num_generations):
            gen_start = time.time()
            try:
                response = ollama.chat(
                    model=LANGUAGE_MODEL,
                    messages=[
                        {'role': 'system', 'content': instruction_prompt},
                        {'role': 'user', 'content': user_query}
                    ],
                    options={
                        'temperature': 0.4,
                        'num_predict': 300,
                        'repeat_penalty': 1.3
                    }
                )['message']['content']

                response = clean_generated_response(response, user_query)

            except Exception as e:
                print(f"Error en generación Ollama: {e}")
                response = "Error en la generación de respuesta."

            quality_score = calculate_response_quality(response, user_query)
            responses.append((response, quality_score))
            print(f'Generación Ollama {i + 1} completada en {time.time() - gen_start:.2f}s. Calidad: {quality_score}')

    # Elegir la mejor respuesta basada en calidad
    if responses:
        best_response = max(responses, key=lambda x: x[1])[0]
        return best_response
    return "No se pudo generar una respuesta válida."


def clean_generated_response(response: str, user_query: str) -> str:
    """
    Limpia la respuesta generada eliminando repeticiones y contenido no deseado.
    """
    if not response:
        return "No se pudo generar respuesta."

    # Eliminar cualquier mención de ejemplos o chunks del prompt
    forbidden_patterns = [
        r"ejemplo\s+\d+", r"Ejemplo\s+\d+", r"NOTA:", r"Nota:",
        r"Chunk\s+\d+", r"chunk\s+\d+", r"razonamiento", r"Razonamiento",
        r"según.*chunk", r"basándome.*chunk", r"citar.*chunk"
    ]

    for pattern in forbidden_patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE)

    # Eliminar repeticiones de líneas
    lines = response.split('\n')
    unique_lines = []
    seen_lines = set()

    for line in lines:
        clean_line = line.strip()
        # Solo agregar líneas no vacías y no repetidas
        if clean_line and len(clean_line) > 5 and clean_line not in seen_lines:
            unique_lines.append(clean_line)
            seen_lines.add(clean_line)

    cleaned = '\n'.join(unique_lines)

    # Eliminar repeticiones de frases al final
    sentences = re.split(r'[.!?]+', cleaned)
    if len(sentences) > 1:
        # Mantener solo oraciones únicas
        unique_sentences = []
        seen_sentences = set()
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence and clean_sentence not in seen_sentences:
                unique_sentences.append(clean_sentence)
                seen_sentences.add(clean_sentence)

        cleaned = '. '.join(unique_sentences) + '.' if unique_sentences else cleaned

    # Cortar en puntos naturales de finalización
    stop_indicators = [
        "Ruta no posible con información disponible",
        "Total:", "Transbordos:", "Líneas:", "Ruta:",
        "Directo en", "Toma", "Consulta el mapa"
    ]

    for indicator in stop_indicators:
        if indicator in cleaned:
            idx = cleaned.find(indicator)
            # Tomar desde el indicador hasta el final del párrafo
            remaining = cleaned[idx:]
            # Encontrar el final del párrafo
            paragraph_end = remaining.find('\n\n')
            if paragraph_end != -1:
                cleaned = cleaned[:idx + paragraph_end]
                break

    # Limpiar espacios múltiples y formato
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def calculate_response_quality(response: str, user_query: str) -> float:
    """
    Calcula la calidad de una respuesta basada en varios factores.
    """
    if not response or len(response) < 10:
        return 0.0

    score = 0.0

    # Puntos por contener información relevante
    if any(word in response for word in ["línea", "estación", "transbordo", "ruta"]):
        score += 2.0

    # Puntos por mencionar estaciones de la consulta
    query_stations = re.findall(r'[A-Z]{2}\d+[A-Z]{2,3}', user_query)
    mentioned_stations = sum(1 for station in query_stations if station in response)
    score += mentioned_stations * 1.0

    # Penalizar por repeticiones
    words = response.split()
    if len(words) > 2:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        score += repetition_ratio * 2.0

    # Penalizar respuestas demasiado largas
    if len(response) > 500:
        score -= 1.0

    # Bonus por formato estructurado
    if any(marker in response for marker in ["→", "Líneas:", "Transbordos:"]):
        score += 1.0

    return max(0.0, score)


def generate_response(query: str, retrieved_knowledge: List[Tuple[str, float, dict]]) -> str:
    """
    Genera una respuesta usando RAG con mejor control.
    """
    if not retrieved_knowledge:
        return "No hay información suficiente para calcular la ruta. Consulta el mapa oficial del metro."

    origen, destino = extract_stations_from_query(query)

    # KNOWLEDGE CONTENT MÁS SIMPLE
    knowledge_content = '\n'.join([
        f'{chunk}'
        for chunk, sim, meta in retrieved_knowledge
    ])

    # PROMPT MÁS DIRECTO Y SIMPLE
    instruction_prompt = f'''Información del metro disponible:
{knowledge_content}

Basándote SOLO en la información anterior, responde esta pregunta:
¿Cómo ir de {origen} a {destino} en metro?

Responde de forma CONCISA y DIRECTA:
- Si hay ruta: describe la ruta específica con líneas y transbordos
- Si no hay ruta: di "Ruta no posible con información disponible"
- NO des ejemplos
- NO repitas información
- NO expliques tu razonamiento'''

    raw_response = generate_with_llm(instruction_prompt, query, num_generations=1)

    # Post-procesamiento final
    return post_process_response(raw_response, retrieved_knowledge, dataset)


def post_process_response(response: str, retrieved_knowledge: List[Tuple[str, float, dict]], dataset) -> str:
    """
    Verificación final de la respuesta.
    """
    # Verificar estaciones válidas
    all_stations = set()
    for line_data in dataset["lineas"].values():
        all_stations.update(line_data['estaciones'])

    mentioned_stations = set(re.findall(r'[A-Z]{2}\d+[A-Z]{2,3}', response))
    invalid_stations = mentioned_stations - all_stations

    if invalid_stations:
        return f"Información: Algunas estaciones mencionadas no existen en el mapa actual. Consulta el mapa oficial para rutas precisas."

    # Si la respuesta es muy corta o genérica, proporcionar una respuesta por defecto
    if len(response) < 20 or "error" in response.lower():
        origen, destino = extract_stations_from_query("")  # Extraer de algún modo
        return f"Para ir de {origen} a {destino}, consulta el mapa oficial del metro o usa la aplicación oficial para rutas en tiempo real."

    return response

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