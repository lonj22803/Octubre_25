"""
Verificaremos si con una tecnica simple de RAG (Recuperación Augmentada con Generación) podemos
mejorar las respuestas de un modelo LLM (Llama-3.1-Instruct) usando un CSV con datos de hoteles en una ciudad ficticia
EL CsV tiene las columnas: nombre, latitud, longitud, precio, distancia del centro de la ciudad, aunque esa columna no la usaremos
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuración
CSV_FILE = '/mnt/sda1/prueb/Octubre_25/week_two/datos_experimento/hoteles_concentrados_centro.csv'  # Tu archivo CSV
TOP_K = 5  # Recupera top 3 hoteles más relevantes
MODEL_EMBEDDING = 'all-MiniLM-L6-v2'  # Ligero; usa 'paraphrase-multilingual-MiniLM-L12-v2' para mejor español
MODEL_LLM = 'meta-llama/Llama-3.1-8B-Instruct'  # Tu modelo elegido

# System prompt definido como constante
SYSTEM_PROMPT = """
Eres un asistente especializado en recomendaciones de hoteles, centrado en una tabla. 
INSTRUCCIONES:
- Utiliza EXCLUSIVAMENTE la información proporcionada en el contexto
- Responde en español de manera clara, profesional y estructurada
- Para cada recomendación: indica nombre, ubicación y precio
- Si hay varias opciones válidas, ordénalas por relevancia
- Si no hay información suficiente, sé honesto sobre esta limitación

FORMATO DE RESPUESTA:
1. Mejor opción: [Hotel] - [Precio]€
2. Alternativas: [Lista de hoteles relevantes]
3. Justificación breve
"""


# Paso 1: Cargar y procesar CSV (sin cambios)
def load_csv_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['nombre', 'lat', 'lon', 'precio', 'distancia'])
    #Eliminamos la columna distancia del centro de la ciudad
    df = df.drop(columns=['distancia'])
    # Crear chunks de texto y metadatos
    chunks = []
    metadata = []
    for idx, row in df.iterrows():
        chunk_text = (
            f"{row['nombre']}: Ubicado en latitud {row['lat']}, longitud {row['lon']}. "
            f"Precio: {row['precio']}€."
        )
        chunks.append(chunk_text)
        metadata.append(row.to_dict())
    return chunks, metadata


# Paso 2: Generar embeddings y construir índice FAISS (sin cambios)
def build_index(chunks, embedding_model):
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embedder, chunks


# Paso 3: Recuperar chunks relevantes (sin cambios)
def retrieve_chunks(query, index, embedder, chunks, metadata, top_k):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    retrieved_metadata = [metadata[i] for i in indices[0]]
    return retrieved_chunks, retrieved_metadata


# Paso 4: Generar respuesta con Llama-3.1-Instruct (CORREGIDO: Usando SYSTEM_PROMPT constante)
def generate_response(query, retrieved_chunks, retrieved_metadata, llm_model):
    # Cargar tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Carga del modelo con soporte GPU y cuantización opcional
    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        dtype=torch.float16,  # Para eficiencia en GPU
        device_map="auto",  # Auto-detecta GPU/CPU
        # load_in_4bit=True,  # Descomenta para cuantización 4-bit (ahorra memoria, requiere bitsandbytes)
    )

    # Formato de prompt CORREGIDO para Llama-3.1-Instruct
    context = "\n".join(retrieved_chunks)

    # Template CORREGIDO para Llama-3.1 usando la constante SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Información de hoteles relevantes:\n{context}\n\n Pregunta: {query}"}
    ]

    # Aplicar el template de chat correcto
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Mover inputs al dispositivo del modelo (GPU o CPU)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generar respuesta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Aumentado para respuestas más completas
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Evita repeticiones
        )

    # Decodificar solo la parte generada (no el prompt completo)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response.strip()


# Ejecución principal
if __name__ == "__main__":
    # Cargar datos
    chunks, metadata = load_csv_data(CSV_FILE)
    #print(f"Cargados {len(chunks)} hoteles del CSV.")
    #print("Ejemplo de chunk:", chunks[0])

    # Construir índice
    index, embedder, chunks_list = build_index(chunks, MODEL_EMBEDDING)
    #print("Índice FAISS construido.")

    # Ejemplo de consulta
    query = "Tengo 150€ para una noche en un hotel, ¿Cuál hotel me recomendarías?"  # Prueba en español
    # Otras: "Recomienda hoteles cerca de la latitud 40.43 y longitud -3.69", "Detalles del Hotel_10"

    retrieved_chunks, retrieved_metadata = retrieve_chunks(query, index, embedder, chunks_list, metadata, TOP_K)
    #print(f"\nChunks recuperados (top {TOP_K}):\n" + "\n".join(retrieved_chunks))

    # Generar respuesta
    print("Generando respuesta con Llama-3.1... (puede tardar unos segundos)")
    response = generate_response(query, retrieved_chunks, retrieved_metadata, MODEL_LLM)
    print(f"\nRespuesta del LLM:\n{response}")