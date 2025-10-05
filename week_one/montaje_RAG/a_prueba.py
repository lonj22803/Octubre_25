import json
import ollama

dataset_file = "sistema_generico.json"
with open(dataset_file, 'r') as f:
    data = json.load(f)
    print(f'Cargado {len(data)} registros desde {dataset_file}')

# Implementación de la base de datos vectorial
"""
Existen dos tipos de modelos: los de embedding (convierten texto en vectores) y los de lenguaje (generan texto).
"""

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Cada elemento en VECTOR_DB será una tupla (chunk, embedding), donde embedding es una lista de floats
VECTOR_DB = []

def add_chunk_to_database(chunk):
    # Obtener el embedding del chunk
    embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk
    )['embeddings'][0]  # Nota: es 'embeddings' (con 's'), no 'embedding'
    VECTOR_DB.append((chunk, embedding))

def create_line_chunks(data):
    chunks = []
    for line_name, line_data in data["lineas"].items():
        # Crear un chunk descriptivo por cada línea
        chunk_text = f"""
        Línea {line_name.capitalize()} del metro:
        - Estaciones: {', '.join(line_data['estaciones'])}
        - Sentido ida: {', '.join(line_data['sentido_ida'])}
        - Sentido vuelta: {', '.join(line_data['sentido_vuelta'])}
        - Total estaciones: {len(line_data['estaciones'])}
        """
        chunks.append(chunk_text.strip())
    return chunks

chunks = create_line_chunks(data)  # Usa 'data', no 'dataset_file'
for i, chunk in enumerate(chunks):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i + 1}/{len(chunks)} to the database')

# Función para similitud coseno (no necesitas import de sympy)
def cosine_similarity(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))  # Corrige variables
    norm_vec1 = sum(x ** 2 for x in vec1) ** 0.5
    norm_vec2 = sum(x ** 2 for x in vec2) ** 0.5
    return dot_product / (norm_vec1 * norm_vec2)

def retrieve(query, top_n=3):  # Corrige 'retireve'
    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )['embeddings'][0]  # Corrige clave

    # Calcular similitud coseno entre el embedding de la consulta y cada embedding en la base de datos
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))  # Orden correcto: (chunk, similarity)

    # Ordenar por similitud descendente (mayor similitud = más relevante)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    input_query = "¿Quiero ir de RD3VC a AG7BH? ¿Que lineas debo coger, y cuantos transbordos habra?"
    retrieved_knowledge = retrieve(input_query, top_n=3)

    print("Relevant Chunks:")
    for chunk, similarity in retrieved_knowledge:  # Orden correcto
        print(f"Similarity: {similarity:.4f}\nChunk: {chunk}\n")

    # Crear el contenido de conocimiento
    knowledge_content = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])

    instruction_prompt = f'''
    Eres un asistente experto en el sistema de metro de [Nombre de la Ciudad, ej. "Madrid"]. Responde preguntas sobre rutas, estaciones y trayectos de manera precisa y útil, basándote **únicamente** en la información proporcionada en los fragmentos de conocimiento a continuación. No inventes datos ni uses conocimiento externo.

    Instrucciones:
    - Construye el trayecto entre dos estaciones, respetando estrictamente el sentido de las líneas (ida o vuelta) y cuales lineas se comparten. No ignores las direcciones de las líneas.
    - Calcula el número de transbordos (cambios de línea) y las líneas utilizadas en la ruta más corta o eficiente.
    - Si no hay ruta posible con la información disponible, indícalo claramente y sugiere alternativas si aplica.
    - Estructura tu respuesta de forma clara: describe la ruta paso a paso, lista las estaciones intermedias, y resume transbordos y líneas al final.
    - Usa un lenguaje amigable y preciso. Si la pregunta no es sobre rutas, responde basándote solo en los fragmentos.

    {knowledge_content}

    Ejemplo de respuesta para una pregunta de ruta:
    Pregunta: ¿Cómo llego de la estación BB2SC a la estación AB2SC?

    Respuesta: Para ir de BB2SC a AB2SC, debes hacer 1 transbordo, ya que no están en la misma línea. Aquí está la ruta más corta utilizando las líneas disponibles:

    1. Toma la línea azul en sentido ida desde BB2SC (estación de partida) hasta la estación de transbordo AG7BH (estación común entre línea azul y amarilla).
    2. En AG7BH, cambia a la línea amarilla en sentido vuelta y continúa hasta AB2SC (estación de llegada).

    En total: 1 transbordo y 2 líneas (azul y amarilla).

    Trayecto completo de estaciones:
    - BB2SC → BC3SC → BD2VB → BE4RC → BF5SC → BG6SC → AG7BH (transbordo)
    - AG7BH → AF6SC → AE5VE → AD4RF → AC3SC → AB2SC
    ¡Buen viaje!
    '''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    # Imprimir la respuesta del chatbot en tiempo real
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print()  # Nueva línea al final

        
