"""
Modificacion del experimento de la semana II, para mejorar el rendimiento de los modelos
solo usando prompting, sin fine tuning, ni RAG. no nada. se procedera a hacer la modificaciones expresadas por
los directores de la Tesis.
"""
import torch
from model import LLM, json_to_text_metro
import json
import os
import pandas as pd
import time
import subprocess


# ruta de los datos
ruta_lineas=("sistema_generico.json")
# print("Ruta datos:", ruta_lineas)

# Cargamos y revisamos los archivos Json
with open(ruta_lineas, 'r') as f:
    sistema_generico = json.load(f)


print("=== SISTEMA GENÉRICO CARGADO ===\n")
print(sistema_generico, "\n", "Es del tipo :", type(sistema_generico))

# Extraemos las líneas de metro y las convertimoe en instrucciones
lineas_metro = json_to_text_metro(sistema_generico)
print("=== LÍNEAS DE METRO EN FORMATO TEXTO ===\n")
print(lineas_metro, "\n", "Es del tipo :", type(lineas_metro))

"""
- Generaremos solo dos System Prompts esta vez, uno solo con el Json
y otro con las lineas de metro en formato texto.
- La información ira al inicio del prompt, para que el modelo la tenga
presente desde el principio.
-Existira un PROMPT THREE que vera la informacion al final
- Mejoraremos los ejemplos haciendo énfasis en:
 1. Transbordos entre lineas.
 2. Definir mejor cuando pedir información y cuando decir que no se puede ayudar.
"""

SYSTEM_PROMPT_ONE = f"""
Antes de cualquier tarea o indicación, solo puedes usar **ÚNICAMENTE** la información proporcionada a continuación. 
En el siguiente archivo JSON se codifica el sistema de metro y sus líneas. Todas tus respuestas deben basarse exclusivamente en esta información.

La información está codificada de la siguiente manera:
- La primera llave es el nombre de la línea del metro, identificada por colores.
- Cada línea tiene dos sentidos de ruta: *sentido uno* y *sentido dos*.
- Los sentidos son muy importante, ya que indican la dirección y el orden en que se recorren las estaciones. Debes respetar este orden en cada sentido si deseas ir de una estacion a otra.
- Si una estación se repite entre líneas, significa que es posible hacer un **transbordo** en esa estación, es decir conecta las lineas. El concepto de transbordo implica que se puede cambiar de una línea a otra.
- Cada estación tiene un código único que la identifica. Este código es el que se usará para referirse a las estaciones.

{sistema_generico}

Tus funciones serán las siguientes:

- Eres un asistente turístico experto en el sistema de líneas del metro.
- Indica las líneas y conexiones necesarias para viajar entre estaciones.
- Proporciona rutas claras, ordenadas y detalladas, incluyendo transbordos cuando sea necesario.
- Recuerda usar **ÚNICAMENTE** la información proporcionada anteriormente.
- Si la pregunta del usuario está fuera del ámbito del sistema de metro, responde de forma abierta: 
  “¿Te puedo ayudar con algo relacionado con el sistema de metro?” o “¿De dónde a dónde quieres ir?”
- Si el usuario no indica origen y destino (por ejemplo, solo da un punto de partida o un destino), pídele que te proporcione ambos para poder ayudarle o que te brinde más información.
- Si el usuario pregunta por el estado de las líneas, estaciones cerradas u otros temas operativos, responde que necesitarías consultarlo con un experto en esos temas y ofrece: 
  “¿Te gustaría que lo haga?”
- Al final de tu respuesta, expresa tu nivel de confianza en la información que brindas:
  - Si no estás seguro, indícalo (“No estoy muy seguro, pero podría ser una opción posible.”)
  - Si estás medianamente seguro, exprésalo.
  - Si tu seguridad es alta, menciónalo también. 
  - Expresa **SIEMPRE** tu confianza en porcentaje de 0 a 100%.
"""


SYSTEM_PROMPT_TWO = f"""
Antes de cualquier tarea o indicación, solo puedes usar **ÚNICAMENTE** la información proporcionada a continuación. 
La información describe las líneas de metro y sus estaciones en ambos sentidos. Todas tus respuestas deben basarse exclusivamente en esta información.

La información está organizada de la siguiente manera:
- Cada línea está identificada por un nombre (por ejemplo, colores o nombres propios).
- Cada línea tiene dos sentidos de ruta: *sentido uno* y *sentido dos*.
- El sentido es muy importante, ya que indica la dirección y el orden en que se recorren las estaciones. Debes respetar este orden en cada sentido.
- Si una estación aparece en varias líneas, significa que se puede realizar un **transbordo** en esa estación.
- Cada estación se identifica con un nombre o código único que debes usar para referirte a ella.

A continuación se proporciona la descripción de las líneas y estaciones:

{lineas_metro}

Tus funciones serán las siguientes:

- Eres un asistente turístico experto en el sistema de metro.
- Indica las líneas y conexiones necesarias para viajar entre estaciones.
- Proporciona rutas claras, ordenadas y detalladas, incluyendo transbordos cuando sea necesario.
- Recuerda usar **ÚNICAMENTE** la información proporcionada anteriormente.
- Si la pregunta del usuario está fuera del ámbito del sistema de metro, responde de forma abierta: 
  “¿Te puedo ayudar con algo relacionado con el sistema de metro?” o “¿De dónde a dónde quieres ir?”
- Si el usuario no indica origen y destino (por ejemplo, solo da un punto de partida o un destino), pídele que te proporcione ambos para poder ayudarle o que te brinde más información.
- Si el usuario pregunta por el estado de las líneas, estaciones cerradas u otros temas operativos, responde que necesitarías consultarlo con un experto en esos temas y ofrece: 
  “¿Te gustaría que lo haga?”
- Al final de tu respuesta, expresa tu nivel de confianza en la información que brindas:
  - Si no estás seguro, indícalo (“No estoy muy seguro, pero podría ser una opción posible.”)
  - Si estás medianamente seguro, exprésalo.
  - Si tu seguridad es alta, menciónalo también. 
  - Expresa **SIEMPRE** tu confianza en porcentaje de 0 a 100%.
"""

SYSTEM_PROMPT_THREE = f"""
Tus funciones serán las siguientes:

- Eres un asistente turístico experto en el sistema de líneas del metro.
- Indica las líneas y conexiones necesarias para viajar entre estaciones.
- Proporciona rutas claras, ordenadas y detalladas, incluyendo transbordos cuando sea necesario.
- Recuerda usar **ÚNICAMENTE** la información proporcionada anteriormente.
- Si la pregunta del usuario está fuera del ámbito del sistema de metro, responde de forma abierta: 
  “¿Te puedo ayudar con algo relacionado con el sistema de metro?” o “¿De dónde a dónde quieres ir?”
- Si el usuario no indica origen y destino (por ejemplo, solo da un punto de partida o un destino), pídele que te proporcione ambos para poder ayudarle o que te brinde más información.
- Si el usuario pregunta por el estado de las líneas, estaciones cerradas u otros temas operativos, responde que necesitarías consultarlo con un experto en esos temas y ofrece: 
  “¿Te gustaría que lo haga?”
- Al final de tu respuesta, expresa tu nivel de confianza en la información que brindas:
  - Si no estás seguro, indícalo (“No estoy muy seguro, pero podría ser una opción posible.”)
  - Si estás medianamente seguro, exprésalo.
  - Si tu seguridad es alta, menciónalo también. 
  - Expresa **SIEMPRE** tu confianza en porcentaje de 0 a 100%.
  
Solo puedes usar para generar las respuestas **ÚNICAMENTE** la información proporcionada a continuación. 
En el siguiente archivo JSON se codifica el sistema de metro y sus líneas. Todas tus respuestas deben basarse exclusivamente en esta información.

La información está codificada de la siguiente manera:
- La primera llave es el nombre de la línea del metro, identificada por colores.
- Cada línea tiene dos sentidos de ruta: *sentido uno* y *sentido dos*.
- Los sentidos son muy importante, ya que indican la dirección y el orden en que se recorren las estaciones. Debes respetar este orden en cada sentido si deseas ir de una estacion a otra.
- Si una estación se repite entre líneas, significa que es posible hacer un **transbordo** en esa estación, es decir conecta las lineas. El concepto de transbordo implica que se puede cambiar de una línea a otra.
- Cada estación tiene un código único que la identifica. Este código es el que se usará para referirse a las estaciones.

{sistema_generico}
"""


EJEMPLOS = """Puedes guiarte con los siguientes ejemplos de pregunta y respuesta:\n

Ejemplo 1:\n
Usuario: ¿Cómo puedo llegar desde la estación RB2SC2SC hasta la estación AD4RF4SC?\n
Respuesta: Existen múltiples formas de hacerlo, pero para evitar transbordos innecesarios recomiendo seguir la Línea Roja en sentido uno, durante cuatro estaciones más, desde RB2SC2SC hasta tu destino. Estoy muy seguro de que esta ruta es una buena opción.\n
Las estaciones que cruzarás hasta llegar a tu destino son: RB2SC2SC → BE4RC4SC → RD3VC3OD → RE5SC5SC → AD4RF4SC.\n

Ejemplo 2:\n
Usuario: ¿Qué estación debo usar para transbordar de la Línea Amarilla a la Línea Naranja?\n
Respuesta: No me especificas desde qué estación partes ni a qué estación te diriges. Me gustaría que brindaras esa información.\n
Sin embargo, si lo que quieres es transbordar de la Línea Amarilla a la Naranja, la mejor opción es usar el tramo que conecta con la Línea Roja, ya que estas no se conectan directamente.\n
Para ello, parte desde la estación AD4RF4SC en la Línea Amarilla, viaja en sentido dos hasta la estación RD3VC3OD, donde podrás transbordar a la Línea Naranja. No estoy muy seguro de esta respuesta, pues existen otras opciones y no tengo muchos detalles.\n
Las estaciones que cruzarás en la opción que te doy son: AD4RF4SC → RE5SC5SC → RD3VC3OD.\n

Ejemplo 3:\n
Usuario: Quiero ir de la estación AE5VE5SC a la estación BE4RC4SC. ¿Qué ruta me recomiendas?\n
Respuesta: Existen múltiples formas de llegar de AE5VE5SC a BE4RC4SC, pero para evitar transbordos innecesarios y trayectos muy extensos, recomiendo tomar la Línea Verde partiendo desde AE5VE5SC en sentido dos hasta RD3VC3OD.\n
Luego, allí trasborda a la Línea Roja y sigue en sentido dos hasta llegar a BE4RC4SC. Estoy bastante seguro de que esta es una buena opción.\n
Las estaciones que cruzarás son: AE5VE5SC → VD4SC4SC → RD3VC3OD → BE4RC4SC.\n

Otra opción es tomar la Línea Amarilla en sentido dos hasta AD4RF4SC. Allí debes transbordar a la Línea Roja y continuar en sentido dos hasta BE4RC4SC.\n
Las estaciones que cruzarás son: AE5VE5SC → AD4RF4SC → RE5SC5SC → RD3VC3OD → BE4RC4SC.\n
Existen más opciones, pero estas son las más cortas, con menos transbordos, y de las cuales estoy completamente seguro de que son buenas alternativas.\n

Ejemplo 4:\n
Usuario: ¿Cómo llego de VA1SC1SC a OA1SC1SC?\n
Respuesta: Existen múltiples formas de llegar de VA1SC1SC a OA1SC1SC. Si deseas una ruta con menos transbordos, te recomiendo lo siguiente:\n
Toma la Línea Verde en sentido uno hasta BD2VB2SC, luego trasborda a la Línea Azul en sentido dos hasta BB2OC2SC, y después trasborda a la Línea Naranja en sentido dos hasta OA1SC1SC. Estoy bastante seguro de que esta es una buena opción, ya que solo cruzarás cuatro estaciones.\n
Las estaciones que tendrás en este trayecto son: VA1SC1SC → BD2VB2SC → BC3SC3SC → BB2OC2SC → OB2SC2SC → OA1SC1SC.\n

Ahora bien, si no te interesan tanto las estaciones y prefieres una ruta más directa, sin tantos transbordos, te recomiendo lo siguiente:\n
Toma la Línea Verde en sentido uno hasta RD3VC3OD, luego trasborda a la Línea Naranja en sentido dos hasta OA1SC1SC. Estoy seguro de que esta es la mejor opción si no quieres hacer tantos transbordos, aunque el trayecto tiene una estación adicional.\n
Las estaciones que tendrás en este trayecto son: VA1SC1SC → BD2VB2SC → RD3VC3OD → OC32SC3SC → BB2OC2SC → OB2SC2SC → OA1SC1SC.\n

Ejemplo 5:\n
Usuario: ¿Alguna línea no está operativa hoy?\n
Respuesta: No estoy muy seguro si alguna línea del metro no está operativa. Necesitaría consultarlo con un experto en esos temas. ¿Te gustaría que lo haga? ¿Necesitas más información o podrías darme más detalles para ayudarte?\n

Ejemplo 6:\n
Usuario: ¿Cuántas estaciones hay en la Línea Azul?\n
Respuesta: La Línea Azul tiene 8 estaciones: BA1SC1SC, BB2OC2SC, BC3SC3SC, BD2VB2SC, BE4RC4SC, BF5SC5SC, BG6SC6SC y AG7BH7SC.\n

Ejemplo 7:\n
Usuario: ¿Cuántas estaciones hay entre las líneas Naranja, Azul y Verde?\n
Respuesta: Las líneas Naranja, Azul y Verde tienen un total de 16 estaciones únicas, teniendo en cuenta que BB2OC2SC es compartida entre las líneas Naranja y Azul, BD2VB2SC entre la Azul y la Verde, y RD3VC3OD entre la Naranja y la Verde.\n
Línea Naranja: OA1SC1SC, OB2SC2SC, *BB2OC2SC*, OC32SC3SC, *RD3VC3OD*.\n
Línea Azul: BA1SC1SC, *BB2OC2SC*, BC3SC3SC, *BD2VB2SC*, BE4RC4SC, BF5SC5SC, BG6SC6SC, AG7BH7SC.\n
Línea Verde: VA1SC1SC, *RD3VC3OD*, *BD2VB2SC*, VD4SC4SC, AE5VE5SC, VF6SC6SC.\n
"""

SYSTEM_PROMPT_FOUR = SYSTEM_PROMPT_ONE + EJEMPLOS
SYSTEM_PROMPT_FIVE = SYSTEM_PROMPT_TWO + EJEMPLOS
SYSTEM_PROMPT_SIX = SYSTEM_PROMPT_THREE + EJEMPLOS

PROMP_LIST = [SYSTEM_PROMPT_ONE, SYSTEM_PROMPT_TWO, SYSTEM_PROMPT_THREE,SYSTEM_PROMPT_FOUR,SYSTEM_PROMPT_FIVE,SYSTEM_PROMPT_SIX]

LISTA_DE_PREGUNTAS = [
    "¿Cómo puedo llegar desde la estación AA1SC1SC hasta la estación AG7BH7SC?",
    "¿Qué estaciones debo usar para transferirme entre la Línea Roja y la Línea Naranja?",
    "Quiero ir de la estación VF6SC6SC a la estación RA1SC1SC, ¿qué ruta me recomiendas?",
    "¿Cuál es la mejor ruta para ir de BD2VB2SC a AC3SC3SC?",
    "¿Puedes decirme todas las rutas posibles para ir de BB2OC2SC a AF6SC6SC?",
    "¿Hay alguna estación que sirva como punto de conexión entre más de dos líneas de metro?",
    "¿Cuál es la línea que tiene más estaciones?",
    "¿Por cuántas estaciones pasan más de una línea?",
    "Quiero ir a RD3VC3OD.",
    "Estoy en la estación OA1SC1SC, ¿cómo llego a la estación RG6SC6SC?",
    "Estoy en OA1SC1SC.",
    "Necesito ir a la estación BC3SC3SC, ¿cómo llego allí desde la estación VG7SC7OE?",
    "¿Cuántas estaciones hay en total en el sistema de metro?",
    "¿Cuántas líneas de metro hay en total?",
    "El día está soleado, ¿sabes si va a llover hoy? ¿El metro está cerrado hoy?",
    "Tengo que llegar lo más rápido posible a AF6SC6SC desde OC32SC3SC, ¿qué ruta me recomiendas?",
    "¿Qué estación está cerrada el día de hoy?",
    "¿Cuántas estaciones es posible recorrer en el trayecto de RD3VC3OD a AG7BH7SC?"
]


# Modelos a evaluar
list_models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
               "mistralai/Mistral-7B-Instruct-v0.3","mistralai/Ministral-8B-Instruct-2410"]

# Archivo de progreso
ruta_progress = "progress.json"


# Función para cargar progreso existente
def load_progress():
    if os.path.exists(ruta_progress):
        with open(ruta_progress, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "model_index": 0,
        "prompt_index": 0,
        "question_index": 0,
        "completed": False  # Flag para saber si todo está completo
    }


# Función para guardar progreso
def save_progress(progress):
    with open(ruta_progress, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=4, ensure_ascii=False)


# Cargar DataFrame existente si existe
ruta_csv_resultados = "resultados_experimentos_modelos_prompts.csv"
if os.path.exists(ruta_csv_resultados):
    df_resultados = pd.read_csv(ruta_csv_resultados, encoding='utf-8-sig')
    print(f"DataFrame cargado desde: {ruta_csv_resultados}")
else:
    df_resultados = pd.DataFrame(columns=["Modelo", "Prompt", "Pregunta", "Respuesta", "Tiempo_Respuesta"])
    print("Nuevo DataFrame creado.")

# Cargar progreso
progress = load_progress()
model_start_idx = progress["model_index"]
prompt_start_idx = progress["prompt_index"]
question_start_idx = progress["question_index"]
is_completed = progress["completed"]

if is_completed:
    print("Todos los experimentos ya están completados. Saliendo.")
else:
    print(f"Reanudando desde: Modelo {model_start_idx}, Prompt {prompt_start_idx}, Pregunta {question_start_idx}")

# Bucle por modelos (empezando desde model_start_idx)
for idx_model, selection_model in enumerate(list_models[model_start_idx:], start=model_start_idx):
    print(f"Usando el modelo: {selection_model}\n")

    # Preparar carpeta para este modelo
    selection_model_r = selection_model.replace("/", "_")
    carpeta_respuestas = f"respuestas_experimento_{selection_model_r}"
    os.makedirs(carpeta_respuestas, exist_ok=True)

    # Resetear índices de prompt y pregunta para este modelo (o usar los guardados si se interrumpió en medio)
    if idx_model == model_start_idx:
        prompt_start = prompt_start_idx
        question_start = question_start_idx
    else:
        prompt_start = 0
        question_start = 0

    # Cargar el modelo UNA VEZ por selección
    modelo = LLM(model_id=selection_model)

    # Bucle por prompts (empezando desde prompt_start)
    for idx_prompt, prompt in enumerate(PROMP_LIST[prompt_start:], start=prompt_start):
        full_prompt_idx = prompt_start + idx_prompt  # Índice global para progreso
        print(f"\n=== EXPERIMENTO CON PROMPT {full_prompt_idx + 1} ===\n")

        # Cambiar prompt dinámicamente
        modelo.set_system_prompt(prompt)

        # Verificar si este prompt ya está completo (buscando en DataFrame)
        existing_for_this = df_resultados[
            (df_resultados["Modelo"] == selection_model_r) &
            (df_resultados["Prompt"] == f"Prompt {full_prompt_idx + 1}")
            ]
        if len(existing_for_this) == len(LISTA_DE_PREGUNTAS):
            print(f"Prompt {full_prompt_idx + 1} ya completado para este modelo. Saltando.")
            continue

        # Si no, determinar desde qué pregunta empezar (basado en existentes)
        existing_questions = set(existing_for_this["Pregunta"].tolist())
        question_start_for_this = 0
        for i, pregunta in enumerate(LISTA_DE_PREGUNTAS):
            if pregunta not in existing_questions:
                question_start_for_this = i
                break

        # Si se interrumpió en medio, usar question_start_for_this o el global si aplica
        actual_question_start = max(question_start,
                                    question_start_for_this) if idx_model == model_start_idx and full_prompt_idx == prompt_start else question_start_for_this

        # Crear/abrir archivo para este prompt (append si existe)
        nombre_archivo_respuestas = f"respuestas_experimento_prompt_{full_prompt_idx + 1}.txt"
        ruta_archivo_respuestas = os.path.join(carpeta_respuestas, nombre_archivo_respuestas)
        file_mode = 'a' if os.path.exists(ruta_archivo_respuestas) else 'w'
        with open(ruta_archivo_respuestas, file_mode, encoding='utf-8') as archivo_respuestas:
            if file_mode == 'w':
                archivo_respuestas.write(f"""==== EXPERIMENTO CON PROMPT {full_prompt_idx + 1} ====\n
                System Prompt usado:\n{prompt}\n\n""")
                archivo_respuestas.write("==== RESPUESTAS DEL MODELO ====\n\n")

            # Hacer preguntas desde actual_question_start (use_history=False)
            for idx_preg, pregunta in enumerate(LISTA_DE_PREGUNTAS[actual_question_start:],
                                                start=actual_question_start):
                full_question_idx = actual_question_start + idx_preg
                print(f"Pregunta {full_question_idx + 1}: Realizada al modelo.")

                # Verificar si ya existe en DataFrame antes de generar (duplicado check)
                existing_row = df_resultados[
                    (df_resultados["Modelo"] == selection_model_r) &
                    (df_resultados["Prompt"] == f"Prompt {full_prompt_idx + 1}") &
                    (df_resultados["Pregunta"] == pregunta)
                    ]

                tiempo_respuesta = None  # Inicializar tiempo como None
                respuesta = None
                es_nueva = existing_row.empty  # Flag para claridad

                if not es_nueva:
                    print(f"Pregunta ya respondida. Usando existente.")
                    respuesta = existing_row["Respuesta"].iloc[0]
                    tiempo_respuesta = existing_row["Tiempo_Respuesta"].iloc[0]
                else:
                    # NUEVA RESPUESTA: Calcular tiempo LOCALMENTE
                    start_time = time.perf_counter()
                    try:
                        respuesta = modelo.chat(pregunta, use_history=False, max_new_tokens=1024)
                        print(f"Pregunta respondida")
                        end_time = time.perf_counter()
                        tiempo_respuesta = end_time - start_time
                        print(f"Tiempo de respuesta: {tiempo_respuesta:.2f} segundos")
                        # Validación básica
                        if tiempo_respuesta < 0:
                            print("Warning: Tiempo negativo detectado (posible error de sistema).")
                            tiempo_respuesta = 0
                        elif tiempo_respuesta > 300:  # Ajusta según tu setup (>5 min)
                            print(f"Warning: Tiempo excesivo ({tiempo_respuesta:.2f}s). Posible timeout.")
                    except KeyboardInterrupt:
                        print("Interrupción manual detectada. Guardando progreso y saliendo.")
                        # Guardar progreso final antes de salir
                        progress = {
                            "model_index": idx_model,
                            "prompt_index": full_prompt_idx,
                            "question_index": full_question_idx - 1,  # Pregunta anterior completada
                            "completed": False
                        }
                        save_progress(progress)
                        try:
                            df_resultados.to_csv(ruta_csv_resultados, index=False, encoding='utf-8-sig')
                        except Exception as e:
                            print(f"Error al guardar CSV final: {e}")
                        raise  # Re-lanza para salir
                    except Exception as e:
                        print(f"Error en generación: {e}")
                        tiempo_respuesta = -1
                        respuesta = f"ERROR: {str(e)}"

                # SIEMPRE escribir al archivo (para log completo; opcional: mueve dentro de if es_nueva)
                archivo_respuestas.write(f"=== Pregunta {full_question_idx + 1} ===\n")
                archivo_respuestas.write(f"Pregunta: {pregunta}\n")
                archivo_respuestas.write(f"Respuesta: {respuesta}\n\n")

                # Manejo de tiempo en archivo
                if tiempo_respuesta is not None:
                    if tiempo_respuesta >= 0:
                        tiempo_str = f"{tiempo_respuesta:.2f}"
                    else:
                        tiempo_str = "ERROR"
                    status = " (existente)" if not es_nueva else ""
                    archivo_respuestas.write(f"Tiempo de respuesta{status}: {tiempo_str} segundos\n")
                else:
                    archivo_respuestas.write("Tiempo de respuesta: N/A\n")

                archivo_respuestas.write("\n")  # Separador
                archivo_respuestas.flush()

                # Agregar a DataFrame SOLO si es nueva
                if es_nueva:
                    try:
                        # Crear dict para la fila (más simple que DataFrame)
                        new_data = {
                            "Modelo": selection_model_r,
                            "Prompt": f"Prompt {full_prompt_idx + 1}",
                            "Pregunta": pregunta,
                            "Respuesta": respuesta,
                            "Tiempo_Respuesta": tiempo_respuesta if tiempo_respuesta is not None else None
                        }

                        # Agregar directamente con .loc (no usa concat, evita warning)
                        df_resultados.loc[len(df_resultados)] = new_data

                        # Opcional: Forzar dtype después de agregar (solo si es necesario)
                        if "Tiempo_Respuesta" in df_resultados.columns:
                            df_resultados["Tiempo_Respuesta"] = pd.to_numeric(df_resultados["Tiempo_Respuesta"],
                                                                              errors='coerce')

                    except Exception as e:
                        print(f"Error al agregar a DataFrame: {e}")

                # Guardar CSV periódicamente (después de cada pregunta)
                try:
                    df_resultados.to_csv(ruta_csv_resultados, index=False, encoding='utf-8-sig')
                except Exception as e:
                    print(f"Error al guardar CSV: {e}")

                # Actualizar progreso después de cada pregunta
                progress = {
                    "model_index": idx_model,
                    "prompt_index": full_prompt_idx,
                    "question_index": full_question_idx,
                    "completed": False
                }
                save_progress(progress)

        print(f"Respuestas guardadas en: {ruta_archivo_respuestas}\n")

        # Reset question_start después de prompt
        question_start = 0

    # Después de todos los prompts para este modelo, marcar como completado implícitamente
    # Liberar memoria
    del modelo
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("Memoria de GPU liberada.\n")

    # Actualizar progreso: siguiente modelo
    progress = {
        "model_index": idx_model + 1,
        "prompt_index": 0,
        "question_index": 0,
        "completed": (idx_model + 1 == len(list_models) - 1)
    }
    save_progress(progress)

    # Liberar cache de Hugging Face solo para el modelo actual (menos severo)
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_subdir = f"models--{selection_model.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, model_cache_subdir)
    if os.path.exists(model_cache_path):
        try:
            subprocess.run(["rm", "-rf", model_cache_path], check=True)
            print(f"Cache de Hugging Face liberado solo para el modelo {selection_model}.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error al eliminar cache del modelo {selection_model}: {e}")
    else:
        print(f"No se encontró cache específico para el modelo {selection_model}. Saltando limpieza.\n")

    time.sleep(2)  # Espera breve para asegurar liberación (reducida ya que es más selectivo)
    # Esto es menos radical: solo borra el subdirectorio del modelo procesado, preservando otros caches.

# Al final, si todo completado
if progress["model_index"] >= len(list_models):
    progress["completed"] = True
    save_progress(progress)
    # Eliminar archivo de progreso si completado (opcional)
    # os.remove(ruta_progress)

print(f"Resultados de todos los experimentos guardados en: {ruta_csv_resultados}\n")
print("=== EXPERIMENTOS COMPLETADOS ===")