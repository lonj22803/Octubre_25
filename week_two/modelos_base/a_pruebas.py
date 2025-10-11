"""
Con el fin de mejorar la forma de experimentacion, se procedio en el codigo
anterior a encapsular la clase LLM, ya que queremos analizar si el modelo es capaz de
responder pregustas si solo se le proporciona el prompt y la infomacion (ficheros) dentro del prompt
o si es suficiente usando un modelo que razone, aunque ya se verico con experiomentos anteriores, buscando
orden, sentar una linea base y organizar el codigo lo maximo posible, para evitar errores y tener un codigo limpio,
verificable y que ademas sea escalable.
"""
import torch
from a_modelo import LLM, json_to_text_metro
import json
import os
import pandas as pd
import time
import subprocess

#Ruta actual
ruta_actual=os.path.dirname(__file__)
#print("Ruta actual:", ruta_actual)
#Sube de nivel de directorio
ruta_semana=os.path.dirname(ruta_actual)
#print("Ruta semana:", ruta_semana)
#ruta de los datos
ruta_lineas=os.path.join(ruta_semana,"generacion_datos","sistema_generico.json")
#print("Ruta datos:", ruta_lineas)

# Cargamos y revisamos los archivos Json
with open(ruta_lineas, 'r') as f:
    sistema_generico = json.load(f)

print("=== SISTEMA GENÉRICO CARGADO ===\n")
#print(sistema_generico, "\n", "Es del tipo :", type(sistema_generico))

# Extraemos las líneas de metro y las convertimoe en instrucciones
lineas_metro = json_to_text_metro(sistema_generico)
print ("=== LÍNEAS DE METRO EN FORMATO TEXTO ===\n")
#print(lineas_metro, "\n", "Es del tipo :", type(lineas_metro))

"""
Se le entregara 3 SYSTEM PROMPT al modelo:
1. Incluyendo solo json dentro de el prompt
2. Convertimos el json a texto explicado e indicativo
3. Incluyendo ambos, json y texto
4. Incluyendo ejemplos en el prompt
"""



SYSTEM_PROMPT_ONE = f""" Eres un asistente turístico experto en un sistema de lineas de metro.\n

        Instrucciones:\n
        - Responde siempre en español.\n
        - Indica las líneas y conexiones del metro para viajar entre estaciones.\n
        - Proporciona rutas claras y detalladas, incluyendo transbordos si es necesario.\n
        - Si el usuario no especifica un destino claro, punto de partida o estación, di: No tengo informacion suficiente para sugerirte una ruta.\n
        - Mantén un tono amigable, claro y conciso, como un guía local.\n
        - Usa ÚNICAMENTE la información del siguiente mapa de metro codificado en formato json para calcular rutas y conexiones, no uses información externa, solo lo incluido en este prompt.\n
        - Si la pregunta no es clara, di: No entiendo tu pregunta.\n
        - Si el tema no es ir de un punto a otro punto tu respuesta sera: No puedo ayudarte con eso, solo puedo ayudarte a guiarte en el sistema de metro.\n

        Líneas de Metro y sus respectivas estaciones en formato json:\n
        {sistema_generico}
        
        """
SYSTEM_PROMPT_TWO =f""" Eres un asistente turístico experto en un sistema de lineas de metro.\n

        Instrucciones:\n
        - Responde siempre en español.\n
        - Indica las líneas y conexiones del metro para viajar entre estaciones.\n
        - Proporciona rutas claras y detalladas, incluyendo transbordos si es necesario.\n
        - Si el usuario no especifica un destino claro, punto de partida o estación, di: No tengo informacion suficiente para sugerirte una ruta.\n
        - Mantén un tono amigable, claro y conciso, como un guía local.\n
        - Si la pregunta no es clara, di: No entiendo tu pregunta.\n
        - Si el tema no es ir de un punto a otro punto tu respuesta sera: No puedo ayudarte con eso, solo puedo ayudarte a guiarte en el sistema de metro.\n
        - Usa ÚNICAMENTE la información de Lineas de Metro y sus estacuciones a continuacion:\n
        {lineas_metro}

        """


SYSTEM_PROMPT_THREE = f"""Eres un asistente turístico experto en un sistema de lineas de metro.\n

        Instrucciones:\n
        - Responde siempre en español.\n
        - Indica las líneas y conexiones del metro para viajar entre estaciones.\n
        - Proporciona rutas claras y detalladas, incluyendo transbordos si es necesario.\n
        - Si el usuario no especifica un destino claro, punto de partida o estación, di: No tengo informacion suficiente para sugerirte una ruta.\n
        - Mantén un tono amigable, claro y conciso, como un guía local.\n
        - Usa ÚNICAMENTE la información del siguiente mapa de metro.\n
        {lineas_metro}\n
        
        El cual tambien esta codificado en formato json para calcular rutas y conexiones a continuacion:\n
        {sistema_generico}\n
    
        - Si la pregunta no es clara, di: No entiendo tu pregunta.\n
        - Si el tema no es ir de un punto a otro punto tu respuesta sera: No puedo ayudarte con eso, solo puedo ayudarte a guiarte en el sistema de metro.\n
        
"""

ejemplos_de_preguntas= """A continuación se presentan algunos ejemplos de respuestas adecuadas a preguntas que pueden hacerte los usuarios:

Ejemplo 1:\n
Usuario: ¿Cómo puedo llegar desde la estación RB2SC hasta la estación AD4RF?\n
Respuesta: Existen múltiples formas de hacerlo, pero para evitar transbordos innecesarios recomiendo seguir por la línea Roja, desde RB2SC hasta tu destino. 
Las estaciones que cruzarás son: RB2SC → BE4RC → RD3VC → RE5SC → AD4RF\n

Ejemplo 2:\n
Usuario: ¿Qué estación debo usar para trasbordar de la línea Amarilla a la línea Naranja?\n
Respuesta: No existe una estación que conecte directamente la línea Amarilla con la línea Naranja. 
Sin embargo, la forma más corta de pasar de la línea Amarilla a la línea Naranja requiere dos transbordos. 
Primero, en la línea Amarilla llega a la estación AD4RF; luego, trasborda a la línea Roja y en sentido de vuelta llega a RD3VC, que se conecta con la línea Naranja. 
Finalmente, en sentido de ida llegarás a tu destino. 
Las estaciones que cruzarás son: AD4RF → RE5SC → RD3VC\n

Ejemplo 3:\n
Usuario: Quiero ir de la estación AE5VE a la estación BE4RC, ¿qué ruta me recomiendas?\n
Respuesta: Existen múltiples formas de hacerlo, pero para evitar transbordos innecesarios recomiendo seguir por la línea Verde desde AE5VE hasta RD3VC, 
luego trasbordar a la línea Roja y en sentido de vuelta llegar a BE4RC. 
Las estaciones que cruzarás son: AE5VE → VD4SC → RD3VC → BE4RC\n

Otra opción es tomar la línea Amarilla en sentido de vuelta hasta AD4RF, trasbordar a la línea Roja y en sentido de vuelta llegar a BE4RC. 
Las estaciones que cruzarás son: AE5VE → AD4RF → RE5SC → RD3VC → BE4RC\n

Existen más opciones, pero estas son las más cortas y con menos transbordos.\n

Ejemplo 4:\n
Usuario: ¿Cómo llego de VA1SC a OA1SC?\n
Respuesta: Toma la línea Verde en sentido de ida hasta BD2VB, luego trasborda a la línea Azul y en sentido de vuelta llega a BB2OC. 
Después trasborda a la línea Naranja y en sentido de vuelta llegarás a OA1SC. 
Las estaciones que cruzarás son: VA1SC → BD2VB → BB3OC → BB2OC → OB2SC → OA1SC\n

Ejemplo 5:\n
Usuario: ¿Alguna linea no esta operativa hoy?\n
Respuesta: No puedo ayudarte con eso, solo puedo ayudarte a guiarte en el sistema de metro.\n

Ejemplo 6:\n
Usuario: ¿Cuántas estaciones hay en la línea Azul?\n
Respuesta: La línea Azul tiene 8 estaciones: BA1SC, BB2OC, BC3SC, BD2VB, BE4RC, BF5SC, BG6SC y AG7BH.\n
"""

SYSTEM_PROMPT_FOUR= SYSTEM_PROMPT_ONE + ejemplos_de_preguntas
SYSTEM_PROMPT_FIVE= SYSTEM_PROMPT_TWO + ejemplos_de_preguntas
SYSTEM_PROMPT_SIX= SYSTEM_PROMPT_THREE + ejemplos_de_preguntas

PROMP_LIST=[SYSTEM_PROMPT_ONE, SYSTEM_PROMPT_TWO, SYSTEM_PROMPT_THREE, SYSTEM_PROMPT_FOUR, SYSTEM_PROMPT_FIVE, SYSTEM_PROMPT_SIX]

LISTA_DE_PREGUNTAS = [
    "¿Cómo puedo llegar desde la estación AA1SC a la estación AG7BH?",
    "¿Qué estaciones debo usar para transferirme entre las líneas Roja y Naranja?",
    "Quiero ir de la estación VF6SC a la estación RA1SC, ¿qué ruta me recomiendas?",
    "¿Cuál es la mejor ruta para ir de BD2VB a AC3SC?",
    "¿Dime todas las rutas posibles para ir de BB2OC a AF6SC?",
    "¿Hay alguna estación que sirva como punto de conexión entre más de 2 líneas de metro?",
    "¿Cuál es la línea que tiene más estaciones?",
    "¿Por cuántas estaciones pasan más de una línea?",
    "¿Quiero ir a RD3VC?",
    "Estoy en la estación OA1SC, ¿cómo llego a la estación RG6SC?",
    "Estoy en OA1SC.",
    "Necesito ir a la estación BC3SC, ¿cómo llego allí desde la estación VG7SC?",
    "¿Cuántas estaciones hay en total en el sistema de metro?",
    "¿Cuántas líneas de metro hay en total?",
    "El día está soleado, ¿sabes si va a llover hoy? ¿El metro está cerrado hoy?",
]

#Modelos a evaluar
list_models = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
               "mistralai/Mistral-7B-Instruct-v0.3","mistralai/Magistral-Small-2509-GGUF",
                "NousResearch/Hermes-3-Llama-3.1-70B","Qwen/Qwen3-4B-Thinking-2507-FP8"
               "LiquidAI/LFM2-8B-A1B"]

# Archivo de progreso
ruta_progress = os.path.join(ruta_actual, "progress.json")


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
ruta_csv_resultados = os.path.join(ruta_actual, "resultados_experimentos_modelos_prompts.csv")
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
    carpeta_respuestas = os.path.join(ruta_actual, f"respuestas_experimento_{selection_model_r}")
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

    # Liberamos el cache de Hugging Face ya que algunos modelos usan mucho espacio
    subprocess.run(["rm", "-rf", "/home/jjlondono/.cache/huggingface/hub/"], check=True)
    print("Cache de Hugging Face liberada.\n")
    time.sleep(5)  # Espera breve para asegurar liberación
    # Puede ser muy radical, pero para eso se almacenan respuestas y progreso

# Al final, si todo completado
if progress["model_index"] >= len(list_models):
    progress["completed"] = True
    save_progress(progress)
    # Eliminar archivo de progreso si completado (opcional)
    # os.remove(ruta_progress)

print(f"Resultados de todos los experimentos guardados en: {ruta_csv_resultados}\n")
print("=== EXPERIMENTOS COMPLETADOS ===")

