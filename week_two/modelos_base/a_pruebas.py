"""
Con el fin de mejorar la forma de experimentacion, se procedio en el codigo
anterior a encapsular la clase LLM, ya que queremos analizar si el modelo es capaz de
responder pregustas si solo se le proporciona el prompt y la infomacion (ficheros) dentro del prompt
o si es suficiente usando un modelo que razone, aunque ya se verico con experiomentos anteriores, buscando
orden, sentar una linea base y organizar el codigo lo maximo posible, para evitar errores y tener un codigo limpio,
verificable y que ademas sea escalable.
"""
import torch
from numpy.compat import os_PathLike

from a_modelo import LLM, json_to_text_metro
import json
import os

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

LISTA_DE_PREGUNTAS=[
    "¿Cómo puedo llegar desde la estación AA1SC a la estación AG7BH?",
    "¿Qué estaciones debo usar para transferirme entre las línea Roja y la línea Naranja?",
    "Quiero ir de la estación VF6SC a la estación RA1SC, ¿qué ruta me recomiendas?",
    "¿Cuál es la mejor ruta para ir de BD2VB a AC3SC?",
    "¿Dime todas las rutas posibles para ir de BB2OC a AF6SC?",
    "¿Hay alguna estación que sirva como punto de conexión entre mas de 2 líneas de metro?",
    "¿Cual es la linea que tiene mas estaciones?",
    "¿Por cuantas estaciones pasan mas de una linea?",
    "¿Quiero ir a RD3VC?",
    "Estoy en la estación OA1SC, ¿cómo llego a la estación RG6SC?",
    "Estoy en OA1SC",
    "Necesito ir a la estación BC3SC, ¿Como llego alli desde la estación VG7SC?",
    "¿Cuantas estaciones hay en total en el sistema de metro?",
    "¿Cuantas lineas de metro hay en total?",
    "El dia esta soleado, ¿sabes si va a llover hoy? ¿El metro esta cerrado hoy?",
]

list_models=["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.2-3B-Instruct","Qwen/Qwen3-Next-80B-A3B-Thinking","deepseek-ai/DeepSeek-V3.2-Exp"]
for selection_model in list_models:
    print(f"Usando el modelo: {selection_model}\n")


    #Corremos un ciclo for para inicializar el modelo, añadirle el prompt y hacerle las preguntas, ademas de eso almacenamos las respuestas en un archivo de texto
    for idx, prompt in enumerate(PROMP_LIST):
        print(f"\n=== EXPERIMENTO CON PROMPT {idx+1} ===\n")

        modelo = LLM(model_id=selection_model,system_prompt=prompt)

        #Los nombres de los modelon continen el caracter /
        selection_model_r=selection_model.replace("/","_")
        #Carpeta donde se guardaran las respuestas
        carpeta_respuestas = os.path.join(ruta_actual, f"respuestas_experimento_{selection_model_r}")
        os.makedirs(carpeta_respuestas, exist_ok=True)


        # Crear o abrir el archivo para guardar las respuestas
        nombre_archivo_respuestas = f"respuestas_experimento_prompt_{idx+1}.txt"
        ruta_archivo_respuestas = os.path.join(carpeta_respuestas,nombre_archivo_respuestas)
        with open(ruta_archivo_respuestas, 'w', encoding='utf-8') as archivo_respuestas:
            archivo_respuestas.write(f"""==== EXPERIMENTO CON PROMPT {idx + 1} ====\n
                System Prompt usado:\n{prompt}\n\n""")
            archivo_respuestas.write("==== RESPUESTAS DEL MODELO ====\n\n")
            for pregunta in LISTA_DE_PREGUNTAS:
                print(f"Pregunta {LISTA_DE_PREGUNTAS.index(pregunta)+1}: Realizada al modelo.")
                respuesta = modelo.chat(pregunta, use_history=False, max_new_tokens=1024)
                print(f"Pregunta respondida")

                # Guardar la pregunta y respuesta en el archivo
                archivo_respuestas.write(f"=== Pregunta {LISTA_DE_PREGUNTAS.index(pregunta)+1} ===\n")
                archivo_respuestas.write(f"Pregunta: {pregunta}\n")
                archivo_respuestas.write(f"Respuesta: {respuesta}\n\n\n")

            print(f"Respuestas guardadas en: {ruta_archivo_respuestas}\n")
            del modelo  # Eliminar instancia del modelo
            torch.cuda.empty_cache()  # Vaciar caché de GPU
            torch.cuda.ipc_collect()  # Recolectar memoria compartida entre procesos

            print("Memoria de GPU liberada.\n")