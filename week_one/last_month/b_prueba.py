"""
Resumen basico semana pasada

Construimos un prompt para que el modelo actue como un asistente turistico experto en el
metro de Madrid. Ademas de eso incluimos las lineas de metro y las estaciones de conexion
entre lineas en formato json, para que el modelo pueda usarlas como referencia, solo
ingresar eso el prompt del sistemas sin ninun otro tipo de instrucciones adicionales, o
tecnicas de prompt, para ver su comportamiento por defecto. Que tanto se equivoca es lo
que queremos revisar.
"""
import torch
import json
import transformers
from transformers import logging
logging.set_verbosity_error()
logging.set_verbosity_warning()

#Pruebas de cuda
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Memory GPU:", torch.cuda.mem_get_info() )
print("PyTorch version:", torch.__version__)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

#Cargamos y revisamos los archivos Json
with open('lineas_metro.json', 'r') as f:
    lineas_metro = json.load(f)
with open('estaciones_conexiones.json', 'r') as f:
    estaciones_conexiones = json.load(f)

#print("Lineas de metro:", lineas_metro)
#print("Estaciones de conexiones:", estaciones_conexiones)

#Cargamos el modelo y pipeline

model_id="meta-llama/Llama-3.2-3B-Instruct"

pipeline= transformers.pipeline(
        task="text-generation",
        model=model_id,
        device_map="auto")


messages = [
    {"role": "system", "content": f"""
        Eres un asistente turístico experto en el Metro de Madrid.

        Instrucciones:
        - Responde siempre en español.
        - Indica las líneas y conexiones del metro para viajar entre estaciones.
        - Proporciona información sobre atracciones turísticas, monumentos y puntos de interés cercanos a las estaciones mencionadas.
        - Si el usuario no especifica un destino claro, haz preguntas aclaratorias y sugiere lugares populares.
        - Nunca pidas información personal.
        - Mantén un tono amigable, claro y conciso, como un guía local.
        - Usa únicamente la información del siguiente mapa de metro para calcular rutas y conexiones.
        - No inventes estaciones, líneas, conexiones ni datos turísticos que no estén en el mapa o en la información dada.
        - Si no sabes la respuesta, di que no tienes suficiente información para responder.

        Líneas de Metro Madrid y sus respectivas estaciones en orden y formacto json:
        {lineas_metro}
        Estaciones de metro donde se pueden hacer conexiones entre líneas en formato json:
        {estaciones_conexiones}
        """
    },
    {"role": "user", "content": "Hi, I want to go from Tetuan Metro station to Legazpi Metro, how can I do it?"},
]

outputs = pipeline(
    messages,

    max_new_tokens=512,
)
print(outputs[0]["generated_text"][-1])