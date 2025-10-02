"""
Implementacion final
Dado el trabajo realizado en los scripts anteriores, este es el script final que implementa hasta una interfaz en gradio
para el modelo Llama 3.2 instrct, este script para el modelo incluye un prompt del sistema con instrucciones detalladas para que el modelo,
el ingreso de los datos en un formaro json, para evaluar la capacidad del modelo para interpretar y utilizar estos datos,
historial de la conversacion para mantener el contexto, y una interfaz en gradio para facilitar la interaccion con el modelo.

Juan Jose Londoño- 02-Octubre de 2025 UPM-ETSIT
"""
import torch
import json
import gradio
import transformers
from transformers import logging

from week_one.last_month.a_prueba import pipeline

logging.set_verbosity_error()  # Suppress warnings and info messages

#Cargamos tanto el jeson con las lineas y estaciones de metro, como el json con las estaciones de conexion entre lineas

with open('lineas_metro.json', 'r') as f:
    lineas_metro = json.load(f)

with open('estaciones_conexiones.json', 'r') as f:
    estaciones_conexiones = json.load(f)

#print("Lineas de metro:", lineas_metro)
#print("Estaciones de conexiones:", estaciones_conexiones)

#Se revisa CUDA, GPU y memoria
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Memory GPU:", round(int(torch.cuda.mem_get_info()[0])/1024**3,3)," GB" ) #Memoria en GB

# Cargamos el modelo y pipeline
model_id="meta-llama/Llama-3.2-3B-Instruct"

pipeline = transformers.pipeline(
        task="text-generation",
        model=model_id,
        device_map="auto"
)

#Prompt fijo del sistema
SYSTEM_PROMPT = f"""
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

Lineas de metro y sus estaciones:
{lineas_metro}

Etsaciones en donde se hacen conexiones entre lineas:
{estaciones_conexiones}
"""