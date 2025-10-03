"""
Implementación final
Dado el trabajo realizado en los scripts anteriores, este es el script final que implementa una interfaz en Gradio
para el modelo Llama 3.2 Instruct. Este script para el modelo incluye un prompt del sistema con instrucciones detalladas,
el ingreso de los datos en formato JSON para evaluar la capacidad del modelo para interpretar y utilizar estos datos,
historial de la conversación para mantener el contexto, y una interfaz en Gradio para facilitar la interacción con el modelo.

Juan Jose Londoño - 02-Octubre-2024 UPM-ETSIT
"""
import torch
import json
import gradio as gr
import transformers
from transformers import logging
from datetime import datetime

logging.set_verbosity_error()  # Suppress warnings and info messages

# Cargamos los JSON con las líneas y estaciones de metro, y las estaciones de conexión
try:
    with open('lineas_metro.json', 'r') as f:
        lineas_metro = json.load(f)
    print("✅ lineas_metro cargado:", list(lineas_metro.keys())[:3], "...")  # Muestra primeras 3 líneas para verificar

    with open('estaciones_conexiones.json', 'r') as f:
        estaciones_conexiones = json.load(f)
    print("✅ estaciones_conexiones cargado. Ejemplo:",
          list(estaciones_conexiones.items())[:3])  # Muestra primeras 3 conexiones
except Exception as e:
    print("❌ Error cargando JSON:", str(e))
    print("Asegúrate de que los archivos existan y sean válidos.")
    exit(1)

# Se revisa CUDA, GPU y memoria
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Memory GPU:", round(int(torch.cuda.mem_get_info()[0]) / 1024 ** 3, 3), " GB")

# Cargamos el modelo y pipeline
model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipeline = transformers.pipeline(
    task="text-generation",
    model=model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    pad_token_id=128000  # Evita warnings (EOS token aproximado para Llama)
)


# Formateamos los datos JSON para que sean más legibles en el prompt
def format_lineas_metro(data):
    formatted = ""
    for linea, estaciones in data.items():
        formatted += f"{linea}: {' -> '.join(estaciones)}\n"
    return formatted.strip()


def format_estaciones_conexiones(data):
    formatted = ""
    for estacion, lineas in data.items():
        formatted += f"{estacion}: {', '.join(lineas)}\n"
    return formatted.strip()


lineas_formatted = format_lineas_metro(lineas_metro)
conexiones_formatted = format_estaciones_conexiones(estaciones_conexiones)

# Prompt fijo del sistema (MEJORADO: más explícito, chain-of-thought, formato legible)
SYSTEM_PROMPT = f"""
Eres un asistente turístico experto en el Metro de Madrid. Usa SOLO los datos proporcionados abajo para calcular rutas. No inventes estaciones, líneas o conexiones.

Instrucciones paso a paso para calcular rutas:
1. Identifica la línea del origen y destino usando el mapa de líneas.
2. Si están en la misma línea, indica la dirección (adelante/atrás) y lista las estaciones intermedias.
3. Si no, encuentra estaciones de conexión cercanas usando el mapa de conexiones. Propón el transbordo más simple (mínimo 1-2 cambios).
4. Lista TODAS las estaciones paso a paso, la línea y dirección en cada segmento.
5. Si no hay ruta posible con los datos, di: "No tengo información suficiente para esta ruta. Sugiero consultar el mapa oficial."
6. Responde SIEMPRE en español, amigable y conciso. Incluye atracciones cercanas SOLO si están en los datos (no inventes).
7. Mantén contexto del historial, pero recalcula rutas si se pregunta de nuevo.

Mapa de LÍNEAS (estaciones en orden de dirección):
{lineas_formatted}

Mapa de CONEXIONES (estaciones donde cambiar de línea):
{conexiones_formatted}

Ejemplos de respuestas CORRECTAS (usa este estilo):
Usuario: "¿Cómo llego de Sol a Francos Rodríguez?"
Asistente: "Para ir de Sol a Francos Rodríguez: Toma Línea 1 desde Sol dirección Pinar de Chamartín hasta Cuatro Caminos (estaciones: Sol -> Gran Vía -> Tribunal -> Bilbao -> Iglesia -> Ríos Rosas -> Cuatro Caminos). En Cuatro Caminos, transbordo a Línea 7 dirección Hospital de Fuencarral hasta Francos Rodríguez (estaciones: Cuatro Caminos -> Guzmán el Bueno -> ... -> Francos Rodríguez). Atracciones cerca: En Sol, Palacio Real."

Usuario: "¿Cómo llego al Aeropuerto desde Nuevos Ministerios?"
Asistente: "Desde Nuevos Ministerios al Aeropuerto T4: Toma directamente Línea 8 dirección Aeropuerto T4 (estaciones: Nuevos Ministerios -> Colombia -> Mar de Cristal -> ... -> Aeropuerto T4). No necesitas transbordos. En el aeropuerto, conexiones con vuelos y trenes."
"""


def chat(messages):
    try:
        if not messages:
            initial_msg = [{"role": "assistant",
                            "content": "¡Hola! Soy tu asistente del Metro de Madrid. ¿A dónde quieres ir hoy?"}]
            return initial_msg

        # Limpia metadata/options de Gradio si existen (solo role y content)
        clean_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                clean_msg = {"role": msg["role"], "content": msg["content"]}
                clean_messages.append(clean_msg)
        messages = clean_messages

        user_message = messages[-1]["content"]
        conversation_history = "\n".join([
            f"{'Usuario' if msg['role'] == 'user' else 'Asistente'}: {msg['content']}"
            for msg in messages[:-1]
        ])

        full_prompt = f"""{SYSTEM_PROMPT}

Historial de la conversación:
{conversation_history}

Usuario:
{user_message}

Asistente (piensa paso a paso primero, luego responde):"""
        # Generamos la respuesta
        outputs = pipeline(
            full_prompt,
            max_new_tokens=256,  # Reducido para concisión
            do_sample=True,
            temperature=0.3,  # Más bajo para menos alucinaciones
            top_p=0.9
        )
        respuesta = outputs[0]["generated_text"][len(full_prompt):].strip()

        # Limpieza
        for separator in ["Usuario:", "Asistente:", "####", "Piensa paso a paso:"]:
            if separator in respuesta:
                respuesta = respuesta.split(separator)[0].strip()

        # Mensaje assistant limpio (solo role y content)
        assistant_msg = {"role": "assistant", "content": respuesta}
        messages.append(assistant_msg)
        return messages

    except Exception as e:
        print("Error during chat processing:", str(e))  # Solo log de error
        error_msg = "Lo siento, ha ocurrido un error al procesar tu solicitud."
        assistant_msg = {"role": "assistant", "content": error_msg}
        messages.append(assistant_msg)
        return messages


# Interfaz en Gradio
with gr.Blocks(title="Chatbot Metro Madrid") as demo:
    gr.Markdown("# Chatbot de Turismo en Español - Metro de Madrid")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Madrid Metro Tourism Assistant",
                type="messages",
                height=500,
                value=[]
            )
            with gr.Row():
                msg = gr.Textbox(placeholder="Escribe tu mensaje aquí...", scale=4)
                send = gr.Button("Enviar", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Preguntas de ejemplo")
            gr.Examples(
                examples=[
                    "¿Cómo llego de Sol a Francos Rodríguez?",
                    "¿Cómo llego al Aeropuerto desde Nuevos Ministerios?",
                    "¿Qué línea tomar para ir al Museo del Prado?",
                    "¿Cuáles son las atracciones cerca de la estación de Atocha?",
                    "¿Cómo puedo ir de Chamartín a Nuevos Ministerios?",
                    "¿Qué estaciones hay entre Plaza de Castilla y Argüelles?",
                    "¿Cómo puedo ir de Tetuán a Legazpi?"
                ],
                inputs=msg,
                label="Haz clic en una pregunta:"
            )


    def user_send(messages, user_message):
        if not user_message.strip():
            return messages, ""
        messages = messages or []
        # Limpia al agregar user
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        updated_messages = chat(messages)
        return updated_messages, ""


    send.click(user_send, inputs=[chatbot, msg], outputs=[chatbot, msg])
    msg.submit(user_send, inputs=[chatbot, msg], outputs=[chatbot, msg])

    # Guardado MEJORADO: Formato de interacciones secuenciales con limpieza agresiva
    save_btn = gr.Button("💾 Guardar chat en JSON")
    save_status = gr.Textbox(label="Estado de guardado", interactive=False)


    def save_conversation(messages):
        if not messages:
            return "❌ No hay historial para guardar."

        # Limpieza inicial: Extrae solo role y content de todos los messages (ignora metadata/options)
        clean_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                clean_msg = {"role": msg["role"], "content": msg["content"]}
                clean_messages.append(clean_msg)

        print("DEBUG SAVE: Mensajes limpios recibidos:", len(clean_messages))  # Temporal para depurar; comenta después

        # Reconstruimos interacciones secuenciales (solo user + assistant pairs)
        interactions = []
        i = 0
        while i < len(clean_messages):
            if clean_messages[i]["role"] == "user":
                user_question = clean_messages[i]["content"]
                # Busca el siguiente assistant (si existe)
                assistant_response = None
                debug_info = {"error": None}
                full_prompt = None
                if i + 1 < len(clean_messages) and clean_messages[i + 1]["role"] == "assistant":
                    assistant_response = clean_messages[i + 1]["content"]
                    # Reconstruye el full_prompt exacto para esta interacción
                    prev_history = clean_messages[:i]  # Historial hasta antes de este user
                    conversation_history = "\n".join([
                        f"{'Usuario' if msg['role'] == 'user' else 'Asistente'}: {msg['content']}"
                        for msg in prev_history
                    ])
                    full_prompt = f"""{SYSTEM_PROMPT}

Historial de la conversación:
{conversation_history}

Usuario:
{user_question}

Asistente (piensa paso a paso primero, luego responde):"""

                    # Debug info
                    debug_info = {
                        "prompt_length": len(full_prompt),
                        "model_id": model_id,
                        "generation_params": {
                            "max_new_tokens": 256,
                            "temperature": 0.3,
                            "top_p": 0.9
                        },
                        "interaction_timestamp": datetime.now().isoformat(),
                        "error": None
                    }
                    i += 2  # Salta al siguiente user
                else:
                    # User sin respuesta
                    debug_info["error"] = "No hay respuesta de assistant para esta pregunta"
                    i += 1

                interactions.append({
                    "user_question": user_question,
                    "model_input": full_prompt,  # Todo lo que ve el modelo (completo)
                    "assistant_response": assistant_response,
                    "debug_info": debug_info
                })
            else:
                i += 1  # Salta assistants iniciales

        print("DEBUG SAVE: Interacciones construidas:", len(interactions))  # Temporal; comenta después

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"chat_history_{timestamp}.json"

        save_data = {
            "metadata": {
                "saved_at": timestamp,
                "model_id": model_id,
                "total_interactions": len(interactions),
                "num_responses_generated": sum(1 for intxn in interactions if intxn["assistant_response"] is not None),
                "system_prompt_included": True,  # En cada model_input
                "sample_model_input_length": len(interactions[0]["model_input"]) if interactions else 0,
                # Verifica longitud
                "data_sources": {
                    "lineas_metro_keys": list(lineas_metro.keys()),
                    "conexiones_sample": list(estaciones_conexiones.items())[:3]
                }
            },
            "interactions": interactions
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            return f"✅ Historial guardado en {filename} (formato: {len(interactions)} interacciones con model_input completo)"
        except Exception as e:
            return f"❌ Error guardando chat: {str(e)}"


    save_btn.click(save_conversation, inputs=[chatbot], outputs=[save_status])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
