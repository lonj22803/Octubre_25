"""
Implementación mejorada de un asistente de chat con Mistral-7B-Instruct-v0.3.
- Nuevo sistema de guardado: Cada chat (sesión) se guarda en un archivo JSON separado.
  - Al iniciar un nuevo chat (al lanzar la app o al presionar 'Nuevo Chat'), se crea un archivo único como 'chat_YYYYMMDD_HHMMSS.json'.
  - El JSON tiene estructura:
    {
      "chat_id": "unique_timestamp",
      "start_time": "YYYY-MM-DD HH:MM:SS",
      "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
      "system_prompt": "El prompt fijo",
      "interactions": [
        {
          "user_question": "Pregunta N",
          "history": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
            // ... historial antes de esta respuesta
          ],
          "response": "Respuesta del modelo N"
        },
        // ... más interacciones
      ]
    }
  - Esto es legible, estructurado y fácil de parsear/programar con (e.g., en Python con json.load).
  - Se guarda incrementalmente después de cada interacción (lee el JSON existente, agrega la nueva, y reescribe).
- El historial en cada interacción incluye el system prompt y mensajes previos (formato legible como lista de dicts).
- Botón 'Nuevo Chat' para iniciar una nueva sesión y crear un nuevo JSON (resetea historial y UI).
- Todas las demás funcionalidades permanecen: interfaz Gradio, generación, limitación de historial, etc.
- Requiere: pip install transformers torch gradio sentencepiece accelerate
  (Y `import json` y `from datetime import datetime` para JSON y timestamps).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import os
import json
from datetime import datetime

# Identificador del modelo (fijo).
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Carga del tokenizador.
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Carga del modelo.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Prompt de sistema fijo (se agrega al inicio de cada conversación).
system_prompt = "Eres un asistente útil y amigable. Responde en español si la pregunta está en español. Sé conciso y preciso."


def generate_chat_filename():
    """
    Genera un nombre de archivo único para el chat basado en timestamp.
    Ejemplo: 'chat_20241025_143022.json'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_{timestamp}.json"


def initialize_chat_json(chat_file):
    """
    Inicializa un nuevo archivo JSON para el chat con metadatos.
    """
    chat_data = {
        "chat_id": os.path.splitext(os.path.basename(chat_file))[0],  # e.g., "chat_20241025_143022"
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "system_prompt": system_prompt,
        "interactions": []
    }
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)
    return chat_data["interactions"]  # Retorna la lista vacía de interacciones.


def save_interaction_to_json(chat_file, user_question, model_response, full_history_before_response):
    """
    Agrega una interacción al JSON del chat (incremental).
    - Lee el JSON existente, agrega la nueva interacción, y reescribe.
    - full_history_before_response: Lista de dicts (incluye system + previos).
    """
    if not os.path.exists(chat_file):
        interactions = initialize_chat_json(chat_file)
    else:
        with open(chat_file, "r", encoding="utf-8") as f:
            chat_data = json.load(f)
        interactions = chat_data["interactions"]

    # Nueva interacción.
    new_interaction = {
        "user_question": user_question,
        "history": full_history_before_response,  # Lista de {"role": "...", "content": "..."}
        "response": model_response
    }
    interactions.append(new_interaction)

    # Reescribir el JSON completo.
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump({
            "chat_id": os.path.splitext(os.path.basename(chat_file))[0],
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Opcional: mantener el start_time original.
            "model_id": model_id,
            "system_prompt": system_prompt,
            "interactions": interactions
        }, f, ensure_ascii=False, indent=2)


def generate_response(message, history_state, chat_file):
    """
    Función para generar respuesta basada en el mensaje y el estado del historial.
    - Ahora recibe chat_file para guardar.
    Retorna la respuesta generada y el historial actualizado.
    """
    # Construir conversation list a partir de history_state + system + nuevo mensaje.
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(history_state)  # Agregar historial previo.

    # full_history_before_response para log (incluye system + previos, antes de este user).
    full_history_before_response = conversation.copy()
    conversation.append({"role": "user", "content": message})

    # Limitar historial (mantener system + últimos 10 mensajes).
    if len(conversation) > 11:
        conversation = [conversation[0]] + conversation[-10:]

    try:
        # Aplicar plantilla de chat.
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generar respuesta.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decodificar respuesta generada.
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_response = full_response[input_length:].strip()

        # Actualizar historial con la nueva respuesta.
        updated_history = history_state + [{"role": "user", "content": message},
                                           {"role": "assistant", "content": generated_response}]

        # Guardar en JSON.
        save_interaction_to_json(chat_file, message, generated_response, full_history_before_response)

        return generated_response, updated_history

    except Exception as e:
        error_msg = f"Error al generar la respuesta: {e}. Intenta de nuevo."
        updated_history = history_state + [{"role": "user", "content": message},
                                           {"role": "assistant", "content": error_msg}]
        save_interaction_to_json(chat_file, message, error_msg, full_history_before_response)
        return error_msg, updated_history


def submit_message(message, history_state, chat_file):
    """
    Función de submit para Gradio: genera respuesta y actualiza chatbot y state.
    """
    if not message.strip():
        return "", history_state, chat_file  # No hacer nada si mensaje vacío.

    response, new_history = generate_response(message, history_state, chat_file)

    # Para el chatbot: agregar el nuevo mensaje y respuesta en formato messages.
    updated_chatbot = history_state + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]

    return "", updated_chatbot, new_history, chat_file  # Limpiar textbox, actualizar chatbot, state y chat_file.


def start_new_chat():
    """
    Inicia un nuevo chat: genera nuevo archivo JSON y resetea historial.
    """
    new_chat_file = generate_chat_filename()
    initialize_chat_json(new_chat_file)
    return [], new_chat_file  # Resetear historial y nuevo archivo.


# Crear interfaz de Gradio con manejo manual.
with gr.Blocks(title="Asistente con Mistral-7B", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Asistente con Mistral-7B-Instruct-v0.3")
    gr.Markdown(
        "Chatea con el modelo. Cada chat se guarda en un archivo JSON separado (e.g., 'chat_YYYYMMDD_HHMMSS.json'). Usa 'Nuevo Chat' para iniciar uno nuevo.")

    # Estado para historial y chat_file (inicia con nuevo chat).
    history_state = gr.State([])
    chat_file_state = gr.State(generate_chat_filename())  # Inicializa con un nuevo archivo al launch.
    initialize_chat_json(chat_file_state.value)  # Crea el archivo inicial.

    # Mostrar el archivo actual (para info del usuario).
    current_file = gr.Markdown("Archivo actual: " + chat_file_state.value)

    # Chatbot con nuevo formato.
    chatbot = gr.Chatbot(type="messages", height=500)

    # Textbox para input.
    msg = gr.Textbox(
        placeholder="Escribe tu mensaje aquí...",
        label="Tu mensaje"
    )

    # Botón de submit.
    submit_btn = gr.Button("Enviar", variant="primary")

    # Evento de submit.
    submit_btn.click(
        fn=submit_message,
        inputs=[msg, history_state, chat_file_state],
        outputs=[msg, chatbot, history_state, chat_file_state]
    ).then(
        fn=lambda x: "Archivo actual: " + x,  # Actualizar display del archivo (aunque no cambie).
        inputs=[chat_file_state],
        outputs=[current_file]
    )

    # También permitir submit con Enter en textbox.
    msg.submit(
        fn=submit_message,
        inputs=[msg, history_state, chat_file_state],
        outputs=[msg, chatbot, history_state, chat_file_state]
    ).then(
        fn=lambda x: "Archivo actual: " + x,
        inputs=[chat_file_state],
        outputs=[current_file]
    )

    # Botón para nuevo chat (reemplaza 'Limpiar Chat').
    new_chat_btn = gr.Button("Nuevo Chat", variant="secondary")
    new_chat_btn.click(
        fn=start_new_chat,
        outputs=[history_state, chat_file_state]
    ).then(
        fn=lambda x: ([], "", "Archivo actual: " + x),  # Resetear chatbot, msg y actualizar display.
        inputs=[chat_file_state],
        outputs=[chatbot, msg, current_file]
    )

# Lanzar la interfaz.
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
