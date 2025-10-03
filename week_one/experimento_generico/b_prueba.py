"""
Implementación mejorada de un asistente de chat con Mistral-7B-Instruct-v0.3.
- Mejora en el almacenamiento del historial en el log: Ahora el historial se muestra como una sección indentada
  con líneas separadas para cada mensaje (incluyendo el system prompt al inicio), sin comillas envolventes largas.
  Esto hace que sea mucho más legible en el archivo 'conversacion_log.txt'.
  Formato actualizado por interacción:
  Pregunta del usuario: "Pregunta N"
  Modelo_id: "mistralai/Mistral-7B-Instruct-v0.3"
  System prompt: "El prompt de sistema fijo"
  Historial:
    Sistema: [contenido del system prompt]
    Usuario: [mensaje anterior 1]
    Asistente: [respuesta anterior 1]
    Usuario: [mensaje anterior 2]
    ...
  Respuesta: "Respuesta del modelo N"

  (Separador vacío entre interacciones para claridad).
- Se incluye el system prompt en el historial del log para contexto completo.
- Todas las demás funcionalidades permanecen iguales: interfaz Gradio, generación, limitación de historial, etc.
- Requiere: pip install transformers torch gradio sentencepiece accelerate
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import os

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

# Archivo para guardar el log de la conversación.
log_file = "conversacion_log.txt"

def save_interaction_to_log(user_question, model_response, full_history_before_response):
    """
    Guarda una interacción en un formato más legible en el archivo de log.
    - Historial: Se muestra como sección indentada con líneas separadas (incluyendo system).
    - full_history_before_response: Lista de mensajes (historial antes de la respuesta actual).
    """
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f'Pregunta del usuario: "{user_question}"\n')
        f.write(f'Modelo_id: "{model_id}"\n')
        f.write(f'System prompt: "{system_prompt}"\n')
        f.write('Historial:\n')
        # Construir historial textual línea por línea, incluyendo system.
        for msg in full_history_before_response:
            if msg["role"] == "system":
                f.write(f'  Sistema: {msg["content"]}\n')
            elif msg["role"] == "user":
                f.write(f'  Usuario: {msg["content"]}\n')
            elif msg["role"] == "assistant":
                f.write(f'  Asistente: {msg["content"]}\n')
        f.write(f'Respuesta: "{model_response}"\n')
        f.write("\n" * 2)  # Separador doble para claridad entre interacciones.

def generate_response(message, history_state):
    """
    Función para generar respuesta basada en el mensaje y el estado del historial.
    - history_state: Lista de dicts en formato {'role': 'user/assistant', 'content': str}.
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
        updated_history = history_state + [{"role": "user", "content": message}, {"role": "assistant", "content": generated_response}]

        # Guardar en log.
        save_interaction_to_log(message, generated_response, full_history_before_response)

        return generated_response, updated_history

    except Exception as e:
        error_msg = f"Error al generar la respuesta: {e}. Intenta de nuevo."
        updated_history = history_state + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]
        save_interaction_to_log(message, error_msg, full_history_before_response)
        return error_msg, updated_history

def submit_message(message, history_state):
    """
    Función de submit para Gradio: genera respuesta y actualiza chatbot y state.
    """
    if not message.strip():
        return "", history_state  # No hacer nada si mensaje vacío.

    response, new_history = generate_response(message, history_state)

    # Para el chatbot: agregar el nuevo mensaje y respuesta en formato messages.
    updated_chatbot = history_state + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]

    return "", updated_chatbot, new_history  # Limpiar textbox, actualizar chatbot y state.

# Inicializar archivo de log si no existe (con header mejorado).
if not os.path.exists(log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Log de Conversación Iniciado\n")
        f.write("=" * 50 + "\n\n")

# Crear interfaz de Gradio con manejo manual para evitar warning.
with gr.Blocks(title="Asistente con Mistral-7B", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Asistente con Mistral-7B-Instruct-v0.3")
    gr.Markdown("Chatea con el modelo. El historial se mantiene y se guarda automáticamente en 'conversacion_log.txt' (formato legible).")

    # Estado para historial (inicia vacío, sin system ya que se maneja internamente).
    history_state = gr.State([])

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
        inputs=[msg, history_state],
        outputs=[msg, chatbot, history_state]
    )

    # También permitir submit con Enter en textbox.
    msg.submit(
        fn=submit_message,
        inputs=[msg, history_state],
        outputs=[msg, chatbot, history_state]
    )

    # Opcional: Botón para limpiar chat.
    clear_btn = gr.Button("Limpiar Chat")
    clear_btn.click(
        fn=lambda: ([], [], []),
        outputs=[chatbot, msg, history_state]
    )

# Lanzar la interfaz.
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)