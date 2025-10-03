from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import os
import json
from datetime import datetime

# Configuraciones por defecto
DEFAULT_SYSTEM_PROMPT = "Eres un asistente útil y amigable. Responde en español si la pregunta está en español. Sé conciso y preciso."
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TORCH_DTYPE = torch.bfloat16
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# Modelos HF predefinidos para dropdown
HF_MODELS = {
    "Mistral-7B-Instruct (Default)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "DeepSeek-Coder-6.7B-Instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "Phi-3-Mini-4k-Instruct": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma-2-9B-IT": "google/gemma-2-9b-it",
    "Personalizado": ""  # Para input manual
}

# Cache global para modelos y tokenizers (evitar recargas)
model_cache = {}
tokenizer_cache = {}


def load_hf_model(model_id, torch_dtype=DEFAULT_TORCH_DTYPE):
    """Carga modelo y tokenizer de HF si no está cacheado."""
    if model_id not in model_cache:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            tokenizer_cache[model_id] = tokenizer
            model_cache[model_id] = model
            print(f"Modelo '{model_id}' cargado exitosamente.")
        except Exception as e:
            raise ValueError(f"Error al cargar el modelo '{model_id}': {e}. Verifica el ID y tu acceso a Hugging Face.")
    return model_cache[model_id], tokenizer_cache[model_id]


def generate_hf_response(conversation, model, tokenizer, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                         temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
    """Genera respuesta usando modelo HF."""
    try:
        # Intentar chat template
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        if inputs is None:  # Fallback si no hay chat template (modelos base)
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            if prompt is None:
                # Formato simple de fallback
                prompt = f"<s>[INST] {DEFAULT_SYSTEM_PROMPT} [/INST] " + " ".join(
                    [f"{msg['role']}: {msg['content']}" for msg in conversation[1:]]) + " [/INST]"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_response = full_response[input_length:].strip()
        return generated_response
    except Exception as e:
        return f"Error al generar respuesta: {e}. Intenta de nuevo o verifica el modelo."


def generate_chat_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_{timestamp}.json"


def initialize_chat_json(chat_file, model_id=DEFAULT_MODEL_ID):
    chat_data = {
        "chat_id": os.path.splitext(os.path.basename(chat_file))[0],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "interactions": {}
    }
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)
    return chat_data["interactions"]


def save_interaction_to_json(chat_file, user_question, model_response, full_history_before_response,
                             model_id=DEFAULT_MODEL_ID):
    if not os.path.exists(chat_file):
        interactions = initialize_chat_json(chat_file, model_id)
    else:
        with open(chat_file, "r", encoding="utf-8") as f:
            chat_data = json.load(f)
        interactions = chat_data["interactions"]

    next_num = len(interactions) + 1
    interaction_key = f"interaccion_{next_num}"

    new_interaction = {
        "user_question": user_question,
        "history": full_history_before_response,
        "response": model_response
    }
    interactions[interaction_key] = new_interaction

    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump({
            "chat_id": os.path.splitext(os.path.basename(chat_file))[0],
            "start_time": chat_data.get("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "model_id": model_id,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "interactions": interactions
        }, f, ensure_ascii=False, indent=2)


def generate_response(message, history_state, chat_file, model_id):
    """Genera respuesta basada en el mensaje, historial y model_id."""
    if not model_id:
        return "Error: Especifica un model_id válido.", history_state

    # Construir conversation
    conversation = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    conversation.extend(history_state)
    full_history_before_response = conversation.copy()
    conversation.append({"role": "user", "content": message})

    # Limitar historial (system + últimos 10 mensajes)
    if len(conversation) > 11:
        conversation = [conversation[0]] + conversation[-10:]

    try:
        model, tokenizer = load_hf_model(model_id)
        response = generate_hf_response(conversation, model, tokenizer)

        # Guardar en JSON
        save_interaction_to_json(chat_file, message, response, full_history_before_response, model_id)

        # Actualizar historial
        updated_history = history_state + [{"role": "user", "content": message},
                                           {"role": "assistant", "content": response}]
        return response, updated_history

    except Exception as e:
        error_msg = f"Error al generar la respuesta: {e}. Intenta de nuevo."
        save_interaction_to_json(chat_file, message, error_msg, full_history_before_response, model_id)
        updated_history = history_state + [{"role": "user", "content": message},
                                           {"role": "assistant", "content": error_msg}]
        return error_msg, updated_history


def submit_message(message, history_state, chat_file, model_id):
    """Función de submit para Gradio."""
    if not message.strip():
        return "", history_state, chat_file

    response, new_history = generate_response(message, history_state, chat_file, model_id)
    updated_chatbot = history_state + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    return "", updated_chatbot, new_history, chat_file


def start_new_chat(model_id):
    """Inicia un nuevo chat con el model_id actual."""
    new_chat_file = generate_chat_filename()
    initialize_chat_json(new_chat_file, model_id)
    return [], new_chat_file


# Función para actualizar model_id desde dropdown
def update_model_id(selected_model):
    return HF_MODELS.get(selected_model, "")


# Interfaz Gradio
with gr.Blocks(title="Asistente HF Generalizado", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Asistente con Modelos de Hugging Face")
    gr.Markdown(
        "Selecciona un modelo HF. Por defecto: Mistral-7B. Cada chat se guarda en un archivo JSON separado. Usa 'Nuevo Chat' para iniciar uno nuevo.")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(HF_MODELS.keys()),
            value="Mistral-7B-Instruct (Default)",
            label="Modelo HF"
        )
        custom_model = gr.Textbox(
            placeholder="ID de modelo personalizado (e.g., meta-llama/Meta-Llama-3-8B-Instruct)",
            label="Modelo Personalizado (si seleccionas 'Personalizado')"
        )
        current_model_display = gr.Markdown(f"Modelo actual: {DEFAULT_MODEL_ID}")


    # Actualizar custom_model y display al cambiar dropdown
    def on_model_change(selected_model, custom_input):
        model_id = HF_MODELS.get(selected_model, custom_input or "")
        if not model_id and selected_model != "Personalizado":
            model_id = HF_MODELS[selected_model]
        return model_id, f"Modelo actual: {model_id or 'No especificado'}"


    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown, custom_model],
        outputs=[custom_model, current_model_display]
    )
    custom_model.change(
        fn=on_model_change,
        inputs=[model_dropdown, custom_model],
        outputs=[custom_model, current_model_display]
    )

    # Estados (inicia con default)
    history_state = gr.State([])
    chat_file_state = gr.State(generate_chat_filename())
    model_id_state = gr.State(DEFAULT_MODEL_ID)
    initialize_chat_json(chat_file_state.value, DEFAULT_MODEL_ID)

    # Mostrar archivo actual
    current_file = gr.Markdown("Archivo actual: " + chat_file_state.value)

    # Chatbot
    chatbot = gr.Chatbot(type="messages", height=500)

    # Input
    msg = gr.Textbox(placeholder="Escribe tu mensaje aquí...", label="Tu mensaje")

    # Botón submit
    submit_btn = gr.Button("Enviar", variant="primary")
    submit_btn.click(
        fn=submit_message,
        inputs=[msg, history_state, chat_file_state, model_id_state],
        outputs=[msg, chatbot, history_state, chat_file_state]
    ).then(
        fn=lambda x: "Archivo actual: " + x,
        inputs=[chat_file_state],
        outputs=[current_file]
    )

    # Submit con Enter
    msg.submit(
        fn=submit_message,
        inputs=[msg, history_state, chat_file_state, model_id_state],
        outputs=[msg, chatbot, history_state, chat_file_state]
    ).then(
        fn=lambda x: "Archivo actual: " + x,
        inputs=[chat_file_state],
        outputs=[current_file]
    )

    # Botón nuevo chat (usa model_id actual)
    new_chat_btn = gr.Button("Nuevo Chat", variant="secondary")
    new_chat_btn.click(
        fn=start_new_chat,
        inputs=[model_id_state],
        outputs=[history_state, chat_file_state]
    ).then(
        fn=lambda x: ([], "", "Archivo actual: " + x),
        inputs=[chat_file_state],
        outputs=[chatbot, msg, current_file]
    )


    # Actualizar model_id_state cuando cambie la selección (para usarlo en submits)
    def update_state_model(custom_input, selected_model):
        model_id = HF_MODELS.get(selected_model, custom_input or DEFAULT_MODEL_ID)
        return model_id


    gr.State.value = gr.update()  # Para trigger inicial
    model_dropdown.change(
        fn=update_state_model,
        inputs=[custom_model, model_dropdown],
        outputs=[model_id_state]
    )
    custom_model.change(
        fn=update_state_model,
        inputs=[custom_model, model_dropdown],
        outputs=[model_id_state]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

