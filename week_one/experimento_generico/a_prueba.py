"""
Dado que las implementaciones anteriores no surgieron fruto, como se esperaba, analizarmeos desde cero el problema
impolementado un asistente con un modelo mas grande, vy trataremos de hacer pruebas con nuestro experimento genérico
dado que ya tenemos el entorno y las herramientas para ello. Ademas despues de probarlo con varios modelos del mercado,
vemos que con facilidad resolvieron el problema.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Definición del identificador del modelo a utilizar.
# Este es un modelo de Mistral AI, versión Instruct v0.3 con 7B parámetros, optimizado para instrucciones y chat.
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Carga del tokenizador preentrenado correspondiente al modelo especificado.
# El tokenizador convierte texto en tokens que el modelo puede procesar.
# Nota: Si no tienes sentencepiece instalado, instálalo con 'pip install sentencepiece'.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Si el tokenizador no tiene un padding token, agregamos uno (común en modelos como Mistral).
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Carga del modelo preentrenado.
# 'torch_dtype=torch.bfloat16' usa precisión bfloat16 para eficiencia en memoria y velocidad (requiere GPU compatible).
# 'device_map="auto"' distribuye automáticamente el modelo en dispositivos disponibles (GPU/CPU).
# 'trust_remote_code=True' permite código remoto si es necesario (para algunos modelos).
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Inicialización del historial de conversación.
# Comienza vacío para permitir interacciones multi-turno.
conversation = []

# Mensaje de bienvenida y sistema (opcional: para guiar al modelo).
# Agregamos un mensaje de sistema para definir el rol del asistente.
system_message = {
    "role": "system",
    "content": "Eres un asistente útil y amigable. Responde en español si la pregunta está en español. Sé conciso y preciso."
}
conversation.append(system_message)

print("¡Bienvenido al Asistente de Preguntas con Mistral-7B! Escribe tu pregunta o 'exit' para salir.\n")

# Bucle principal para interacciones interactivas.
while True:
    # Solicitar input del usuario.
    user_input = input("Tú: ").strip()

    # Condición de salida.
    if user_input.lower() in ["exit", "salir", "quit"]:
        print("¡Adiós! Gracias por usar el asistente.")
        break

    # Si el input está vacío, continuar.
    if not user_input:
        continue

    # Agregar el mensaje del usuario al historial.
    conversation.append({"role": "user", "content": user_input})

    try:
        # Aplicación de la plantilla de chat para formatear el prompt completo (incluyendo historial).
        # 'add_generation_prompt=True' agrega un prompt para generar la respuesta del asistente.
        # 'return_dict=True' devuelve un diccionario con los tensores de entrada.
        # 'return_tensors="pt"' convierte los tokens en tensores de PyTorch.
        # Padding para manejar longitudes variables.
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True  # Agrega padding si es necesario para batches (aquí es single-turn).
        )

        # Movimiento de los inputs al dispositivo del modelo (por ejemplo, GPU).
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generación de la respuesta del modelo.
        # '**inputs' desempaqueta el diccionario de tensores.
        # 'max_new_tokens=300' limita la longitud de la respuesta para mantenerla concisa.
        # 'do_sample=True' habilita muestreo para respuestas más variadas.
        # 'temperature=0.7' controla la creatividad (menor = más determinista).
        # 'top_p=0.9' usa nucleus sampling para diversidad.
        # 'eos_token_id=tokenizer.eos_token_id' detiene en token de fin.
        with torch.no_grad():  # Desactiva gradientes para inferencia eficiente.
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decodificación de la salida generada.
        # 'outputs[0]' toma la primera (y única) secuencia generada.
        # 'skip_special_tokens=True' ignora tokens especiales como <s> o </s>.
        # Extraemos solo la parte nueva (después del input original).
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Encontramos el inicio de la respuesta del asistente (después del último [/INST]).
        response_start = full_response.rfind("[/INST]") + 8  # Aproximación para Mistral.
        if response_start > 8:
            generated_response = full_response[response_start:].strip()
        else:
            generated_response = full_response[
                len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

        # Imprimir la respuesta.
        print(f"Asistente: {generated_response}\n")

        # Agregar la respuesta al historial para mantener el contexto.
        conversation.append({"role": "assistant", "content": generated_response})

        # Opcional: Limitar el historial para evitar que crezca demasiado (por memoria).
        # Mantener solo los últimos 10 intercambios (5 user + 5 assistant).
        if len(conversation) > 11:  # Incluyendo system.
            conversation = [conversation[0]] + conversation[-10:]  # Mantiene system + últimos 10.

    except Exception as e:
        print(f"Error al generar la respuesta: {e}")
        print("Intenta de nuevo con una pregunta diferente.\n")