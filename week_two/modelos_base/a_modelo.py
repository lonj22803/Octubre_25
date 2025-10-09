import transformers
from transformers import logging
import torch

# Suppress warnings and info messages
logging.set_verbosity_error()


class LLM:
    """
    English:
    A flexible class wrapper for using different LLM models via the Hugging Face transformers pipeline.
    Supports text-generation with chat-style messages (system/user/assistant roles).
    Automatically handles device mapping for GPU/CPU.
    Spanish:
    Una clase envoltorio flexible para usar diferentes modelos LLM a través del pipeline de hugging Face transformers.
    Soporta generación de texto con mensajes estilo chat (roles system/user/assistant).
    Maneja automáticamente el mapeo de dispositivos para GPU/CPU.
    """

    # VARIABLES DE CLASE GLOBALES
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # Default model ID

    def __init__(self, model_id: str = MODEL_ID, system_prompt: str = None, device_map: str = "auto"):
        """
        Initializes the LLM pipeline and optionally sets a system prompt for conversational use.

        Args:
            model_id (str): The Hugging Face model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
            system_prompt (str, optional): The system prompt to encapsulate (e.g., "You are a helpful assistant.").
                                           If provided, it will be used to initialize the conversation history.
            device_map (str): Device mapping strategy (default: "auto" for multi-GPU/CPU fallback).
        """
        print(f"Loading model: {model_id}")
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs:", torch.cuda.device_count())

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            device_map=device_map,
            dtype=torch.float16,  # Use float16 for efficiency; change if needed
            trust_remote_code=True  # Enable if the model requires it (e.g., for custom code)
        )
        print(f"Model {model_id} loaded successfully.")

        # Encapsulate system prompt and initialize conversation history for chatbot functionality
        self.system_prompt = system_prompt
        self.conversation_history = []
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})
            print(f"System prompt encapsulated: {system_prompt[:50]}...")

    def set_system_prompt(self, new_system_prompt: str):
        """
        Cambia el system prompt dinámicamente y resetea la historia de conversación.
        No recarga el modelo/pipeline.

        Args:
            new_system_prompt (str): Nuevo system prompt.
        """
        self.system_prompt = new_system_prompt
        self.reset_history()  # Resetea historia, pero mantiene el nuevo prompt
        print(f"System prompt actualizado: {new_system_prompt[:50]}...")

    def chat(self, user_input: str, max_new_tokens: int = 512, use_history: bool = True, **kwargs) -> str:
        """
        Generates a response in a conversational manner. Maintains history for future interactions if use_history=True.
        If use_history=False, generates a single response using only the system prompt + current user input (no history modification).

        Args:
            user_input (str): The user's question or message.
            max_new_tokens (int): Maximum number of new tokens to generate (default: 512).
            use_history (bool): If True (default), adds to and uses conversation history. If False, generates a one-off response without affecting history.
            **kwargs: Additional arguments to pass to the pipeline (e.g., temperature, do_sample).

        Returns:
            str: The generated assistant response.

        Raises:
            Exception: If generation fails.
        """
        if not self.system_prompt:
            raise ValueError("No system prompt provided. Use __init__ with system_prompt for chat functionality.")

        try:
            if use_history:
                # Conversational mode: Add user to history and generate with full history
                self.conversation_history.append({"role": "user", "content": user_input})
                messages_to_use = self.conversation_history.copy()
            else:
                # One-off mode: Use only system + current user input, no history modification
                messages_to_use = [{"role": "system", "content": self.system_prompt}]
                messages_to_use.append({"role": "user", "content": user_input})

            # Generate using the messages
            outputs = self.pipeline(
                messages_to_use,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            # Extract the last generated message (assistant response)
            generated_text = outputs[0]["generated_text"]
            if isinstance(generated_text, list):
                # For chat formats, it's often a list of messages; take the last one (new assistant message)
                last_message = generated_text[-1]
                if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                    response = last_message["content"]
                else:
                    response = str(last_message)
            else:
                # Fallback: extract the assistant response from the full generated text
                # This assumes the model appends only the new assistant message
                response = str(generated_text).strip()

            # Only append to history if using history mode
            if use_history:
                self.conversation_history.append({"role": "assistant", "content": response})

            return response
        except Exception as e:
            raise Exception(f"Chat generation failed: {str(e)}")

    def generate(self, messages: list[dict], max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generates text based on the provided messages (non-conversational, overrides history).

        Args:
            messages (list[dict]): List of message dictionaries with 'role' and 'content' keys
                                   (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]).
            max_new_tokens (int): Maximum number of new tokens to generate (default: 512).
            **kwargs: Additional arguments to pass to the pipeline (e.g., temperature, do_sample).

        Returns:
            str: The generated response text (last assistant message).

        Raises:
            Exception: If generation fails.
        """
        try:
            outputs = self.pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            # Extract the last generated message (assistant response)
            generated_text = outputs[0]["generated_text"]
            if isinstance(generated_text, list):
                # For chat formats, it's often a list of messages; take the last one
                last_message = generated_text[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
                else:
                    return str(last_message)
            else:
                # Fallback for plain text output
                return str(generated_text)
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")

    def reset_history(self):
        """
        Resets the conversation history (keeps the system prompt if set).
        """
        self.conversation_history = []
        if self.system_prompt:
            self.conversation_history.append({"role": "system", "content": self.system_prompt})
        print("Conversation history reset.")


def json_to_text_metro(data: dict) -> str:
    """Convierte un JSON de líneas de metro a texto descriptivo."""
    output = ["Lineas de Metro de una ciudad y sus respectivas estaciones en sentido de ida y regreso:"]

    for nombre_linea, info in data.get("lineas", {}).items():
        estaciones = " → ".join(info.get("estaciones", []))
        sentido_ida = info.get("sentido_ida", [])
        sentido_vuelta = " → ".join(info.get("sentido_vuelta", []))


        bloque = (
            f"- Línea {nombre_linea}:\n"
            f"  - Estaciones: {estaciones}\n"
            f"  - Sentido ida: {sentido_ida}\n"
            f"  - Sentido vuelta: {sentido_vuelta}."
        )
        output.append(bloque)

    return "\n".join(output)


# Example usage for chatbot mode (encapsulating system prompt)
if __name__ == "__main__":
    # Using default MODEL_ID
    system_prompt = "You are a tourism assistant who always responds in Spanish. You are a specialist in the Madrid Metro."

    llm = LLM(system_prompt=system_prompt)  # Uses MODEL_ID by default

    # One-off response (no history, single generation)
    user_question_oneoff = "Hi, I want to go from Tetuan Metro station to Legazpi Metro, how can I do it?"
    response_oneoff = llm.chat(user_question_oneoff, use_history=False, max_new_tokens=512)
    print("One-off Response:", response_oneoff)

    # Now start a conversational chat (uses history)
    user_question1 = "How do I get to Sol from here?"
    response1 = llm.chat(user_question1, max_new_tokens=512)  # use_history=True by default
    print("Chat Response 1:", response1)

    # Second interaction (maintains history)
    user_question2 = "And how long does it take?"
    response2 = llm.chat(user_question2, max_new_tokens=512)
    print("Chat Response 2:", response2)

    # Reset history if needed
    # llm.reset_history()

    # You can still use the original generate method for one-off queries
    # messages = [{"role": "user", "content": "Hello!"}]
    # response = llm.generate(messages)
    # print("One-off response:", response)




