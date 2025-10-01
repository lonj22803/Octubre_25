import transformers
from transformers import logging
logging.set_verbosity_error()  # Suppress warnings and info messages
import torch

print("CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto"

)

messages = [
    {"role": "system", "content": "You are a tourism assistant who always responds in Spanish. You are a specialist in the Madrid Metro."},
    {"role": "user", "content": "Hi, I want to go from Tetuan Metro station to Legazpi Metro, how can I do it?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=512,
)
print(outputs[0]["generated_text"][-1])
