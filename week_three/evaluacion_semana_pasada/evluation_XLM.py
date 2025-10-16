import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargar datos
df_preguntas_respuestas = pd.read_csv("preguntas_respuestas.csv", encoding="utf-8-sig")
df_resultados_experimentos = pd.read_csv("resultados_experimentos_modelos_prompts.csv", encoding="utf-8-sig")

# Normalizar textos para facilitar emparejamiento
df_preguntas_respuestas["pregunta_norm"] = df_preguntas_respuestas["pregunta"].str.strip().str.lower()
df_resultados_experimentos["pregunta_norm"] = df_resultados_experimentos["Pregunta"].str.strip().str.lower()

# Verificar si CUDA está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")

# Unir ambos dataframes por la pregunta normalizada
df_merged = pd.merge(
    df_resultados_experimentos,
    df_preguntas_respuestas,
    how="inner",
    on="pregunta_norm"
)

# Cargar modelo XLM-RoBERTa-large-XNLI
model_name = "joeddav/xlm-roberta-large-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()

# Función para evaluar respuestas usando el nuevo modelo
def evaluar_respuesta(pregunta, respuesta_base, respuesta_usuario):
    premise = pregunta  # la pregunta como premisa
    hypothesis = f"This example is {respuesta_usuario}."  # la respuesta del usuario como hipótesis

    inputs = tokenizer.encode(premise, hypothesis, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        logits = model(inputs)[0]

    # Tomamos solo entailment (índice 2) y contradiction (índice 0)
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = torch.softmax(entail_contradiction_logits, dim=1)[0]
    prob_entailment = probs[1].item()
    prob_contradiction = probs[0].item()

    return {
        "es_correcta": prob_entailment > 0.5,
        "confianza": prob_entailment,
        "probabilidades": {
            "entailment": prob_entailment,
            "contradiction": prob_contradiction
        }
    }

# Evaluar todas las respuestas
resultados = []
for idx, row in df_merged.iterrows():
    modelo = row["Modelo"]
    pregunta = row["Pregunta"]
    respuesta_base = row["respuesta_esperada"]
    respuesta_usuario = row["Respuesta"]
    prompt = row["Prompt"]

    resultado = evaluar_respuesta(pregunta, respuesta_base, respuesta_usuario)

    resultados.append({
        "modelo": modelo,
        "pregunta": pregunta,
        "prompt": prompt,
        "es_correcta": resultado["es_correcta"],
        "confianza": resultado["confianza"],
        "prob_entailment": resultado["probabilidades"]["entailment"],
        "prob_contradiction": resultado["probabilidades"]["contradiction"]
    })

# Crear y guardar resultados
df_resultados_xlm = pd.DataFrame(resultados)
df_resultados_xlm.to_csv("resultados_xlm_roberta.csv", index=False)

print(f"✅ Evaluación completada ({len(df_resultados_xlm)} filas) y guardada en 'resultados_xlm_roberta.csv'.")
