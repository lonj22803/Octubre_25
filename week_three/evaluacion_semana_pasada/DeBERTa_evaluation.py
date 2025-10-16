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

# Cargar modelo mDeBERTa multilingüe
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()  # poner en modo evaluación

# Función para evaluar respuestas
def evaluar_respuesta(pregunta, respuesta_base, respuesta_usuario):
    premisa = f"{pregunta} La respuesta es {respuesta_base}."
    hipotesis = respuesta_usuario

    inputs = tokenizer(premisa, hipotesis, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=1)[0]
    prob_entailment = probabilities[0].item()
    prob_neutral = probabilities[1].item()
    prob_contradiction = probabilities[2].item()

    return {
        "es_correcta": prob_entailment > 0.7,
        "confianza": prob_entailment,
        "probabilidades": {
            "entailment": prob_entailment,
            "neutral": prob_neutral,
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
        "prob_neutral": resultado["probabilidades"]["neutral"],
        "prob_contradiction": resultado["probabilidades"]["contradiction"]
    })

# Crear y guardar resultados
df_resultados_deberta = pd.DataFrame(resultados)
df_resultados_deberta.to_csv("resultados_mdeberta_two.csv", index=False)

print(f"✅ Evaluación completada ({len(df_resultados_deberta)} filas) y guardada en 'resultados_mdeberta_two.csv'.")

