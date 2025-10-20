from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargamos un modelo mejor para español: BERT finetuneado en XNLI
model_name = "Recognai/bert-base-spanish-wwm-cased-xnli"  # Modelo recomendado para español
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Función para verificar entailment (igual que antes, pero con el nuevo modelo)
def check_entailment(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    # Para XNLI, las etiquetas son: 0=contradiction, 1=entailment, 2=neutral
    labels = ["contradiction", "entailment", "neutral"]
    predicted_label = labels[torch.argmax(probabilities)]
    return predicted_label

# Definimos la pregunta y las respuestas
pregunta = "¿Cómo puedo llegar desde la estación AA1SC a la estación AG7BH?"
respuestas = {
    "R1": "Puedes llegar desde la estación AA1SC a la estación AG7BH tomando la línea amarilla. Primero, dirígete hacia la estación AA1SC, que es la primera estación del sentido ida de la línea amarilla. Luego, sigue el recorrido de la línea amarilla en sentido ida: AA1SC, AB2SC, AC3SC, AD4RF, AE5VE, AF6SC y finalmente AG7BH.",
    "R2": """Existe una forma directa de llegar desde la estación AA1SC a la estación AG7BH.
    Toma la línea Amarilla en sentido de ida desde AA1SC hasta AG7BH.
    Las estaciones que cruzarás son: AA1SC → AB2SC → AC3SC → AD4RF → AE5VE → AF6SC → AG7BH."""
}

# Creamos una lista de “hechos atómicos”

hechos = [
    "Lo mas recomendable es coger la linea amarilla en sentido de ida \n Las estaciones que cruzarás son: AA1SC → AB2SC → AC3SC → AD4RF → AE5VE → AF6SC → AG7BH",
    "La linea que las une es la linea amarilla",
    "El sentido de la linea amarilla es ida"
]

# Evaluamos factualidad usando NLI (igual que antes)
for rid, resp in respuestas.items():
    print(f"\nEvaluando {rid}:")
    soportados = 0
    for h in hechos:
        label = check_entailment(resp, h)
        print(f" - {h} → {label}")
        if label == "entailment":
            soportados += 1
    factscore = soportados / len(hechos)
    print(f"FActScore aproximado: {factscore:.2f}")
