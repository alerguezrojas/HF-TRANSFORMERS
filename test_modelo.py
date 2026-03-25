from transformers import pipeline

# 1. Cargamos TU modelo (el que acabas de entrenar)
# En lugar de un nombre de Hugging Face, le damos la ruta de tu carpeta
model_path = "./modelo_final_critico"
pipe = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0)

# 2. Vamos a darle un par de críticas inventadas por nosotros
criticas = [
    "The acting was subpar and the plot was predictable. A total waste of time.",
    "Absolutely breathtaking! The cinematography was out of this world and the lead actor deserves an Oscar.",
    "It was okay, but I've seen better movies this year."
]

print("\n--- RESULTADOS DEL MODELO ENTRENADO POR TI ---")

for critica in criticas:
    resultado = pipe(critica)[0]
    sentimiento = "POSITIVO" if resultado['label'] == 'LABEL_1' else "NEGATIVO"
    confianza = resultado['score'] * 100
    
    print(f"\nCrítica: {critica}")
    print(f"Predicción: {sentimiento} ({confianza:.2f}% de confianza)")