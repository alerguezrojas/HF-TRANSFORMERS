from transformers import pipeline
import torch

# 1. Creamos el clasificador y lo enviamos a la GPU (device=0)
# La primera vez que lo corras, descargará el modelo (unos cientos de MB)
classifier = pipeline("sentiment-analysis", device=0)

# 2. Probamos con un par de frases
frases = [
    "I am amazed by how easy it is to use Transformers with my GPU!",
    "I'm a bit frustrated because the installation was tricky, but now it works."
]

resultados = classifier(frases)

# 3. Mostramos los resultados
for frase, res in zip(frases, resultados):
    print(f"\nFrase: {frase}")
    print(f"Resultado: {res['label']} (Confianza: {res['score']:.4f})")

# 4. Confirmación extra de la GPU
print(f"\n--- Memoria reservada en la GPU: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB ---")