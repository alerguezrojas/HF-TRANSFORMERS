from transformers import pipeline
from accelerate import Accelerator
import torch

# 1. Detectar el dispositivo automáticamente
device = Accelerator().device
print(f"Ejecutando en: {device}")

# 2. Cargar el modelo de segmentación
# Cambiamos el nombre de la variable de 'pipeline' a 'segmenter' para evitar errores
segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device=device)

# 3. Procesar la imagen de los loros (parrots)
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
segments = segmenter(url)

# 4. Ver los resultados
print(f"\nSe han detectado {len(segments)} segmentos.")

# 5. Mostrar detalles de cada segmento
for i, segment in enumerate(segments):
    print(f"\nSegmento {i + 1}:")
    print(f"  Etiqueta: {segment['label']}")

# 6. Confirmación extra de la GPU
print(f"\n--- Memoria reservada en la GPU: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB ---")

