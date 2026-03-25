# 1. Cargamos el modelo para generar texto (Causal Language Modeling)
from transformers import AutoModelForCausalLM # Importamos la clase para modelos de generación de texto

modelo_generador = AutoModelForCausalLM.from_pretrained("gpt2") # Cargamos el modelo GPT-2, que es un modelo pequeño y fácil de manejar para aprender
print("Este modelo está listo para escribir texto.")

# 2. Cargamos EL MISMO modelo, pero preparado para clasificar textos
from transformers import AutoModelForSequenceClassification # Importamos la clase para modelos de clasificación de texto 

modelo_clasificador = AutoModelForSequenceClassification.from_pretrained("gpt2") # Cargamos el mismo modelo GPT-2, pero esta vez lo preparamos para clasificar textos.
print("Este modelo está listo para clasificar.")
