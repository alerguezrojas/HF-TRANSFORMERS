# Importamos la herramienta para cargar modelos de texto
from transformers import AutoModelForCausalLM

# Aquí ocurre la magia. Al poner device_map="auto", 
# activamos la carga inteligente y dinámica. 
# Si el modelo es muy grande, lo dividirá automáticamente.
modelo = AutoModelForCausalLM.from_pretrained(
    "gpt2", # Usamos GPT-2, un modelo pequeñito ideal para aprender
    device_map="auto" # Esto le dice a Transformers que gestione la memoria de forma inteligente, 
                      # es decir, que cargue partes del modelo en la GPU y otras en la CPU según sea necesario.
)

print("¡Modelo cargado de forma inteligente!")

