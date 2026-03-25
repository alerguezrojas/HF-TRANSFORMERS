# Importamos la herramienta de los "planos" (Config) y el constructor del modelo
from transformers import AutoConfig, AutoModelForCausalLM

# 1. Descargamos SOLAMENTE los planos del modelo gpt2
planos = AutoConfig.from_pretrained("gpt2")

print(f"Capas originales del GPT-2: {planos.n_layer}")

# 2. ¡Modificamos los planos! Le decimos que solo queremos 2 capas
planos.n_layer = 2
print(f"Capas de nuestro modelo personalizado: {planos.n_layer}")

# 3. Construimos un modelo NUEVO y vacío usando nuestros planos modificados
# Ojo: Usamos .from_config() en lugar de .from_pretrained() porque 
# estamos construyendo el edificio desde cero, no descargando uno ya hecho.
mi_mini_modelo = AutoModelForCausalLM.from_config(planos)

print("¡Mini-modelo personalizado construido y listo para hacer pruebas!")

# Utilizamos el mni-modelo para generar texto a partir de un prompt
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_text = "El secreto para hornear un buen pastel es "
model_inputs = tokenizer([input_text], return_tensors="pt")
generated_ids = mi_mini_modelo.generate(**model_inputs, max_new_tokens=30)
resultado = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n--- RESULTADO DEL MINI-MODELO ---")
print(resultado)

