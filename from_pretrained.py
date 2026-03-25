from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Elegimos un modelo que SÍ quepa en tus 6GB de RAM y sea público
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

print("Cargando el traductor (Tokenizer)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Cargando el cerebro (Modelo) en la GPU...")
# device_map="auto" envía el modelo a tu RTX 3060 automáticamente
# torch_dtype="auto" usa la precisión óptima para tu tarjeta
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype="auto"
)

# 2. PREPARACIÓN: Convertimos texto a números (Tensors)
input_text = "The secret to baking a good cake is "
model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

print(f"El texto se ha convertido a estos números: {model_inputs['input_ids']}")

# 3. GENERACIÓN: El modelo calcula los siguientes números (palabras)
print("El modelo está pensando...")
generated_ids = model.generate(**model_inputs, max_new_tokens=30)

# 4. DECODIFICACIÓN: Convertimos los números resultantes otra vez a texto humano
resultado = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- RESULTADO FINAL ---")
print(resultado)