from transformers import AutoModelForCausalLM
import torch

# Cargamos el modelo, pero le decimos explícitamente qué "motor de atención" usar
modelo_rapido = AutoModelForCausalLM.from_pretrained(
    "gpt2", 
    torch_dtype=torch.float16, # FlashAttention requiere un formato numérico específico
    attn_implementation="flash_attention_2" # ¡AQUÍ ESTÁ LA MAGIA!
)

print("¡Modelo cargado con el motor FlashAttention hiper-rápido!")