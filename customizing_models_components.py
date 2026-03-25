from transformers import AutoModelForCausalLM
import torch.nn as nn

# 1. Cargamos el chasis de nuestro mini-modelo de siempre
modelo = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Imprimimos cómo es la primera capa de atención
print("Pieza original:", modelo.transformer.h[0].mlp.act)

# 3. Cambiamos esa pieza por una función matemática diferente (ReLU)
# Esto hace que nuestro modelo se comporte de forma diferente, en concreto, 
# que las neuronas se activen de forma diferente.
modelo.transformer.h[0].mlp.act = nn.ReLU()

# 4. Comprobamos que el cambiazo ha funcionado
print("Pieza hackeada:", modelo.transformer.h[0].mlp.act)