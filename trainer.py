from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
import torch
import os

# Evitar el aviso de symlinks en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. CARGAR MODELO (Cerebro base)
model_id = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# 2. CARGAR Y TRADUCIR DATASET
dataset = load_dataset("rotten_tomatoes")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. PLAN DE ENTRENAMIENTO (Corregido)
training_args = TrainingArguments(
    output_dir="./mi_modelo_cine",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",    # <--- ANTES ERA evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                # Usa tu RTX 3060 a tope
    report_to="none"          # Para que no te pida cuentas de trackers externos
)

# 4. ENTRENADOR
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer, # En versiones nuevas se usa processing_class en vez de tokenizer
    data_collator=data_collator,
)

# ¡A VOLAR!
print("\nIniciando entrenamiento en tu GPU...")
trainer.train()

# Guardar
trainer.save_model("./modelo_final_critico")
print("\n¡Hecho! El modelo 'crítico de cine' está guardado en './modelo_final_critico'")