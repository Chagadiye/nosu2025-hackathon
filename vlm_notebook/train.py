import torch
from transformers import QwenTokenizer, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("mdwiratathya/ROCO-radiology", cache_dir="./cache")

# Load the tokenizer and model
tokenizer = QwenTokenizer.from_pretrained("Qwen/Qwen2VL")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2VL")

# Check if MPS is available for Mac M1 GPU
device = torch.device("mps") if torch.has_mps else torch.device("cpu")
model.to(device)

# Tokenization function
def tokenize_function(example):
    inputs = tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    save_total_limit=2,
    save_steps=500,
    learning_rate=5e-5,
    num_train_epochs=3,
    load_best_model_at_end=True,
    logging_steps=100,
    report_to="tensorboard",
    optim="adamw_torch",
    fp16=False,  # MPS on Mac M1 does not support fp16
    bf16=False,  # BF16 not supported on MPS yet
    dataloader_pin_memory=False,  # Prevent potential MPS issues
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Fine-tune the model
if __name__ == "__main__":
    trainer.train()

