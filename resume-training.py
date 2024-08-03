from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

# Specify the model and tokenizer you were using
model_name_or_path = "bert-base-uncased"  # Replace with your specific model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# load the checkpoint
checkpoint_path = "./results/checkpoint-75"  

# Load the model from the checkpoint
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# Reload the training arguments if needed
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_steps=50,
    save_total_limit=3
)

# Initialize the Trainer with the model, tokenizer, and training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Replace with your actual training dataset
    eval_dataset=eval_dataset     # Replace with your actual evaluation dataset
)

# Continue training from the checkpoint
trainer.train(resume_from_checkpoint=checkpoint_path)

# Alternatively, if you just want to use the model for inference or evaluation:
results = trainer.evaluate()
print(results)

# Now you can use the model for predictions
predictions = trainer.predict(test_dataset)  # Replace with your actual test dataset
print(predictions)