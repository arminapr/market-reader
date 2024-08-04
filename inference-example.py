import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('./saved_model/config.json')
tokenizer = BertTokenizerFast.from_pretrained('./saved_tokenizer')

# setting the model to evaluation model
model.eval()

# use GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# generated texts through chatGPT, we should change these
texts = ["The company reported a record increase in profits.", "The product quality has been poor."]

# tokenize the text
encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

# perform inference
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

sentiment_mapping = {0: 'neutral', 1: 'negative', 2: 'positive'}
predicted_labels = [sentiment_mapping[prediction.item()] for prediction in predictions]

print(predicted_labels)