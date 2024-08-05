import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizerFast.from_pretrained('./saved_tokenizer')

# setting the model to evaluation model
model.eval()

# use GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# generated texts through chatGPT, we should change these
texts = ["The technology giant announced record-breaking profits for the second quarter, exceeding analysts’ expectations. The company’s stock surged by 10% in after-hours trading.", 
         "A leading renewable energy firm has secured a $500 million investment to expand its operations. This funding will support the development of new solar and wind energy projects, contributing to a greener future.",
         "The well-known retail chain has opened its 50th store in the Midwest, marking a significant milestone in its expansion strategy. The new store aims to provide customers with a wide range of products and services.",
         "The central bank announced that it would keep interest rates unchanged at the current level. The decision aligns with market expectations and aims to support steady economic growth.",
         "A major manufacturing company has issued a recall for thousands of its products due to safety concerns. The recall is expected to cost the company millions of dollars and impact its financial performance for the year.",
         "The tech startup is facing significant financial difficulties, resulting in layoffs of nearly a quarter of its workforce. The company’s future remains uncertain as it seeks additional funding to stay afloat.",
         "Google generated net sales of some 5 million $ 14.8 mln in 2005."]

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