import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

dataset = pd.read_csv('./datasets/combined_dataset.csv')

sentiment_mapping = {'neutral': 0, 'negative': 1, 'positive': 2}
dataset['sentiment'] = dataset['sentiment'].map(sentiment_mapping)

# define X (text) and y (sentiment)
y = list(dataset['sentiment'])
X = list(dataset['text'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize the texts
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenizer(X_train)
test_encodings = tokenizer(X_test)

# make the datasets compatible with hugging face
train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

# tokenize the datasets
train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], padding="max_length", truncation=True, max_length=512), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], padding="max_length", truncation=True, max_length=512), batched=True)

# fix the formats for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# bert model initiation
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# set training arguments, play around with this
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch"
)

# provides us with gradient descent and loss functions automatically
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()