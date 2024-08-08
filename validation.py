import torch
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

# use GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the dataset
dataset = pd.read_csv('./datasets/combined_dataset.csv')

# Map sentiments to numerical values
sentiment_mapping = {'neutral': 0, 'negative': 1, 'positive': 2}
dataset['sentiment'] = dataset['sentiment'].map(sentiment_mapping)

# Define X (text) and y (sentiment)
y = list(dataset['sentiment'])
X = list(dataset['text'])

# Load the pre-trained BERT model and tokenizer
model_id = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_id)
model = BertForSequenceClassification.from_pretrained("./saved_model").to(device)

# Tokenize the dataset
encodings = tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
labels = torch.tensor(y)

dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

BATCH_SIZE = 16
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation function
def evaluate(model, dataloader, device):
    print("evaluating")
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# Evaluate the model
val_loss, val_accuracy = evaluate(model, test_loader, device)
print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")