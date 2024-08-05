import torch
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
import pickle

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
BATCH_SIZE = 16

dataset = pd.read_csv('./datasets/combined_dataset.csv')

sentiment_mapping = {'neutral': 0, 'negative': 1, 'positive': 2}
dataset['sentiment'] = dataset['sentiment'].map(sentiment_mapping)

# define X (text) and y (sentiment)
y = list(dataset['sentiment'])
X = list(dataset['text'])

# print(dataset)
print("loading a pre-trained BERT model")
model_id = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_id)

print("tokenizing the dataset")
# tokenize the dataset
encodings = tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
labels = torch.tensor(y)

print("splitting the dataset")
dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# using BERT for sequence classification model
model = BertForSequenceClassification.from_pretrained(model_id, num_labels=3).to(device)

print("defining optimizer")
# define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

print("training starting")
# training and evaluation functions
def train_epoch(model, dataloader, optimizer, device):
    print("training")
    model.train()
    total_loss = 0
    correct = 0
    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

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

# lists to store the values
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_accuracy = evaluate(model, test_loader, device)
    
    # append the losses/accuracy to the lists
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

model_save_path = "./saved_model"
tokenizer_save_path = "./saved_tokenizer"

# save the model and tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

# save the training and validation losses and accuracies
metrics_save_path = "./training_metrics.pkl"
with open(metrics_save_path, 'wb') as f:
    pickle.dump({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, f)