import pickle
import matplotlib.pyplot as plt

metrics_save_path = "./training_metrics.pkl"
with open(metrics_save_path, 'rb') as f:
    metrics = pickle.load(f)

train_losses = metrics['train_losses']
train_accuracies = metrics['train_accuracies']
val_losses = metrics['val_losses']
val_accuracies = metrics['val_accuracies']

# plot the metrics
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')

plt.tight_layout()
plt.show()