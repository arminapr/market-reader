import pickle
import matplotlib.pyplot as plt

# retrieve the training metrics
metrics_save_path = 'training_metrics.pkl'
with open(metrics_save_path, 'rb') as f:
    metrics = pickle.load(f)

# retrieve the training/validation losses and accuracies
train_losses = metrics['train_losses']
train_accuracies = metrics['train_accuracies']
val_losses = metrics['val_losses']
val_accuracies = metrics['val_accuracies']

# plot for losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# plot for accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
