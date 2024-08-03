import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress info and warning messages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers, losses
import time

# Load data
df = pd.read_csv('datasets/combined_dataset.csv')

y = df['sentiment']
X = df['text']

# Encode the labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=123)

tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(X_train)

X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

vocabulary_size = len(tfidf_vectorizer.vocabulary_)

# Create model object
nn_model = models.Sequential()

# Create the input layer and add it to the model object: 
input_layer = layers.InputLayer(input_shape=(vocabulary_size,))
nn_model.add(input_layer)

# Create the hidden layers and add them to the model object:
hidden1 = layers.Dense(units=64, activation='relu')
nn_model.add(hidden1)

hidden2 = layers.Dense(units=32, activation='relu')
nn_model.add(hidden2)

hidden3 = layers.Dense(units=16, activation='relu')
nn_model.add(hidden3)

# Create the output layer and add it to the model object:
output_layer = layers.Dense(units=3, activation='softmax')  # 3 classes
nn_model.add(output_layer)

# Print summary of neural network model structure
nn_model.summary()

# Compile the model
sgd_optimizer = optimizers.SGD(learning_rate=0.15)
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)  # Use SparseCategoricalCrossentropy for integer labels
nn_model.compile(optimizer=sgd_optimizer, loss=loss_fn, metrics=['accuracy'])

# Define the progress bar logger
class ProgBarLoggerNEpochs(keras.callbacks.Callback):
    
    def __init__(self, num_epochs: int, every_n: int = 50):
        self.num_epochs = num_epochs
        self.every_n = every_n
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            s = 'Epoch [{}/ {}]'.format(epoch + 1, self.num_epochs)
            logs_s = ['{}: {:.4f}'.format(k.capitalize(), v)
                      for k, v in logs.items()]
            s_list = [s] + logs_s
            print(', '.join(s_list))

# Train the model
t0 = time.time()  # start time

num_epochs = 75  # epochs
nn_model.add(layers.Dropout(0.15))

history = nn_model.fit(
    X_train_tfidf.toarray(),
    y_train,
    epochs=num_epochs,
    verbose=0,
    validation_split=0.2,
    callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=5)]
)

t1 = time.time()  # stop time

print('Elapsed time: %.2fs' % (t1-t0))

# Plot training and validation loss
plt.plot(range(1, num_epochs + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, num_epochs + 1), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(range(1, num_epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = nn_model.evaluate(X_test_tfidf.toarray(), y_test)
print('Loss:', str(loss), 'Accuracy:', str(accuracy))

# Predict and interpret results
probability_predictions = nn_model.predict(X_test_tfidf.toarray())
y_test_array = y_test

# Convert indices to class labels
class_labels = label_encoder.classes_

for i in range(20):
    probabilities = probability_predictions[i]  # Probabilities for each class
    predicted_class = np.argmax(probabilities)  # Get index of highest probability
    actual_label = y_test_array[i]
    
    print(f"Example {i + 1}:")
    print(f"  Predicted Probabilities: {probabilities}")
    print(f"  Predicted Class: {class_labels[predicted_class]}")
    print(f"  Actual Label: {class_labels[actual_label]}")
    print()
