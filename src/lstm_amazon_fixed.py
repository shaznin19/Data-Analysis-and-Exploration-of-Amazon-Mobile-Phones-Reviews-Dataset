# -*- coding: utf-8 -*-

import pandas as pd
from time import perf_counter
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, classification_report, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Load the dataset
data_path = "/data/Amazon_Unlocked_Mobile.csv"
df = pd.read_csv(data_path).reset_index(drop = True)

df["Brand Name"].fillna(value = "Missing", inplace = True)
df["Price"].fillna(value = 0, inplace = True)
df["Review Votes"].fillna(value = 0, inplace = True)
df = df.dropna(subset=['Reviews'])
df.isnull().sum()

#Load the stop words from nltk
stop_words = set(stopwords.words('english'))

# Function to preprocess a single review
def preprocess_review(review):
    # Convert all letters to lowercase
    review = review.lower()

    # Remove specific characters: "-", "/", ":", "?"
    review = re.sub(r"[-/:?]", "", review)
    # review = review.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words
    review = ' '.join([word for word in review.split() if word not in stop_words])

    # Remove words that start with '@'
    review = ' '.join([word for word in review.split() if not word.startswith('@')])

    # Replace repeated letters (more than 2 times) with 2 occurrences
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)
    return review

df['preprocessed_reviews'] = df['Reviews'].apply(preprocess_review)

# Calculating percentage reduction in feature size
def feature_size_reduction(original_reviews, preprocessed_reviews):
    original_words = Counter(' '.join(original_reviews).split())
    preprocessed_words = Counter(' '.join(preprocessed_reviews).split())
    original_size = len(original_words)
    preprocessed_size = len(preprocessed_words)
    reduction_percentage = (original_size - preprocessed_size) / original_size * 100
    return original_size, preprocessed_size, reduction_percentage

original_size, preprocessed_size, reduction_percentage = feature_size_reduction(df['Reviews'], df['preprocessed_reviews'])

def get_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['Sentiment'] = df['Rating'].apply(get_sentiment)
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)
df = df[['Reviews','Sentiment']]
encode_dict = {}
def encode_cat(x):
    if x not in encode_dict:
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['Sentiment'].apply(lambda x: encode_cat(x))
print(df[['Reviews','Sentiment', 'ENCODE_CAT']].head())
df['ENCODE_CAT'].value_counts()

# Defining hyperparameters
input_length = 150
max_fatures = 2000
embed_dim = 128
lstm_out = 196
batch_size = 512

# Tokenize the data
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['Reviews'].values)
X = tokenizer.texts_to_sequences(df['Reviews'].values)
X = pad_sequences(X, maxlen=input_length)
X.shape

# Creating the LSTM model
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(batch_size, X.shape[1]))
print(model.summary())

Y = df['ENCODE_CAT'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

Y_train = to_categorical(Y_train, num_classes=3)
Y_test = to_categorical(Y_test, num_classes=3)
Y_test.shape
X_test.shape

# Calculate training time
start_train_time = perf_counter()
# Training the model
history = model.fit(X_train, Y_train, epochs=15, batch_size=batch_size, verbose=2, validation_split=0.1)

end_train_time = perf_counter()
epoch_time = end_train_time - start_train_time
print(f"Training Time: {epoch_time:.2f} seconds")

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('/results/train_val_loss_accuracy_lstm.png')

# Calculate testing time
start_test_time = perf_counter()
# Testing the model
Y_pred = model.predict(X_test, batch_size=batch_size, verbose=2)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = Y_test
Y_true = np.argmax(Y_true, axis=1)

end_test_time = perf_counter()
testing_time = end_test_time - start_test_time 
print(f"Testing Time: {testing_time:.2f} seconds")

# Calculate Metrics
accuracy = accuracy_score(Y_true, Y_pred_classes)
precision = precision_score(Y_true, Y_pred_classes, average='weighted') 
recall = recall_score(Y_true, Y_pred_classes, average='weighted')
f1 = f1_score(Y_true, Y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

print("\nClassification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=['Positive', 'Negative', 'Neutral']))

cm = confusion_matrix(Y_true, Y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'],
            yticklabels=['Positive', 'Negative', 'Neutral'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
plt.savefig('/results/confusion_matrix_lstm.png')