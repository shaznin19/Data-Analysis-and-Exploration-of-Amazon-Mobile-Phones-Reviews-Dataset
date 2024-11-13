# -*- coding: utf-8 -*-
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from time import perf_counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertModel, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModel
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, classification_report, f1_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import namedtuple
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
import time
from time import perf_counter
from wordcloud import WordCloud
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

data_path = "/bsuhome/shazninsultana/Data_sci_project/Amazon_Unlocked_Mobile.csv"
df = pd.read_csv(data_path).reset_index(drop = True)
print(df.head())

df.describe()
df.info()
df.isnull().sum()

df["Brand Name"].fillna(value = "Missing", inplace = True)
df["Price"].fillna(value = 0, inplace = True)
df["Review Votes"].fillna(value = 0, inplace = True)
df = df.dropna(subset=['Reviews'])
df.isnull().sum()

df.info()

# Load the stop words from nltk
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

    # 4. Remove words that start with '@'
    review = ' '.join([word for word in review.split() if not word.startswith('@')])

    # 5. Replace repeated letters (more than 2 times) with 2 occurrences
    review = re.sub(r'(.)\1{2,}', r'\1\1', review)

    return review

# Apply preprocessing to the 'reviews' column
df['preprocessed_reviews'] = df['Reviews'].apply(preprocess_review)

# Example: Calculating percentage reduction in feature size
def feature_size_reduction(original_reviews, preprocessed_reviews):
    original_words = Counter(' '.join(original_reviews).split())
    preprocessed_words = Counter(' '.join(preprocessed_reviews).split())

    original_size = len(original_words)
    preprocessed_size = len(preprocessed_words)

    reduction_percentage = (original_size - preprocessed_size) / original_size * 100

    return original_size, preprocessed_size, reduction_percentage

# Calculate feature size reduction
original_size, preprocessed_size, reduction_percentage = feature_size_reduction(df['Reviews'], df['preprocessed_reviews'])

print(f"Original feature size: {original_size}")
print(f"Preprocessed feature size: {preprocessed_size}")
print(f"Percentage reduction in feature size: {reduction_percentage:.2f}%")

df.info()

df.describe()

# We'll assume ratings of 4 or 5 are positive, 3 is neutral, and 1 or 2 are negative

def get_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

# Apply the sentiment derivation function
df['Sentiment'] = df['Rating'].apply(get_sentiment)

# Display the first few rows with sentiment
print("\nFirst few rows with sentiment derived from rating:")
print(df.head())

sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)

df = df[['Reviews','Rating','Sentiment']]

# Build a feature matrix (30 pts)

vectorizer = TfidfVectorizer()
tfidf_text = vectorizer.fit_transform(df['Reviews'])

X = tfidf_text
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(df['Sentiment'])
y = encoded_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DataFrame with the Sentiment and its corresponding encoded label
temp_df = pd.DataFrame({'Sentiment': df['Sentiment'], 'encoded_labels': encoded_labels})

# Display the unique encoded labels for each sentiment
print(temp_df.groupby('Sentiment')['encoded_labels'].unique())

model = RandomForestClassifier(n_estimators = 100, class_weight='balanced')
start_train_time = perf_counter()
model.fit(X_train, y_train)
end_train_time = perf_counter()
training_time = end_train_time - start_train_time
print(f"Training Time: {training_time:.2f} seconds")

start__test_time = perf_counter()
y_pred = model.predict(X_test)
end_test_time = perf_counter()
prediction_time = end_test_time - start_test_time
print(f"Prediction Time: {prediction_time:.2f} seconds")


# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.2f}')

# Print classification report for more detailed performance metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_RF.png')

# Assuming y_test and y_pred are already defined from your code
f1 = f1_score(y_test, y_pred, average='weighted') # Use 'weighted' for multi-class
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")