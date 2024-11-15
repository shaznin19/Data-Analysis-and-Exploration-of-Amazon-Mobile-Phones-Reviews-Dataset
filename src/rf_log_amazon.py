# -*- coding: utf-8 -*-
"""rf_log_amazon.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NzGplvXuv1kE5QTnwWiOCelANOeh1YoK
"""

from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, classification_report,precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

data_path = '/bsuhome/shazninsultana/Data_sci_project/Amazon_Unlocked_Mobile.csv'
df = pd.read_csv(data_path).reset_index(drop = True)


df["Brand Name"].fillna(value = "Missing", inplace = True)
df["Price"].fillna(value = 0, inplace = True)
df["Review Votes"].fillna(value = 0, inplace = True)
df = df.dropna(subset=['Reviews'])
df.isnull().sum()



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

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Calculate the accuracy of the model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'\nAccuracy: {rf_accuracy:.2f}')

# Print classification report for more detailed performance metrics
print("\nClassification Report_rf:")
print(classification_report(y_test, y_pred_rf))

# Assuming y_test and y_pred are already defined from your code
f1 = f1_score(y_test, y_pred_rf, average='weighted') # Use 'weighted' for multi-class
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')

print(f"F1 Score_rf: {f1}")
print(f"Precision_rf: {precision}")
print(f"Recall_rf: {recall}")

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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

plt.savefig('/bsuhome/shazninsultana/Data_sci_project/confusion_matrix.png')
# Get predicted probabilities for the positive class
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability estimates for class 1

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig('/bsuhome/shazninsultana/Data_sci_project/roc_auc.png')

# Assuming y_test and y_pred are already defined from your code
f1 = f1_score(y_test, y_pred, average='weighted') # Use 'weighted' for multi-class
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# Create a dictionary to store the metrics
metrics_dict = {
    'Metric': ['F1 Score', 'Precision', 'Recall'],
    'Value': [f1, precision, recall]
}

# Convert the dictionary into a pandas DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Save the DataFrame to a CSV file
metrics_df.to_csv('/bsuhome/shazninsultana/Data_sci_project/classification_metrics.csv', index=False)

print("Metrics saved to classification_metrics.csv")