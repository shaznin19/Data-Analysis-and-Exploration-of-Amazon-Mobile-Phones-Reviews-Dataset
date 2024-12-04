# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, classification_report,precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = '/data/Amazon_Unlocked_Mobile.csv'
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

df['Sentiment'] = df['Rating'].apply(get_sentiment)

print("\nFirst few rows with sentiment derived from rating:")
print(df.head())

sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)

df = df[['Reviews','Rating','Sentiment']]

# Build a feature matrix
vectorizer = TfidfVectorizer()
tfidf_text = vectorizer.fit_transform(df['Reviews'])
X = tfidf_text
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(df['Sentiment'])
y = encoded_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.2f}')

# Print classification report
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
plt.savefig('/results/confusion_matrix_logreg.png')

f1 = f1_score(y_test, y_pred, average='weighted') 
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")