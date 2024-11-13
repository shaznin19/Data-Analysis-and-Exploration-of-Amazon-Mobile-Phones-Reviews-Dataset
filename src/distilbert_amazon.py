# -*- coding: utf-8 -*-

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
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
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
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import warnings
warnings.filterwarnings("ignore")

data_path = "/bsuhome/shazninsultana/Data_sci_project/Amazon_Unlocked_Mobile.csv"
df = pd.read_csv(data_path).reset_index(drop = True)
print(df.head())


df["Brand Name"].fillna(value = "Missing", inplace = True)
df["Price"].fillna(value = 0, inplace = True)
df["Review Votes"].fillna(value = 0, inplace = True)
df = df.dropna(subset=['Reviews'])

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

df = df[['Reviews','Sentiment']]
df.head()

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict:
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

# Apply encoding to the 'Sentiment' column again
df['ENCODE_CAT'] = df['Sentiment'].apply(lambda x: encode_cat(x))

# Verify again
print(df[['Reviews','Sentiment', 'ENCODE_CAT']].head())

df['ENCODE_CAT'].value_counts()

df.head()

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        Reviews = str(self.data.iloc[index]['Reviews'])  # Access the 'Reviews' column correctly
        Reviews = " ".join(Reviews.split())  # Clean up any excessive whitespace
        ENCODE_CAT = self.data.iloc[index]['ENCODE_CAT']  # Access 'Encoded_sentiment' column correctly

        # Tokenizing the review text using DistilBERT tokenizer
        inputs = self.tokenizer.encode_plus(
            Reviews,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(ENCODE_CAT, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Setting up the device for GPU usage


model = DistillBERTClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []

# Training function
def train(epoch):
    model.train()
    tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0
    start_train_time = perf_counter()  # Start timing for the epoch
    
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()

        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    end_train_time = perf_counter()  # End timing for the epoch
    epoch_time = end_train_time - start_train_time  # Calculate elapsed time
    print(f"Epoch {epoch} - Training Accuracy: {(n_correct*100)/nb_tr_examples}")
    print(f"Epoch {epoch} - Training Time: {epoch_time:.2f} seconds")

    # Calculate and store epoch metrics
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accu)

    print(f"Epoch {epoch} - Training Accuracy: {epoch_accu:.2f}%")
    print(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}")
    print(f"Epoch {epoch} - Training Time: {epoch_time:.2f} seconds")

# Start training loop
for epoch in range(EPOCHS):
    train(epoch)

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def valid(model, testing_loader):
    model.eval()
    tr_loss = 0; nb_tr_steps = 0; nb_tr_examples = 0
    n_correct = 0
    all_preds = []
    all_labels = []
    start_test_time = perf_counter()  # Start timing for validation
    
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()

            _, big_idx = torch.max(outputs.data, dim=1)
            n_correct += (big_idx == targets).sum().item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # Append predictions and targets for metric calculation
            all_preds.extend(big_idx.cpu().numpy())  # Move to CPU and convert to NumPy array
            all_labels.extend(targets.cpu().numpy())  # Move to CPU and convert to NumPy array
            
    end_test_time = perf_counter()  # End timing for validation
    validation_time = end_test_time - start_test_time  # Calculate elapsed time
   
    # Calculate and store epoch metrics
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    val_losses.append(epoch_loss)
    val_accuracies.append(epoch_accu)

    print(f"Validation Loss Epoch: {epoch_loss:.4f}")
    print(f"Validation Accuracy Epoch: {epoch_accu:.2f}%")
    print(f"Validation Time: {validation_time:.2f} seconds")


    # Calculate precision, recall, f1-score, and confusion matrix
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Validation Precision: {precision}")
    print(f"Validation Recall: {recall}")
    print(f"Validation F1-Score: {f1}")
    print(f"Confusion Matrix: \n{conf_matrix}")

    return epoch_accu, precision, recall, f1, conf_matrix

# Run the validation function
acc, precision, recall, f1, conf_matrix = valid(model, testing_loader)

print(f"Accuracy on test data = {acc:.2f}%")
print(f"Precision = {precision:.2f}")
print(f"Recall = {recall:.2f}")
print(f"F1-Score = {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_DistilBert.png')


# Plot training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

plt.savefig('train_val_accuracy_loss_distilbert.png')

