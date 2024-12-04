# -*- coding: utf-8 -*-

import pandas as pd
from time import perf_counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from collections import Counter
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = "/data/Amazon_Unlocked_Mobile.csv"
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


# We'll assume ratings of 4 or 5 are positive, 3 is neutral, and 1 or 2 are negative

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

df = df[['Reviews','Sentiment']]
df.head()

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict:
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['Sentiment'].apply(lambda x: encode_cat(x))

print(df[['Reviews','Sentiment', 'ENCODE_CAT']].head())

df['ENCODE_CAT'].value_counts()

df.head()

# Defining hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        Reviews = str(self.data.iloc[index]['Reviews'])
        Reviews = " ".join(Reviews.split()) 
        ENCODE_CAT = self.data.iloc[index]['ENCODE_CAT']  
        
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

train_size = 0.6
val_size = 0.2
train_dataset = df.sample(frac=train_size, random_state=200)
remaining_dataset = df.drop(train_dataset.index).reset_index(drop=True)
val_dataset = remaining_dataset.sample(frac=val_size / (1 - train_size), random_state=200)
test_dataset = remaining_dataset.drop(val_dataset.index).reset_index(drop=True)

train_dataset = train_dataset.reset_index(drop=True)
val_dataset = val_dataset.reset_index(drop=True)
test_dataset = test_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VALIDATION Dataset: {}".format(val_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
validation_set = Triage(val_dataset, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **valid_params)
testing_loader = DataLoader(testing_set, **test_params)

# Creating the model

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

model = DistillBERTClass()
model.to(device)

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

# Train function
def train(epoch):
    model.train()
    tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0

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

        # Print training progress every 5000 steps
        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step:.4f}")
            print(f"Training Accuracy per 5000 steps: {accu_step:.2f}%")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate training metrics
    epoch_train_loss = tr_loss / nb_tr_steps
    epoch_train_accu = (n_correct * 100) / nb_tr_examples
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accu)

    print(f"Epoch {epoch} - Training Accuracy: {epoch_train_accu:.2f}%")
    print(f"Epoch {epoch} - Training Loss: {epoch_train_loss:.4f}")

    # Validation Loop
    model.eval()
    val_loss, n_correct_val, nb_val_steps, nb_val_examples = 0, 0, 0, 0

    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()

            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct_val += calcuate_accu(big_idx, targets)
            nb_val_steps += 1
            nb_val_examples += targets.size(0)

    # Calculate validation metrics
    epoch_val_loss = val_loss / nb_val_steps
    epoch_val_accu = (n_correct_val * 100) / nb_val_examples
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accu)

    print(f"Epoch {epoch} - Validation Accuracy: {epoch_val_accu:.2f}%")
    print(f"Epoch {epoch} - Validation Loss: {epoch_val_loss:.4f}")

    return epoch_train_loss, epoch_train_accu, epoch_val_loss, epoch_val_accu

# Calculate training time
start_train_time = perf_counter()
for epoch in range(EPOCHS):
    train(epoch)
    
end_train_time = perf_counter()
epoch_time = end_train_time - start_train_time

print(f"Training Time: {epoch_time:.2f} seconds")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

plt.savefig('/results/train_val_loss_distilbert.png')

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.savefig('/results/train_val_accuracy_distilbert.png')

# Testing Function
def test(model, testing_loader):
    model.eval()
    tr_loss = 0; nb_tr_steps = 0; nb_tr_examples = 0
    n_correct = 0
    all_preds = []
    all_labels = []
    # Calculate testing time
    start_test_time = perf_counter() 
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

            all_preds.extend(big_idx.cpu().numpy()) 
            all_labels.extend(targets.cpu().numpy())

    end_test_time = perf_counter() 
    testing_time = end_test_time - start_test_time  
    print(f"Testing Time: {testing_time:.2f} seconds")

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    
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

acc, precision, recall, f1, conf_matrix = test(model, testing_loader)

print(f"Accuracy on test data = {acc:.2f}%")
print(f"Precision = {precision:.2f}")
print(f"Recall = {recall:.2f}")
print(f"F1-Score = {f1:.2f}")
print("Confusion Matrix Distilbert:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Distilbert')
plt.show()
plt.savefig('/results/confusion_matrix_distilbert.png')

