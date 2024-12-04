# Sentiment Analysis on Amazon Mobile Phone Reviews

## Project Overview
 This project uses transformer models like DistilBERT and ELECTRA, alongside traditional algorithms like Logistic Regression and Random Forest, to tackle the complexities of sentiment analysis. Through these approaches, this project aims to explore the role of SA in transforming unstructured data into strategic insights that empower businesses to thrive in competitive markets.

## Dataset
The dataset used for this project is the Amazon Unlocked
Mobile Phones Dataset, a comprehensive collection of over
400,000 customer reviews of unlocked mobile phones sold
on Amazon. It contains information such as:
- Review text
- Rating
- Review date
- Reviewer details
- Product details

## Objectives
1. Utilize machine learning and deep learning models to classify customer reviews as positive, neutral, or negative. Focus on implementing advanced models such as DistilBERT, ELECTRA, Logistic Regression, Random Forest, and LSTM to achieve robust classification.
2. Compare the performance of these models to identify the most effective model for sentiment classification and understand their performance and computational efficiency.
3. Analyze patterns and trends in customer feedback to gain deeper insights into product quality, customer satisfaction, and potential areas for improvement. 
4. Use the results of sentiment analysis to empower businesses in making data-informed decisions

## Tools and Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Natural Language Toolkit (NLTK)

## Project Structure
```
/Sentiment-Analysis-on-Amazon-Mobile-Phone-Reviews
│
├── data
├── src
│   ├── distilbert_amazon.py
│   ├── electra_amazon.py
│   ├── electra_amazon_fixed.py
│   ├── rf_amazon.py
│   ├── logreg_amazon.py
├── borah-scripts
│   ├── run_distilbert.sh
│   ├── run_electra.sh
│   ├── run_script.sh
├── results
├── README.md
```

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/shaznin19/Sentiment-Analysis-on-Amazon-Mobile-Phone-Reviews.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Sentiment-Analysis-on-Amazon-Mobile-Phone-Reviews
    ```

## Usage
1. Run sentiment analysis using DistilBert:
    ```bash
    python src/distilbert_amazon.py
    ```
2. Run sentiment analysis using Electra:
    ```bash
    python src/electra_amazon_fixed.py
    ```
3. Run sentiment analysis using Random Forest:
    ```bash
    python src/rf_log_amazon.py
    ```
4. Run sentiment analysis using Logistic Regression:
    ```bash
    python src/rf_log_amazon.py
    ```
