📰 AG News Topic Classification using NLP & Machine Learning

An end-to-end Natural Language Processing (NLP) and Machine Learning project that automatically classifies news articles into four major categories:

🌍 World
⚽ Sports
💼 Business
💻 Sci/Tech

The project uses TF-IDF feature extraction with Logistic Regression and Multinomial Naive Bayes classifiers. It also supports predicting the category of a completely new news article entered by the user.

📌 Project Overview

News websites publish thousands of articles every day, making automatic organization and categorization useful for search, recommendation systems, content management, and news aggregation.

This project builds a text classification system that learns patterns from the AG News dataset and predicts the topic of unseen news articles.

Classification Categories
Label	Category
1	🌍 World
2	⚽ Sports
3	💼 Business
4	💻 Sci/Tech
🎯 Objectives

The main objectives of this project are to:

Load and analyze the AG News dataset.
Perform exploratory data analysis.
Clean and preprocess news text.
Combine article titles and descriptions.
Convert text into numerical features using TF-IDF.
Compare unigram and unigram + bigram representations.
Train Logistic Regression and Multinomial Naive Bayes models.
Evaluate models using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Analyze classification errors.
Test the model on new fictional news articles.
Accept user-provided news and predict its category.
Save the trained model and TF-IDF vectorizer for future use.
📂 Dataset

The project uses the AG News Classification Dataset.

The dataset contains:

120,000 training articles
7,600 test articles
4 categories
Title and description text for each article
Dataset Structure
train.csv
test.csv

Each file contains:

Class Index
Title
Description

The columns are renamed in the project as:

label
title
description

The final text used for classification is:

text = title + description
🛠️ Technologies Used
Technology	Purpose
Python	Programming language
Pandas	Data manipulation
NumPy	Numerical operations
NLTK	Text preprocessing
Scikit-learn	Machine Learning
Matplotlib	Visualization
Seaborn	Data visualization
WordCloud	Word frequency visualization
Joblib	Model serialization
Google Colab	Development environment
🔄 Project Workflow
                AG News Dataset
                       │
                       ▼
               Data Loading
                       │
                       ▼
               Data Analysis
                       │
                       ▼
          Title + Description
                       │
                       ▼
             Text Preprocessing
                       │
        ┌──────────────┴──────────────┐
        │                             │
     Lowercase                 Remove Noise
        │                             │
        └──────────────┬──────────────┘
                       ▼
            Stopword Removal
                       │
                       ▼
               Lemmatization
                       │
                       ▼
                TF-IDF
             ┌─────────┴─────────┐
             │                   │
          Unigrams         Unigrams+Bigrams
             │                   │
             └─────────┬─────────┘
                       ▼
             Machine Learning
             ┌─────────┴─────────┐
             │                   │
       Logistic Regression   Naive Bayes
             │                   │
             └─────────┬─────────┘
                       ▼
                 Evaluation
                       │
                       ▼
              Final Classifier
                       │
                       ▼
              New News Article
                       │
                       ▼
              Predicted Category
🧹 Text Preprocessing

The raw news articles are cleaned before feature extraction.

Preprocessing Steps
Convert text to lowercase.
Remove HTML tags.
Remove URLs.
Remove punctuation.
Remove special characters.
Remove extra whitespace.
Tokenize the text.
Remove English stopwords.
Remove very short words.
Apply lemmatization.
Example

Original:

Apple Inc. announced a NEW AI processor at
https://example.com today!

After preprocessing:

apple inc announced new ai processor today
Why preprocessing?

Preprocessing reduces unnecessary variations and noise in the text, allowing the machine learning models to focus on useful words and phrases.

📊 Exploratory Data Analysis

The project performs several EDA tasks:

Class Distribution

The number of articles in each category is analyzed and visualized.

Text Length

The number of words in each article is calculated to understand the distribution of article lengths.

Most Frequent Words

The most common words are extracted separately for each category.

Word Clouds

Word clouds are generated for:

World
Sports
Business
Sci/Tech

These visualizations help identify vocabulary patterns associated with each topic.

🔢 TF-IDF Feature Extraction

Machine learning algorithms cannot directly process raw text, so the cleaned articles are converted into numerical vectors using TF-IDF.

TF-IDF stands for:

Term Frequency — Inverse Document Frequency

It gives higher importance to words that are useful for distinguishing documents while reducing the importance of extremely common words.

Experiment 1 — Unigrams
TfidfVectorizer(
    ngram_range=(1,1)
)

This considers individual words.

Example:

stock
market
company
football
Experiment 2 — Unigrams + Bigrams
TfidfVectorizer(
    ngram_range=(1,2)
)

This considers individual words and two-word combinations.

Example:

stock
market
stock market
financial market

Bigrams can capture useful phrases that individual words may not represent well.

🤖 Machine Learning Models

Two machine learning algorithms are compared.

1. Logistic Regression

Logistic Regression is used as a linear classifier for multi-class text classification.

It learns relationships between TF-IDF features and the four news categories.

2. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic algorithm commonly used for text classification.

It estimates the probability of each category based on the words present in the article.

🧪 Train/Validation Split

The training dataset is divided using:

train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=labels
)

This creates:

80% → Training
20% → Validation

The same split is used for comparing the models.

The separate official AG News test.csv is kept untouched and used for final evaluation.

📈 Model Evaluation

The models are evaluated using:

Accuracy

Percentage of correctly classified articles.

Precision

Measures how many articles predicted as a category actually belong to that category.

Recall

Measures how many articles belonging to a category were correctly identified.

F1-Score

Harmonic mean of precision and recall.

Confusion Matrix

Shows which categories are correctly classified and which categories are confused with one another.

🔬 Experiments

The project compares four configurations:

Experiment	Features	Model
1	Unigrams	Logistic Regression
2	Unigrams	Multinomial Naive Bayes
3	Unigrams + Bigrams	Logistic Regression
4	Unigrams + Bigrams	Multinomial Naive Bayes

The results are compared using:

Accuracy
Precision
Recall
F1 Score

The model configuration with the strongest measured validation performance can then be selected for the final classifier.
