# 📰 AG News Topic Classification using NLP & Machine Learning

An end-to-end **Natural Language Processing (NLP) and Machine Learning** project that automatically classifies news articles into four major categories:

* 🌍 **World**
* ⚽ **Sports**
* 💼 **Business**
* 💻 **Sci/Tech**

The project uses **TF-IDF feature extraction** with **Logistic Regression** and **Multinomial Naive Bayes** classifiers. It also supports predicting the category of a completely new news article entered by the user.

---

## 📌 Project Overview

News websites publish thousands of articles every day, making automatic organization and categorization useful for search, recommendation systems, content management, and news aggregation.

This project builds a text classification system that learns patterns from the **AG News dataset** and predicts the topic of unseen news articles.

### Classification Categories

| Label | Category    |
| ----: | ----------- |
|     1 | 🌍 World    |
|     2 | ⚽ Sports    |
|     3 | 💼 Business |
|     4 | 💻 Sci/Tech |

---

## 🎯 Objectives

The main objectives of this project are to:

* Load and analyze the AG News dataset.
* Perform exploratory data analysis.
* Clean and preprocess news text.
* Combine article titles and descriptions.
* Convert text into numerical features using **TF-IDF**.
* Compare **unigram** and **unigram + bigram** representations.
* Train Logistic Regression and Multinomial Naive Bayes models.
* Evaluate models using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
* Analyze classification errors.
* Test the model on new fictional news articles.
* Accept user-provided news and predict its category.
* Save the trained model and TF-IDF vectorizer for future use.

---

## 📂 Dataset

The project uses the **AG News Classification Dataset**.

The dataset contains:

* **120,000 training articles**
* **7,600 test articles**
* **4 categories**
* Title and description text for each article

### Dataset Structure

```text
train.csv
test.csv
```

Each file contains:

```text
Class Index
Title
Description
```

The columns are renamed in the project as:

```text
label
title
description
```

The final text used for classification is:

```python
text = title + description
```

---

## 🛠️ Technologies Used

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Programming language         |
| Pandas       | Data manipulation            |
| NumPy        | Numerical operations         |
| NLTK         | Text preprocessing           |
| Scikit-learn | Machine Learning             |
| Matplotlib   | Visualization                |
| Seaborn      | Data visualization           |
| WordCloud    | Word frequency visualization |
| Joblib       | Model serialization          |
| Google Colab | Development environment      |

---

## 🔄 Project Workflow

```text
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
```

---

## 🧹 Text Preprocessing

The raw news articles are cleaned before feature extraction.

### Preprocessing Steps

1. Convert text to lowercase.
2. Remove HTML tags.
3. Remove URLs.
4. Remove punctuation.
5. Remove special characters.
6. Remove extra whitespace.
7. Tokenize the text.
8. Remove English stopwords.
9. Remove very short words.
10. Apply lemmatization.

### Example

**Original:**

```text
Apple Inc. announced a NEW AI processor at
https://example.com today!
```

**After preprocessing:**

```text
apple inc announced new ai processor today
```

### Why preprocessing?

Preprocessing reduces unnecessary variations and noise in the text, allowing the machine learning models to focus on useful words and phrases.

---

## 📊 Exploratory Data Analysis

The project performs several EDA tasks:

### Class Distribution

The number of articles in each category is analyzed and visualized.

### Text Length

The number of words in each article is calculated to understand the distribution of article lengths.

### Most Frequent Words

The most common words are extracted separately for each category.

### Word Clouds

Word clouds are generated for:

* World
* Sports
* Business
* Sci/Tech

These visualizations help identify vocabulary patterns associated with each topic.

---

## 🔢 TF-IDF Feature Extraction

Machine learning algorithms cannot directly process raw text, so the cleaned articles are converted into numerical vectors using **TF-IDF**.

TF-IDF stands for:

> **Term Frequency — Inverse Document Frequency**

It gives higher importance to words that are useful for distinguishing documents while reducing the importance of extremely common words.

### Experiment 1 — Unigrams

```python
TfidfVectorizer(
    ngram_range=(1,1)
)
```

This considers individual words.

Example:

```text
stock
market
company
football
```

### Experiment 2 — Unigrams + Bigrams

```python
TfidfVectorizer(
    ngram_range=(1,2)
)
```

This considers individual words and two-word combinations.

Example:

```text
stock
market
stock market
financial market
```

Bigrams can capture useful phrases that individual words may not represent well.

---

## 🤖 Machine Learning Models

Two machine learning algorithms are compared.

### 1. Logistic Regression

Logistic Regression is used as a linear classifier for multi-class text classification.

It learns relationships between TF-IDF features and the four news categories.

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic algorithm commonly used for text classification.

It estimates the probability of each category based on the words present in the article.

---

## 🧪 Train/Validation Split

The training dataset is divided using:

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=labels
)
```

This creates:

```text
80% → Training
20% → Validation
```

The same split is used for comparing the models.

The separate official AG News `test.csv` is kept untouched and used for final evaluation.

---

## 📈 Model Evaluation

The models are evaluated using:

### Accuracy

Percentage of correctly classified articles.

### Precision

Measures how many articles predicted as a category actually belong to that category.

### Recall

Measures how many articles belonging to a category were correctly identified.

### F1-Score

Harmonic mean of precision and recall.

### Confusion Matrix

Shows which categories are correctly classified and which categories are confused with one another.

---

## 🔬 Experiments

The project compares four configurations:

| Experiment | Features           | Model                   |
| ---------- | ------------------ | ----------------------- |
| 1          | Unigrams           | Logistic Regression     |
| 2          | Unigrams           | Multinomial Naive Bayes |
| 3          | Unigrams + Bigrams | Logistic Regression     |
| 4          | Unigrams + Bigrams | Multinomial Naive Bayes |

The results are compared using:

```text
Accuracy
Precision
Recall
F1 Score
```

The model configuration with the strongest measured validation performance can then be selected for the final classifier.

---

## 🔎 Error Analysis

The project examines:

* Category with the highest recall.
* Category with the lowest recall.
* Most frequently confused categories.
* Business vs World confusion.
* Sports classification behavior.
* Effect of adding bigrams.

### Why can Business and World be confused?

Some articles contain overlapping vocabulary involving:

* Governments
* International trade
* Companies
* Economic policies
* Markets
* Countries
* Financial decisions

Therefore, an article discussing both international politics and economics may contain signals from both categories.

---

## 🧪 Testing New Articles

The trained classifier can be used on completely new news articles.

### Example

```python
news = """
Researchers have developed a new artificial intelligence
system that can analyze medical images and identify signs
of disease.
"""

category, confidence = predict_news(news)

print("Predicted Category:", category)
print(f"Confidence: {confidence:.2f}%")
```

Possible output:

```text
Predicted Category: Sci/Tech
Confidence: XX.XX%
```

---

## 💡 Interactive News Classification

The project also provides an interactive prediction system.

```python
while True:

    news = input("\nEnter a news article (type 'exit' to stop): ")

    if news.lower() == "exit":
        break

    category, confidence = predict_news(news)

    print("\nPredicted Category:", category)
    print(f"Confidence: {confidence:.2f}%")
```

### Example

```text
Enter a news article:

The football team won the championship after scoring
two goals in the final match.

Predicted Category: Sports
Confidence: XX.XX%
```

---

## 📝 Example Test Articles

### 🌍 World

```text
The governments of India and Japan held a bilateral meeting
to discuss trade, regional security and infrastructure cooperation.
```

Expected:

```text
World
```

### ⚽ Sports

```text
The national football team defeated its rivals 3-1 in the
championship final after scoring two goals in the second half.
```

Expected:

```text
Sports
```

### 💼 Business

```text
The company reported a 15 percent increase in quarterly revenue
as demand for its cloud services continued to grow.
```

Expected:

```text
Business
```

### 💻 Sci/Tech

```text
Researchers developed a new artificial intelligence system
that can analyze medical images and identify signs of disease.
```

Expected:

```text
Sci/Tech
```

---

## 📁 Project Structure

```text
AG-News-Topic-Classification/
│
├── train.csv
├── test.csv
│
├── AG_News_Topic_Classification.ipynb
│
├── agnews_model.pkl
├── agnews_tfidf.pkl
├── agnews_labels.pkl
│
└── README.md
```

---

## 💾 Saved Model

The trained components can be saved using Joblib:

```python
joblib.dump(final_model, "agnews_model.pkl")
joblib.dump(final_vectorizer, "agnews_tfidf.pkl")
joblib.dump(label_map, "agnews_labels.pkl")
```

This allows the trained model to be reused without training it again.

---

## ⚠️ Limitations

* The classifier is trained specifically on the four AG News categories.
* Real-world news may contain topics outside these four categories.
* Some articles can naturally belong to multiple topics.
* Business and World articles may contain overlapping vocabulary.
* New terminology may not be represented in the training dataset.
* Confidence values are model probabilities, not guarantees of correctness.
* A machine learning classifier cannot guarantee 100% accuracy on every unseen real-world article.

---

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning.
* More advanced NLP preprocessing.
* Word embeddings.
* Neural network-based text classification.
* BERT-based classification.
* A web interface using Streamlit.
* Django-based news classification application.
* Database storage for classified articles.
* Confidence-based "Uncertain" prediction.
* Deployment as a web API.

---

## 🎓 Assignment Requirements Covered

| Requirement                 | Status |
| --------------------------- | ------ |
| Load dataset using Pandas   | ✅      |
| Display first 5 records     | ✅      |
| Dataset dimensions          | ✅      |
| Missing value analysis      | ✅      |
| Duplicate analysis          | ✅      |
| Articles per category       | ✅      |
| Class balance               | ✅      |
| Combine title + description | ✅      |
| Text preprocessing          | ✅      |
| Lowercase                   | ✅      |
| HTML removal                | ✅      |
| Special character removal   | ✅      |
| Stopword removal            | ✅      |
| Tokenization                | ✅      |
| Lemmatization               | ✅      |
| Average text length         | ✅      |
| Class distribution graph    | ✅      |
| Text-length distribution    | ✅      |
| Frequent words              | ✅      |
| Word clouds                 | ✅      |
| TF-IDF unigrams             | ✅      |
| TF-IDF unigrams + bigrams   | ✅      |
| 80/20 split                 | ✅      |
