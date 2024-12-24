# Machine Learning Internship Projects

This repository contains two projects completed as part of the AspireNex Machine Learning Internship:

1. **Movie Genre Classification**
2. **Credit Card Fraud Detection**

## Movie Genre Classification

The goal of this project is to predict the genre of a movie based on its plot summary. The model uses TF-IDF for feature extraction and various classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machines to make predictions.

### Key Steps:
- Data Collection and Preprocessing
- Feature Extraction using TF-IDF
- Model Training and Evaluation
- Hyperparameter Tuning
- Model Deployment using Flask

### Files:
- `train_data.txt`: Training data with movie details.
- `test_data.txt`: Test data for evaluation.
- `movie_genre_model.pkl`: Trained model.
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer.
- `app.ipynb`: Interactive prediction function for using model.

## Credit Card Fraud Detection

The objective of this project is to detect fraudulent credit card transactions using machine learning algorithms. The dataset includes various features related to transactions and the task is to classify them as fraudulent or non-fraudulent.

### Key Steps:
- Data Preprocessing
- Feature Engineering
- Model Training and Evaluation using various classifiers
- Handling Class Imbalance


### Files:
- `creditcard.csv`: Dataset containing transaction details.
- `best_rf.joblib`: Trained model with random forest best accuracy.
- `fraud_detection.ipynb`: Jupyter notebook with code and explanations.
- `decision_tree_model.pkl`: Trained model using decision tree.
- `logictic_regression_model.pkl`: Trained model using logistic regression.
- `random_forest_model.pkl`: Trained model using random forest.
