# Spam Detection using Machine Learning

## Overview
This project implements a spam vs ham text classification system using traditional machine learning techniques. It serves as a foundational baseline before extending to deep learning and transformer-based models.

## Problem Statement
Spam messages cause security risks and poor user experience. The objective is to accurately classify messages as spam or ham while minimizing false positives.

## Approach
1. Text preprocessing
2. Feature extraction using TF-IDF
3. Model training using:
   - Multinomial Naive Bayes
   - Linear Support Vector Machine (SVM)
4. Evaluation using precision, recall, F1-score, and confusion matrix

## Feature Engineering
- TF-IDF captures word importance based on frequency and rarity
- Stop-word removal to reduce noise

## Models Used
### Naive Bayes
A probabilistic classifier that models word likelihoods for spam detection.

### Linear SVM
A margin-based classifier that performs well in high-dimensional sparse text spaces.

## Evaluation Metrics
- Precision (important to reduce false positives)
- Recall
- F1-score
- Confusion Matrix

## Results
Linear SVM achieved better performance compared to Naive Bayes, especially in reducing false positives.

## Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib
- seaborn

## Future Work
- Transformer-based models (BERT)
- Ensemble learning
- Adaptive spam filtering
