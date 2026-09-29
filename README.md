# 📊 Sentiment Analysis on Product Reviews

A complete NLP pipeline that classifies product reviews as **Positive**, **Negative**, or **Neutral** — built with classic text preprocessing, TF-IDF feature extraction, and a comparison of three machine learning models. Includes an interactive **Streamlit web app** for live predictions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Project Overview

This project builds a full sentiment classification pipeline for e-commerce-style product reviews:

1. **Data** — 3,600 labeled product reviews across 30 product categories (with realistic ambiguity and ~6% label noise to simulate real annotation conditions).
2. **Preprocessing (NLP)** — lowercasing, punctuation/number stripping, tokenization, and stopword removal.
3. **Feature Extraction** — TF-IDF vectorization (unigrams + bigrams, 5,000 features).
4. **Modeling** — trains and compares three classifiers:
   - Multinomial **Naive Bayes**
   - **Linear SVM** (Support Vector Machine)
   - **Neural Network** (Multi-Layer Perceptron)
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, and confusion matrices for each model.
6. **Visualization** — sentiment distribution chart, model comparison bar chart, confusion matrix heatmaps.
7. **Deployment** — an interactive Streamlit app for real-time sentiment prediction.

---

## 📁 Project Structure

```
sentiment_project/
├── data/
│   └── product_reviews.csv          # 3,600 labeled reviews
├── models/                          # saved after running train_models.py
│   ├── tfidf_vectorizer.joblib
│   ├── naive_bayes.joblib
│   ├── svm_linear.joblib
│   ├── neural_network_mlp.joblib
│   ├── best_model.json
│   ├── results_summary.json
│   └── model_comparison.csv
├── visuals/                         # generated charts
│   ├── sentiment_distribution.png
│   ├── model_performance_comparison.png
│   └── confusion_matrices.png
├── generate_dataset.py              # synthetic dataset generator
├── train_models.py                  # full training/evaluation pipeline
├── app.py                           # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
A ready-to-use dataset is already included at `data/product_reviews.csv`. To regenerate it:
```bash
python generate_dataset.py
```

### 3. Train and evaluate the models
```bash
python train_models.py
```
This will:
- Clean and vectorize the text
- Train Naive Bayes, SVM, and a Neural Network
- Print accuracy/precision/recall/F1 for each model
- Save all visuals to `visuals/`
- Save trained models + vectorizer to `models/`

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naive Bayes | ~0.94 | ~0.94 | ~0.94 | ~0.94 |
| Linear SVM | ~0.94 | ~0.94 | ~0.94 | ~0.94 |
| Neural Network (MLP) | ~0.93 | ~0.93 | ~0.93 | ~0.93 |

*(Exact numbers may vary slightly between runs/re-generated datasets — see `models/results_summary.json` after training.)*

### Visualizations
- **Sentiment Distribution** — `visuals/sentiment_distribution.png`
- **Model Performance Comparison** — `visuals/model_performance_comparison.png`
- **Confusion Matrices** — `visuals/confusion_matrices.png`

---

## 🖥️ Streamlit App Features

- **Predict Sentiment** — paste any review text, pick a model, get an instant prediction with confidence scores
- **Dataset Overview** — explore the sentiment distribution and sample reviews by class
- **Model Performance** — view accuracy/precision/recall/F1 table, comparison chart, and confusion matrices

---

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** — TF-IDF, Naive Bayes, SVM, MLPClassifier, metrics
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualization
- **Streamlit** — interactive web app
- **joblib** — model persistence

---

## 📌 Notes on the Dataset

The dataset is synthetically generated (`generate_dataset.py`) to keep the project fully reproducible and free of any licensing/privacy concerns, while still mimicking realistic review patterns: varied product categories, varied sentence structure, shared/ambiguous filler sentences across classes, and ~6% intentional label noise. You can swap in your own real review dataset (e.g., Amazon or Yelp reviews) by replacing `data/product_reviews.csv` — just keep the `review_text` and `sentiment` columns.

---

## 📄 License

This project is released under the MIT License — feel free to use, modify, and build on it.

---

## 🙋 Author

Built as an academic / portfolio NLP project demonstrating end-to-end sentiment analysis: text preprocessing, feature engineering, model comparison, and deployment.
