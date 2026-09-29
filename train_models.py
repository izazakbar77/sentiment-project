"""
train_models.py
----------------
Full NLP pipeline for sentiment classification on product reviews.

Steps:
1. Load dataset
2. Text cleaning + tokenization + stopword removal
3. TF-IDF vectorization
4. Train 3 models: Multinomial Naive Bayes, Linear SVM, Neural Network (MLP)
5. Evaluate & compare (accuracy, precision, recall, F1)
6. Visualize sentiment distribution + model performance + confusion matrices
7. Save best model and vectorizer for the Streamlit app
"""

import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Text preprocessing: cleaning, tokenization, stopword removal
# ----------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_and_tokenize(text: str):
    """Lowercase, strip punctuation/numbers, tokenize, remove stopwords."""
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    """Returns a cleaned, stopword-free string ready for TfidfVectorizer."""
    return " ".join(clean_and_tokenize(text))


def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv("data/product_reviews.csv")
    print(f"Loaded {len(df)} reviews")

    df["clean_text"] = df["review_text"].apply(preprocess_for_tfidf)

    # ------------------------------------------------------------------
    # Sentiment distribution plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4.5))
    order = ["positive", "neutral", "negative"]
    palette = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}
    ax = sns.countplot(data=df, x="sentiment", order=order,
                        palette=[palette[s] for s in order])
    ax.set_title("Sentiment Distribution in Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig("visuals/sentiment_distribution.png", dpi=150)
    plt.close()
    print("Saved visuals/sentiment_distribution.png")

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"]
    )

    # ------------------------------------------------------------------
    # TF-IDF vectorization
    # ------------------------------------------------------------------
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"TF-IDF matrix shape: train={X_train.shape}, test={X_test.shape}")

    # ------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM (Linear)": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=300,
            random_state=RANDOM_STATE, early_stopping=False
        ),
    }

    results = {}
    trained_models = {}
    confusions = {}

    labels_order = ["positive", "neutral", "negative"]

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )

        results[name] = {
            "accuracy": acc, "precision": precision,
            "recall": recall, "f1_score": f1
        }
        trained_models[name] = model
        confusions[name] = confusion_matrix(y_test, preds, labels=labels_order)

        print(f"{name} -> Accuracy: {acc:.4f} | Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, preds, zero_division=0))

    # ------------------------------------------------------------------
    # Model performance comparison plot
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results).T
    results_df.to_csv("models/model_comparison.csv")

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(results_df))
    width = 0.2

    plt.figure(figsize=(9, 5.5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, results_df[metric], width, label=metric.replace("_", " ").title())
    plt.xticks(x + width * 1.5, results_df.index, rotation=10)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("visuals/model_performance_comparison.png", dpi=150)
    plt.close()
    print("Saved visuals/model_performance_comparison.png")

    # ------------------------------------------------------------------
    # Confusion matrices (one figure, 3 subplots)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, cm) in zip(axes, confusions.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels_order, yticklabels=labels_order, ax=ax, cbar=False)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("visuals/confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved visuals/confusion_matrices.png")

    # ------------------------------------------------------------------
    # Save best model + vectorizer + all models for the Streamlit app
    # ------------------------------------------------------------------
    best_model_name = results_df["f1_score"].astype(float).idxmax()
    print(f"\nBest model: {best_model_name}")

    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"models/{safe_name}.joblib")

    with open("models/best_model.json", "w") as f:
        json.dump({"best_model": best_model_name}, f)

    with open("models/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll models, vectorizer, and visuals saved successfully.")


if __name__ == "__main__":
    main()
