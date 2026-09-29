"""
app.py
-------
Streamlit UI for the Product Review Sentiment Classifier.

Tabs:
1. Predict Sentiment  - type/paste a review, pick a model, see prediction + confidence
2. Dataset Overview   - sentiment distribution, sample data
3. Model Performance  - accuracy/precision/recall/F1 comparison + confusion matrices

Run with:  streamlit run app.py
"""

import re
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Product Review Sentiment Classifier", page_icon="📊", layout="wide")

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_and_tokenize(text: str):
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    return " ".join(clean_and_tokenize(text))


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    model_files = {
        "Naive Bayes": "models/naive_bayes.joblib",
        "SVM (Linear)": "models/svm_linear.joblib",
        "Neural Network (MLP)": "models/neural_network_mlp.joblib",
    }
    models = {name: joblib.load(path) for name, path in model_files.items()}
    with open("models/best_model.json") as f:
        best_model_name = json.load(f)["best_model"]
    with open("models/results_summary.json") as f:
        results = json.load(f)
    return vectorizer, models, best_model_name, results


@st.cache_data
def load_dataset():
    return pd.read_csv("data/product_reviews.csv")


vectorizer, models, best_model_name, results = load_artifacts()
df = load_dataset()

SENTIMENT_EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
SENTIMENT_COLOR = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f1c40f"}

st.title("📊 Product Review Sentiment Classifier")
st.caption("NLP pipeline: tokenization → stopword removal → TF-IDF → "
           "Naive Bayes / SVM / Neural Network comparison")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Sentiment", "📁 Dataset Overview", "📈 Model Performance"])

# ----------------------------------------------------------------------
# TAB 1: Predict
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Try it yourself")
    col_a, col_b = st.columns([2, 1])

    with col_b:
        model_choice = st.selectbox(
            "Choose model",
            list(models.keys()),
            index=list(models.keys()).index(best_model_name) if best_model_name in models else 0,
        )
        st.info(f"⭐ Best performing model: **{best_model_name}**")

    with col_a:
        review_text = st.text_area(
            "Paste a product review",
            placeholder="e.g. This blender stopped working after two days, very disappointed.",
            height=140,
        )
        predict_clicked = st.button("Analyze Sentiment", type="primary")

    if predict_clicked:
        if not review_text.strip():
            st.warning("Please enter some review text first.")
        else:
            clean = preprocess_for_tfidf(review_text)
            X = vectorizer.transform([clean])
            model = models[model_choice]
            pred = model.predict(X)[0]

            emoji = SENTIMENT_EMOJI.get(pred, "")
            color = SENTIMENT_COLOR.get(pred, "#333")
            st.markdown(
                f"<h2 style='color:{color};'>{emoji} Predicted Sentiment: {pred.upper()}</h2>",
                unsafe_allow_html=True,
            )

            # Show class probabilities / confidence if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                proba_df = pd.DataFrame({"sentiment": model.classes_, "confidence": proba})
                proba_df = proba_df.sort_values("confidence", ascending=False)
                st.bar_chart(proba_df.set_index("sentiment"))
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)[0]
                score_df = pd.DataFrame({"sentiment": model.classes_, "decision score": scores})
                score_df = score_df.sort_values("decision score", ascending=False)
                st.bar_chart(score_df.set_index("sentiment"))

            with st.expander("See cleaned/tokenized text used by the model"):
                st.code(clean if clean else "(empty after preprocessing)")

# ----------------------------------------------------------------------
# TAB 2: Dataset overview
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Unique Product Categories", df["product_category"].nunique())
    col3.metric("Classes", ", ".join(sorted(df["sentiment"].unique())))

    dist = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dist.index, dist.values, color=[SENTIMENT_COLOR[s] for s in dist.index])
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Sentiment Distribution")
    for i, v in enumerate(dist.values):
        ax.text(i, v + 5, str(v), ha="center")
    st.pyplot(fig)

    st.markdown("#### Sample Reviews")
    sentiment_filter = st.selectbox("Filter by sentiment", ["all"] + sorted(df["sentiment"].unique()))
    show_df = df if sentiment_filter == "all" else df[df["sentiment"] == sentiment_filter]
    st.dataframe(show_df[["review_text", "product_category", "sentiment"]].sample(
        min(10, len(show_df)), random_state=1
    ), use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3: Model performance
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance Comparison")
    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    results_df[["accuracy", "precision", "recall", "f1_score"]].plot(kind="bar", ax=ax2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title("Model Comparison")
    plt.xticks(rotation=10)
    st.pyplot(fig2)

    st.markdown("#### Confusion Matrices")
    st.image("visuals/confusion_matrices.png", use_container_width=True)

    st.success(f"🏆 Best model based on weighted F1-score: **{best_model_name}**")

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit | TF-IDF based NLP sentiment classifier")
