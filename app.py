from pathlib import Path
import re

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


st.set_page_config(
    page_title="NewsLens AI | AG News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --ink: #f8f7ff;
    --muted: #b9b4ce;
    --border: rgba(255, 255, 255, .14);
    --panel: rgba(25, 16, 43, .78);
    --red: #ff3d68;
    --purple: #9b5cff;
    --pink: #ff70b7;
}

* { box-sizing: border-box; }
.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 0% 0%, rgba(255, 61, 104, .22), transparent 28rem),
        radial-gradient(circle at 100% 10%, rgba(155, 92, 255, .25), transparent 30rem),
        linear-gradient(135deg, #10091d 0%, #180d2a 48%, #0b0716 100%);
    font-family: 'DM Sans', sans-serif;
}
.block-container { max-width: 1280px; padding: 1.25rem 2rem 4rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14091f, #0c0714);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--ink); }

.hero {
    position: relative; overflow: hidden; isolation: isolate;
    min-height: 390px; padding: clamp(2rem, 6vw, 5rem);
    border: 1px solid rgba(255, 112, 183, .3); border-radius: 32px;
    background:
        linear-gradient(115deg, rgba(36, 13, 53, .94), rgba(24, 13, 43, .82)),
        radial-gradient(circle at 85% 25%, rgba(255, 61, 104, .32), transparent 24rem);
    box-shadow: 0 28px 90px rgba(0, 0, 0, .35);
    animation: rise .75s ease both;
}
.hero::before, .hero::after {
    content: ""; position: absolute; z-index: -1; border-radius: 50%;
    filter: blur(1px); opacity: .65; pointer-events: none;
}
.hero::before {
    width: 260px; height: 260px; right: -80px; top: -110px;
    background: radial-gradient(circle, rgba(255, 61, 104, .7), transparent 68%);
    animation: drift 7s ease-in-out infinite;
}
.hero::after {
    width: 190px; height: 190px; right: 24%; bottom: -110px;
    background: radial-gradient(circle, rgba(155, 92, 255, .7), transparent 68%);
    animation: drift 9s ease-in-out infinite reverse;
}
.hero-badge, .eyebrow {
    display: inline-flex; align-items: center; gap: .5rem;
    padding: .55rem .85rem; border-radius: 999px;
    color: #ffd5e4; background: rgba(255, 61, 104, .13);
    border: 1px solid rgba(255, 112, 183, .3);
    font-size: .76rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
}
.hero h1 {
    max-width: 780px; margin: 1.2rem 0 .9rem;
    font-family: 'Manrope', sans-serif; font-size: clamp(2.7rem, 7vw, 5.8rem);
    line-height: .98; letter-spacing: -.06em;
    background: linear-gradient(100deg, #fff 15%, #ffb1cb 55%, #b998ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { max-width: 680px; color: #d2cce2; font-size: clamp(1rem, 2vw, 1.18rem); line-height: 1.7; }
.hero-art {
    position: absolute; right: 8%; top: 23%; display: grid; gap: .7rem;
    width: 235px; padding: 1rem; transform: rotate(5deg);
    border: 1px solid rgba(255,255,255,.18); border-radius: 22px;
    background: rgba(14, 8, 27, .52); box-shadow: 0 25px 60px rgba(0,0,0,.25);
    animation: float 5s ease-in-out infinite;
}
.hero-art span { height: 9px; border-radius: 999px; background: linear-gradient(90deg, var(--red), var(--purple)); }
.hero-art span:nth-child(2) { width: 76%; background: linear-gradient(90deg, var(--purple), var(--pink)); }
.hero-art span:nth-child(3) { width: 52%; background: #ffc0d5; }
.section-title { margin: 2.5rem 0 1rem; font: 800 1.6rem 'Manrope', sans-serif; }
.section-copy { color: var(--muted); line-height: 1.65; }
.glass-card, .category-card, .result {
    border: 1px solid var(--border); border-radius: 22px;
    background: var(--panel); box-shadow: 0 18px 55px rgba(0,0,0,.18);
    backdrop-filter: blur(18px);
}
.glass-card { padding: 1.35rem; }
.category-card {
    min-height: 155px; padding: 1.35rem; transition: transform .25s, border-color .25s, box-shadow .25s;
}
.category-card:hover { transform: translateY(-6px); border-color: rgba(255,112,183,.55); box-shadow: 0 18px 35px rgba(255,61,104,.13); }
.category-icon { font-size: 2rem; }
.category-name { margin-top: .75rem; font-weight: 800; font-size: 1.05rem; }
.category-desc { margin-top: .35rem; color: var(--muted); font-size: .84rem; line-height: 1.45; }
.result {
    padding: 2rem; text-align: center;
    background: radial-gradient(circle at 50% 0%, rgba(255,61,104,.25), transparent 55%), var(--panel);
    animation: rise .45s ease both;
}
.result-label { color: #d5cde5; font-size: .75rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
.result-category {
    margin: .4rem 0; font: 800 clamp(2.3rem, 6vw, 4rem) 'Manrope', sans-serif;
    background: linear-gradient(90deg, #ff82a7, #b998ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.result-confidence { color: #d8d1e8; }
.status {
    padding: .7rem .85rem; border-radius: 12px; color: #ffd6e2;
    background: rgba(255,61,104,.1); border: 1px solid rgba(255,112,183,.25); font-size: .82rem;
}
.stTextArea textarea {
    color: var(--ink) !important; background: rgba(11, 6, 21, .68) !important;
    border: 1px solid rgba(255,255,255,.16) !important; border-radius: 16px !important;
    font-size: 1rem !important; line-height: 1.65 !important;
}
.stTextArea textarea:focus { border-color: var(--pink) !important; box-shadow: 0 0 0 2px rgba(255,112,183,.16) !important; }
div.stButton > button {
    min-height: 2.8rem; border-radius: 13px; color: var(--ink);
    border: 1px solid rgba(255,255,255,.17); background: rgba(45, 25, 70, .8);
    font-weight: 700; transition: transform .2s, box-shadow .2s, border-color .2s;
}
div.stButton > button:hover { transform: translateY(-2px); border-color: var(--pink); box-shadow: 0 10px 25px rgba(255,61,104,.18); }
div.stButton > button[kind="primary"] { border: 0; background: linear-gradient(100deg, var(--red), var(--purple)); box-shadow: 0 12px 28px rgba(255,61,104,.22); }
[data-testid="stMetric"] { border: 1px solid var(--border); border-radius: 16px; background: rgba(25,16,43,.7); }
[data-testid="stMetricValue"] { color: #fff !important; }
div[data-baseweb="select"] > div { border-radius: 12px !important; background: rgba(11,6,21,.75) !important; border-color: var(--border) !important; }
.footer { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border); text-align: center; color: #9289aa; font-size: .82rem; }
@keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes float { 0%,100% { transform: rotate(5deg) translateY(0); } 50% { transform: rotate(2deg) translateY(-12px); } }
@keyframes drift { 0%,100% { transform: translate(0); } 50% { transform: translate(-22px, 18px); } }
@media (max-width: 760px) {
    .block-container { padding: .8rem 1rem 3rem; }
    .hero { min-height: 420px; padding: 2rem 1.35rem; }
    .hero-art { opacity: .3; right: -2rem; top: 48%; transform: scale(.8) rotate(5deg); }
    .section-title { margin-top: 2rem; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def setup_nltk():
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            continue


setup_nltk()
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = set()
LEMMATIZER = WordNetLemmatizer()

BASE_DIR = Path(__file__).resolve().parent
TFIDF_PATH = BASE_DIR / "agnews_tfidf.pkl"
LABELS_PATH = BASE_DIR / "agnews_labels.pkl"
LR_PATHS = [BASE_DIR / "agnews_lr_model.pkl", BASE_DIR / "logistic_regression_model.pkl", BASE_DIR / "agnews_model.pkl"]
NB_PATHS = [BASE_DIR / "agnews_nb_model.pkl", BASE_DIR / "naive_bayes_model.pkl"]


def first_existing(paths):
    return next((path for path in paths if path.exists()), None)


LR_PATH = first_existing(LR_PATHS)
NB_PATH = first_existing(NB_PATHS)


@st.cache_resource
def load_artifacts(tfidf_path, labels_path, lr_path, nb_path):
    vectorizer = joblib.load(tfidf_path)
    labels = joblib.load(labels_path)
    lr_model = joblib.load(lr_path) if lr_path else None
    nb_model = joblib.load(nb_path) if nb_path else None
    return vectorizer, labels, lr_model, nb_model


if not TFIDF_PATH.exists() or not LABELS_PATH.exists() or LR_PATH is None:
    st.error("Required model files were not found beside app.py.")
    st.info("Add agnews_tfidf.pkl, agnews_labels.pkl, and agnews_model.pkl (or a Logistic Regression model) to continue.")
    st.stop()

try:
    vectorizer, label_map, lr_model, nb_model = load_artifacts(TFIDF_PATH, LABELS_PATH, LR_PATH, NB_PATH)
except Exception as exc:
    st.error(f"Could not load the saved ML files: {exc}")
    st.stop()


def category_name(label):
    if isinstance(label_map, dict):
        for key in (label, str(label)):
            if key in label_map:
                return str(label_map[key])
        try:
            if int(label) in label_map:
                return str(label_map[int(label)])
        except (TypeError, ValueError):
            pass
    fallback = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech", 4: "Sci/Tech"}
    try:
        return fallback[int(label)]
    except (TypeError, ValueError, KeyError):
        return str(label)


def preprocess(text):
    cleaned = re.sub(r"<.*?>|http\S+|www\S+", " ", str(text).lower())
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    words = re.sub(r"\s+", " ", cleaned).strip().split()
    return " ".join(LEMMATIZER.lemmatize(word) for word in words if word not in STOP_WORDS and len(word) > 2)


def predict(article, selected_model):
    cleaned = preprocess(article)
    if not cleaned:
        raise ValueError("The article became empty after preprocessing. Please enter a longer news article.")
    features = vectorizer.transform([cleaned])
    label = selected_model.predict(features)[0]
    probabilities = selected_model.predict_proba(features)[0] if hasattr(selected_model, "predict_proba") else None
    return category_name(label), probabilities, cleaned


EXAMPLES = {
    "World": "The leaders of several countries met at an international summit to discuss a new agreement on global cooperation and security.",
    "Sports": "The football team won the championship final after scoring two goals in the second half of the match.",
    "Business": "The company reported strong quarterly revenue as sales increased across international markets and investor confidence improved.",
    "Sci/Tech": "Researchers developed a new artificial intelligence system that can analyze medical images and detect diseases more accurately.",
}
CATEGORIES = [
    ("🌍", "World", "International events and global affairs."),
    ("⚽", "Sports", "Matches, athletes and competitions."),
    ("💼", "Business", "Companies, finance and markets."),
    ("🔬", "Sci/Tech", "Science, AI and technology."),
]

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "article" not in st.session_state:
    st.session_state.article = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None


with st.sidebar:
    st.markdown("## 📰 NewsLens AI")
    st.markdown('<div class="status">● ML model loaded successfully</div>', unsafe_allow_html=True)
    st.markdown("### Navigate")
    page = st.radio("Page", ["Home", "Classify article", "About"], index=["Home", "Classify article", "About"].index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = page
    st.divider()
    st.markdown("### Classifier")
    available_models = ["Logistic Regression"] + (["Multinomial Naive Bayes"] if nb_model is not None else [])
    selected_model_name = st.selectbox("Choose classifier", available_models)
    selected_model = lr_model if selected_model_name == "Logistic Regression" else nb_model
    st.caption(f"Pipeline: Article → TF-IDF → {selected_model_name} → Category")
    if nb_model is None:
        st.info("Naive Bayes is optional and was not found. Logistic Regression is active.")


def render_hero():
    st.markdown(
        """
<section class="hero">
  <div class="hero-badge">✦ NLP · Machine learning · instant insights</div>
  <h1>Turn headlines into clear signals.</h1>
  <p>NewsLens AI classifies news into World, Sports, Business, or Sci/Tech using your trained TF-IDF pipeline and machine learning models.</p>
  <div class="hero-art" aria-hidden="true"><span></span><span></span><span></span></div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_categories():
    st.markdown('<div class="section-title">Explore the news universe</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (icon, name, desc) in zip(cols, CATEGORIES):
        with col:
            st.markdown(f'<div class="category-card"><div class="category-icon">{icon}</div><div class="category-name">{name}</div><div class="category-desc">{desc}</div></div>', unsafe_allow_html=True)


def render_home():
    render_hero()
    render_categories()
    st.markdown('<div class="section-title">Ready when you are</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="glass-card"><span class="eyebrow">How it works</span><h3>One article. Four possibilities.</h3><p class="section-copy">Paste a headline or full story and let the local model identify its strongest topic. No external API calls, no data leaving your machine.</p></div>', unsafe_allow_html=True)
    with right:
        if st.button("Start classifying →", type="primary", width="stretch"):
            st.session_state.page = "Classify article"
            st.rerun()


def render_classifier():
    render_hero()
    st.markdown('<div class="section-title">Classify an article</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><p class="section-copy">Enter a headline, paragraph, or complete article. The same preprocessing and TF-IDF representation used during training will be applied before prediction.</p></div>', unsafe_allow_html=True)
    article = st.text_area("News article", value=st.session_state.article, height=220, placeholder="Example: The technology company announced a new artificial intelligence processor...", label_visibility="collapsed")
    st.session_state.article = article
    st.markdown("#### Try a sample")
    sample_cols = st.columns(4)
    for col, (icon, name, _) in zip(sample_cols, CATEGORIES):
        with col:
            if st.button(f"{icon} {name}", key=f"example_{name}", width="stretch"):
                st.session_state.article = EXAMPLES[name]
                st.session_state.prediction = None
                st.rerun()
    action_col, clear_col = st.columns([3, 1])
    with action_col:
        predict_clicked = st.button(f"🚀 Predict with {selected_model_name}", type="primary", width="stretch")
    with clear_col:
        clear_clicked = st.button("Clear", width="stretch")
    if clear_clicked:
        st.session_state.article = ""
        st.session_state.prediction = None
        st.rerun()
    if predict_clicked:
        if not article.strip():
            st.warning("Please enter a news article before predicting.")
        else:
            try:
                with st.spinner("Analyzing article..."):
                    category, probabilities, cleaned = predict(article, selected_model)
                st.session_state.prediction = {"category": category, "probabilities": probabilities, "cleaned": cleaned, "model": selected_model_name}
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    render_result()


def render_result():
    result = st.session_state.prediction
    if result is None:
        return
    st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result"><div class="result-label">Predicted news category</div><div class="result-category">{result["category"]}</div><div class="result-confidence">Classifier: <strong>{result["model"]}</strong></div></div>', unsafe_allow_html=True)
    probabilities = result["probabilities"]
    if probabilities is not None:
        classes = list(selected_model.classes_)
        probability_df = pd.DataFrame({"Category": [category_name(item) for item in classes], "Probability (%)": probabilities * 100}).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
        confidence = float(probability_df.iloc[0]["Probability (%)"])
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Prediction", result["category"])
        metric2.metric("Confidence", f"{confidence:.2f}%")
        metric3.metric("Classifier", result["model"])
        st.markdown('<div class="section-title">Probability by category</div>', unsafe_allow_html=True)
        st.bar_chart(probability_df.set_index("Category")["Probability (%)"], height=320)
        st.dataframe(probability_df.assign(**{"Probability (%)": probability_df["Probability (%)"].round(2)}), width="stretch", hide_index=True)
    with st.expander("🔍 View processed text"):
        st.write(result["cleaned"])
    with st.expander("ℹ️ How this prediction was produced"):
        st.markdown(f"**Selected model:** {result['model']}\n\n1. Lowercase and clean HTML, URLs, and special characters.\n2. Remove stopwords and lemmatize words.\n3. Transform the text with the saved TF-IDF vectorizer.\n4. Predict one of the four AG News categories.\n\nThe application does not call an external AI API.")


def render_about():
    render_hero()
    st.markdown('<div class="section-title">About NewsLens AI</div>', unsafe_allow_html=True)
    info_cols = st.columns(3)
    cards = [
        ("Dataset", "AG News<br>4 categories<br>World · Sports · Business · Sci/Tech"),
        ("Feature extraction", "TF-IDF<br>Unigrams / bigrams<br>Local preprocessing"),
        ("Models", "Logistic Regression<br>Multinomial Naive Bayes<br>Confidence-aware output"),
    ]
    for col, (title, body) in zip(info_cols, cards):
        with col:
            st.markdown(f'<div class="glass-card"><span class="eyebrow">{title}</span><p class="section-copy">{body}</p></div>', unsafe_allow_html=True)


if st.session_state.page == "Home":
    render_home()
elif st.session_state.page == "Classify article":
    render_classifier()
else:
    render_about()

st.markdown('<div class="footer"><strong>NewsLens AI</strong> · AG News topic classification · TF-IDF + local machine learning · No external API</div>', unsafe_allow_html=True)
