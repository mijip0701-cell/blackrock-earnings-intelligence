import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BlackRock Earnings Intelligence", layout="wide")
st.title("BlackRock Earnings Intelligence")
st.caption("Sentiment (fine-tuned FinBERT) + Summary + Topic Focus")

# Your fine-tuned model on Hugging Face Hub
FINETUNED_REPO = "mparkai/FinalProjectMijiPark"

@st.cache_resource
def load_pipes():
    # Pipeline 1 (fine-tuned): sentiment
    sentiment = pipeline(
        "text-classification",
        model=FINETUNED_REPO,
        tokenizer=FINETUNED_REPO
    )

    # Pipeline 2: summarization
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn"
    )

    # Pipeline 3: zero-shot topic classification
    topic_clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        tokenizer="facebook/bart-large-mnli"
    )

    return sentiment, summarizer, topic_clf

def shorten(text: str, max_chars: int = 3500) -> str:
    return " ".join(text.split())[:max_chars]

sentiment, summarizer, topic_clf = load_pipes()

with st.sidebar:
    st.header("Input")
    text = st.text_area("Paste earnings call / financial text", height=220)

    st.header("Topics")
    topics = st.multiselect(
        "Choose topics",
        ["Pricing", "Demand", "Margins", "Capex", "Supply Chain", "FX/Macro", "Regulatory", "Competition", "Inflation"],
        default=["Pricing", "Demand", "Margins", "FX/Macro", "Inflation"]
    )

    run = st.button("Run analysis", type="primary", use_container_width=True)

if not run:
    st.info("Paste text and click **Run analysis**.")
    st.stop()

if not text.strip():
    st.error("Please paste some text.")
    st.stop()

text_short = shorten(text)

c1, c2 = st.columns(2)

with c1:
    st.subheader("1) Sentiment (Fine-tuned FinBERT)")
    st.write(sentiment(text_short))

with c2:
    st.subheader("2) Topic Focus (Zero-shot)")
    if topics:
        out = topic_clf(text_short, topics, multi_label=True)
        for lab, score in list(zip(out["labels"], out["scores"]))[:7]:
            st.write(f"**{lab}**: {score:.3f}")
    else:
        st.warning("Select at least one topic.")

st.subheader("3) Executive Summary (BART)")
try:
    s = summarizer(text_short, max_length=140, min_length=50, do_sample=False)[0]["summary_text"]
    st.write(s)
except Exception as e:
    st.error("Summarization failed. Try shorter text.")
    st.write(str(e))
