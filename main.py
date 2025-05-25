import streamlit as st
from sentence_transformers import SentenceTransformer
import joblib, faiss, numpy as np
from openai import OpenAI
from utils import build_prompt
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ---------- bootstrap -------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def bootstrap():
    embed  = SentenceTransformer("all-MiniLM-L6-v2")
    clf    = joblib.load("strategy_clf.joblib")
    idx    = faiss.read_index("esconv_strategy.idx")
    texts  = np.load("esconv_texts.npy", allow_pickle=True)
    labels = np.load("esconv_labels.npy", allow_pickle=True)
    return embed, clf, idx, texts, labels

embed, clf, idx, texts, labels = bootstrap()
openai = OpenAI(api_key=OPENAI_API_KEY)
# ---------------------------------------------------------------------

st.title("Counselor Copilot")
summary = st.text_area("Describe the client’s current challenge:", height=180)

if st.button("Get guidance") and summary.strip():
    vec = embed.encode([summary]).astype("float32")
    faiss.normalize_L2(vec)
    strategy = clf.predict(vec)[0]

    # retrieve top-k with same strategy
    mask = np.where(labels == strategy)[0]
    D, I = idx.search(vec, 10)
    hits = [(texts[i], labels[i]) for i in I[0] if i in mask][:3]

    messages = build_prompt(summary, strategy, hits)
    gpt = openai.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.5
    )

    st.markdown(f"**Predicted strategy:** `{strategy}`")
    st.markdown("### Suggested approach")
    st.write(gpt.choices[0].message.content)

    with st.expander("Similar past cases"):
        for ex, _ in hits:
            st.markdown(f"- {ex}")
