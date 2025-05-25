import json, joblib, faiss, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score


# ─────────────────────────────────────────────────────────────
# 1) Load and unpack ESConv
# ─────────────────────────────────────────────────────────────
raw = load_dataset("thu-coai/esconv", split="train")   # 910 rows

pairs = {"text": [], "label": []}

for row in raw["text"]:                     # each row is a JSON string
    data = json.loads(row)                  # dict with keys: dialog, strategy...
    dialog = data["dialog"]                 # list[ {text, speaker, strategy?} ]

    for i, turn in enumerate(dialog[:-1]):  # ignore final turn
        if turn["speaker"] == "usr" and dialog[i+1]["speaker"] == "sys":
            usr_text = turn["text"]
            strategy = dialog[i+1].get("strategy") or "Others"
            pairs["text"].append(usr_text)
            pairs["label"].append(strategy)

print(f"⇒ Collected {len(pairs['text'])} patient–strategy pairs")

# ─────────────────────────────────────────────────────────────
# 2) Encode with MiniLM
# ─────────────────────────────────────────────────────────────
embed = SentenceTransformer("all-MiniLM-L6-v2")
vecs  = embed.encode(pairs["text"], batch_size=64, show_progress_bar=True)

# ─────────────────────────────────────────────────────────────
# 3) Train classifier
# ─────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    vecs, pairs["label"], test_size=0.2,
    stratify=pairs["label"], random_state=42
)

clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")
print("3-fold macro-F1:",
      cross_val_score(clf, X_tr, y_tr, cv=3, scoring="f1_macro").mean())

clf.fit(X_tr, y_tr)
print("Hold-out accuracy:", clf.score(X_te, y_te))

joblib.dump(clf, "strategy_clf.joblib")

# ─────────────────────────────────────────────────────────────
# 4) Build FAISS index for retrieval
# ─────────────────────────────────────────────────────────────
vecs_f32 = vecs.astype("float32")
faiss.normalize_L2(vecs_f32)

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs_f32)
faiss.write_index(index, "esconv_strategy.idx")

np.save("esconv_texts.npy", np.array(pairs["text"]))
np.save("esconv_labels.npy", np.array(pairs["label"]))
