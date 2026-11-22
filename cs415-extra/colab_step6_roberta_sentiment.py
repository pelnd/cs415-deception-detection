# Step 6: RoBERTa (TweetEval) sentiment for all 17,289 messages.  Run this in Colab (GPU runtime).
# 1) Upload out/messages.csv to the Colab session   2) paste this file into a cell   3) download roberta.csv
# NOT yet run: the workspace used to write it cannot reach Hugging Face, so the first Colab run is the test.
# !pip -q install transformers
import re, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"   # labels: 0=negative 1=neutral 2=positive
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()


def prep(t):  # preprocessing from the model card
    t = re.sub(r"@\w+", "@user", str(t))
    return re.sub(r"http\S+", "http", t)


df = pd.read_csv("messages.csv", keep_default_na=False)
texts = [prep(t) for t in df.text]
order = np.argsort([len(t) for t in texts])           # length-sorted batches = faster
probs = np.zeros((len(texts), 3), dtype=np.float32)
bs = 64
with torch.no_grad():
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(dev)
        probs[idx] = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

out = df[["split", "game_id", "conv_id", "pos"]].copy()
out["rob_neg"], out["rob_neu"], out["rob_pos"] = probs.T
out["rob_score"] = out.rob_pos - out.rob_neg           # one number in [-1, 1]
out.to_csv("roberta.csv", index=False)
print(out.groupby("split")[["rob_neg", "rob_neu", "rob_pos", "rob_score"]].mean())
# Truncated messages (max_length 256 tokens): how many?
print("truncated:", sum(len(tok(t)["input_ids"]) > 256 for t in texts[:2000]), "of first 2000")
