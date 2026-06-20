from datasets import load_dataset
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

# =========================
# 1. LOAD DATASET
# =========================
dataset = load_dataset("stanfordnlp/imdb")
train_df = pd.DataFrame(dataset["train"])

print("\nShape:", train_df.shape)
print("\nLabel distribution:")
print(train_df["label"].value_counts())

# =========================
# 2. FEATURES (EDA)
# =========================
train_df["length"] = train_df["text"].apply(len)
train_df["word_count"] = train_df["text"].apply(lambda x: len(x.split()))

print("\nStats:")
print(train_df[["length", "word_count"]].describe())

# =========================
# 3. CLEANING TEXT
# =========================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train_df["clean_text"] = train_df["text"].apply(clean_text)

# Après le nettoyage
X = train_df["clean_text"]
y = train_df["label"]

# =========================
# 4. VOCABULARY
# =========================
all_words = " ".join(train_df["clean_text"]).split()
counter = Counter(all_words)

print("\nTop 20 words:")
print(counter.most_common(20))

# =========================
# 5. PLOTS (SAVE ONLY)
# =========================

# Sentiment distribution
sns.countplot(x="label", data=train_df)
plt.title("Sentiment Distribution")
plt.savefig("sentiment_distribution.png")
plt.close()

# Word length distribution
plt.hist(train_df["word_count"], bins=50)
plt.title("Word Count Distribution")
plt.savefig("word_distribution.png")
plt.close()

# =========================
# 6. FINAL CHECK
# =========================
print("\nClean text sample:")
print(train_df["clean_text"].head())

# Sauvegarde du dataset nettoyé
train_df.to_csv("imdb_cleaned.csv", index=False)

print("Dataset sauvegardé : imdb_cleaned.csv")