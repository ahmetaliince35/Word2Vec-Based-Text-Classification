

# -----------------------------
# 1. Gerekli Kütüphaneler
# -----------------------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab') # This line is added to fix the LookupError

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

# -----------------------------
# 2. Veri Yükleme
# -----------------------------
df = pd.read_csv("Yelp Restaurant Reviews.csv")
print(df.columns)
df = df[['Review Text', 'Rating']]               # Önemli sütunlar
df.dropna(inplace=True)

print(df.head())

# -----------------------------
# 3. Veri Temizleme Fonksiyonları
# -----------------------------
negations={"not","never","n't","no"}
stop_words = set(stopwords.words("english"))
stop_words= stop_words - negations

def preprocess(text):
    text = text.lower()                                                   # küçük harf
    text = re.sub(r"[^a-z\s]", "", text)                                  # noktalama temizleme
    tokens = word_tokenize(text)                                          # tokenizasyon
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]    # stopwords temizleme
    return tokens

df["tokens"] = df["Review Text"].apply(preprocess)

# -----------------------------
# 4. Word2Vec Modeli Eğitimi
# -----------------------------
w2v_model = Word2Vec(
    sentences=df["tokens"],
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1                                      # Skip-gram daha iyi sonuç verir
)

# Kelime vektörü alma örneği
print(w2v_model.wv["good"])

# -----------------------------
# 5. Cümleleri Vektöre Dönüştürme
# -----------------------------

def vectorize(tokens):
    """Cümledeki kelimelerin ortalama Word2Vec vektörü"""
    vecs = []
    for token in tokens:
        if token in w2v_model.wv:
            vecs.append(w2v_model.wv[token])
    if len(vecs) == 0:
        return np.zeros(100)
    return np.mean(vecs, axis=0)

df["vector"] = df["tokens"].apply(vectorize)

X = np.vstack(df["vector"].values)
y = df["Rating"].values




ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)


# -----------------------------
# 6. Train-Test Bölme
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# -----------------------------
# 7. 3 Farklı Modelin Eğitimi
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc}")
    print(classification_report(y_test, preds))

# -----------------------------
# 8. Başarı Karşılaştırma Tablosu
# -----------------------------
results_df = pd.DataFrame.from_dict(results, orient='index', columns=["Accuracy"])
print(results_df)

# -----------------------------
# 9. Başarı Grafiği
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(results_df.index, results_df["Accuracy"])
plt.title("Sınıflandırma Algoritmalarının Karşılaştırılması")
plt.ylabel("Accuracy")
plt.show()

# -----------------------------
# 10. En Başarılı Modeli Seçme
# -----------------------------
best_model_name = results_df["Accuracy"].idxmax()
best_model = models[best_model_name]

print("\nEn başarılı model:", best_model_name)

# -----------------------------
# 11. Yeni Yorumların Tahmini
# -----------------------------
new_reviews = [
    "Basically the best thing since sliced bread. Definitely worth the trip outside of Champaign",
    "Ice cream and milk shake were good there are a lot of varieties. Staff so not so friendly"
]

new_tokens = [preprocess(r) for r in new_reviews]
new_vectors = np.vstack([vectorize(t) for t in new_tokens])
preds = best_model.predict(new_vectors)

for text, pred in zip(new_reviews, preds):
    print("\nReview:", text)
    print("Predicted Rating:", pred)

