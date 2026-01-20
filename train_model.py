import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("all_tickets_processed_improved_v3.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text

df["clean_text"] = df["Document"].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["Topic_group"]

# Train model (VERY stable)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save
pickle.dump(model, open("model/category_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("Model trained & saved successfully")
