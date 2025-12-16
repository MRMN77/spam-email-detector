import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import clean_text


def main():
    # 1. Load dataset
    data_path = "data/raw/spam.csv"
    df = pd.read_csv(data_path)

    # Rename columns if needed
    df.columns = ["label", "text"]

    # 2. Encode labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # 3. Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # 4. Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 7. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 8. Save model & vectorizer
    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\nModel and vectorizer saved successfully.")


if __name__ == "__main__":
    main()
