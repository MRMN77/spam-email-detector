import joblib
from preprocessing import clean_text


def predict_email(text: str) -> str:
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]

    return "SPAM 🚨" if prediction == 1 else "HAM ✅"


def main():
    print("=== Spam Email Detector ===")
    print("Type your email text below (or type 'exit' to quit)\n")

    while True:
        text = input("Email text: ")

        if text.lower() == "exit":
            print("Goodbye 👋")
            break

        result = predict_email(text)
        print("Result:", result)
        print("-" * 30)


if __name__ == "__main__":
    main()
