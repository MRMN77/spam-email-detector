import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# فقط یک‌بار لازم است
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean and preprocess email text
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove links
    text = re.sub(r"\S+@\S+", "", text)          # remove emails
    text = re.sub(r"[^a-z]", " ", text)          # keep only letters
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)
