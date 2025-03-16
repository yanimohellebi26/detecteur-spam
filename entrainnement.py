# entrainement.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Meilleur pour le texte
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Initialisation des composants NLP
STEMMER = SnowballStemmer("english")
STOP_WORDS = set(stopwords.words("english"))

class SpamTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Capture les bigrammes
            max_features=3000,    # Réduit la dimensionnalité
            stop_words=list(STOP_WORDS)
        )
        self.model = LogisticRegression(
            class_weight="balanced",  # Gère le déséquilibre
            max_iter=1000
        )

    def clean_text(self, text):
        # Nettoyage avancé du texte
        text = re.sub(r"[^a-zA-Z]", " ", text.lower())
        words = [STEMMER.stem(word) for word in text.split() if word not in STOP_WORDS]
        return " ".join(words)

    def load_data(self):
        data = pd.read_csv(self.data_path, encoding="ISO-8859-1", usecols=["v1", "v2"])
        data.columns = ["label", "message"]
        data["cleaned"] = data["message"].apply(self.clean_text)
        return data

    def train(self, test_size=0.2):
        data = self.load_data()
        
        # Vectorisation
        X = self.vectorizer.fit_transform(data["cleaned"])
        y = data["label"].map({"ham": 0, "spam": 1})  # Encodage direct
        
        # Split équilibré
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y,  # Maintient la distribution des classes
            random_state=42
        )
        
        # Entraînement
        self.model.fit(X_train, y_train)
        
        # Évaluation
        print("\nRapport de classification:")
        print(classification_report(y_test, self.model.predict(X_test)))
        
        return X_test, y_test

    def save_model(self, model_path="spam_model.joblib"):
        joblib.dump({
            "model": self.model,
            "vectorizer": self.vectorizer,
            "classes_": ["ham", "spam"]  # Pour l'inverse transform
        }, model_path)

# Dans la partie __main__ du fichier entrainnement.py
if __name__ == '__main__':
    # Déplacer l'import ici
    import nltk
    
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    trainer = SpamTrainer("spam.csv")
    trainer.train()
    trainer.save_model()
    print("\nModèle sauvegardé avec succès!")