# detection.py
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class SpamDetector:
    def __init__(self, model_path='spam_model.joblib'):
        self.STEMMER = SnowballStemmer("english")
        self.STOP_WORDS = set(stopwords.words("english"))
        self.load_model(model_path)
        
    def clean_text(self, text):
        """Identique à la méthode de nettoyage dans entrainement.py"""
        text = re.sub(r"[^a-zA-Z]", " ", text.lower())
        words = [self.STEMMER.stem(word) for word in text.split() if word not in self.STOP_WORDS]
        return " ".join(words)

    def load_model(self, model_path):
        """Chargement du modèle avec gestion d'erreur améliorée"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.classes = model_data['classes_']  # ['ham', 'spam']
        except Exception as e:
            raise ValueError(f"Erreur de chargement du modèle: {str(e)}")

    def predict(self, text, confidence_threshold=0.7):
        """Prédiction avec seuil de confiance ajustable"""
        try:
            # Nettoyage du texte identique à l'entraînement
            cleaned_text = self.clean_text(text)
            
            # Vectorisation
            vector = self.vectorizer.transform([cleaned_text])
            
            # Prédiction probabiliste
            probas = self.model.predict_proba(vector)[0]
            spam_prob = probas[1]  # Probabilité pour la classe 'spam'
            
            # Décision avec seuil
            is_spam = spam_prob >= confidence_threshold
            label = self.classes[1 if is_spam else 0]
            
            return {
                'is_spam': is_spam,
                'label': label,
                'confidence': spam_prob,
                'original_text': text
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'input_text': text
            }

# Exemple d'utilisation
if __name__ == '__main__':
    detector = SpamDetector()
    test_text = "WINNER!! Claim your free prize now!!!"
    print(detector.predict(test_text))