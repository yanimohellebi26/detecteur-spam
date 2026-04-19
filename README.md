# Détecteur de Spam + Chatbot Telegram

Système de **détection de spam par machine learning** couplé à un **chatbot Telegram** pour classifier des messages en temps réel.

## Fonctionnalités

- Détection spam/ham avec modèle ML entraîné (`spam_model.joblib`)
- Chatbot Telegram interactif pour tester le modèle
- Visualisations : matrice de confusion, courbe ROC

## Structure

```
├── entrainnement.py         # Entraînement du modèle (scikit-learn)
├── detection.py             # Module de détection spam
├── chatbot.py               # Bot Telegram (python-telegram-bot)
├── traitementdonnees.py     # Prétraitement du dataset
├── test_model.py            # Tests du modèle
├── spam.csv                 # Dataset d'entraînement
├── spam_model.joblib        # Modèle sérialisé
├── confusion_matrix.png     # Matrice de confusion
└── roc_curve.png            # Courbe ROC-AUC
```

## Installation

```bash
pip install scikit-learn pandas numpy python-telegram-bot joblib
```

## Utilisation

```bash
# Entraîner le modèle
python entrainnement.py

# Tester le modèle
python test_model.py

# Lancer le chatbot Telegram
python chatbot.py
```

## Technologies

`Python` · `scikit-learn` · `Telegram Bot API` · NLP · Machine Learning
