# test_modele.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from entrainnement import SpamTrainer

def evaluer_modele():
    # Initialisation avec logging
    print("⏳ Chargement des données et entraînement du modèle...")
    trainer = SpamTrainer('spam.csv')
    X_test, y_test = trainer.train()
    
    # Prédictions
    print("\n🔍 Évaluation des performances...")
    y_pred = trainer.model.predict(X_test)
    y_proba = trainer.model.predict_proba(X_test)[:, 1]
    
    # Rapport de classification détaillé
    print("\n📊 Rapport de classification:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')
    plt.savefig('confusion_matrix.png')
    print("\n✅ Matrice de confusion sauvegardée dans confusion_matrix.png")
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux Faux Positifs')
    plt.ylabel('Taux Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("✅ Courbe ROC sauvegardée dans roc_curve.png")

    # Analyse des erreurs
    test_data = trainer.load_data().iloc[y_test.index]
    test_data['prediction'] = y_pred
    errors = test_data[test_data['label'] != test_data['prediction']]
    
    print(f"\n🔧 Erreurs d'analyse ({len(errors)} cas):")
    print(errors.sample(n=min(5, len(errors)))[['message', 'label', 'prediction']])
    

    # Sauvegarde des erreurs
    errors.to_csv('erreurs_analyse.csv', index=False)
    print("\n📝 Échantillon d'erreurs sauvegardé dans erreurs_analyse.csv")

if __name__ == '__main__':
    evaluer_modele()
    print("\n🎯 Analyse terminée. Vérifiez les fichiers générés!")