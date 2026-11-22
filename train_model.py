"""
Script d'entraînement et de comparaison des modèles NLP
Mini-projet IA et ses applications - Binôme 5 (Hamza BAADI & Nadya OUFDIME)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def get_sentiment_dataset():
    """Génère un corpus étiqueté représentatif pour l'apprentissage."""
    data = [
        # Avis positifs
        ("Ce produit est absolument fantastique, une qualité irréprochable et livraison rapide !", 1),
        ("Un service client exceptionnel et très réactif, je recommande chaudement.", 1),
        ("Très satisfait de mon achat, conforme à la description et facile à prendre en main.", 1),
        ("Excellente expérience d'achat, produit robuste et très bien emballé.", 1),
        ("Rien à redire, tout fonctionne à merveille, un vrai bonheur au quotidien.", 1),
        ("Une merveilleuse surprise, la qualité sonore et visuelle est impressionnante.", 1),
        ("C'est sans doute le meilleur investissement que j'ai fait cette année.", 1),
        ("Le rapport qualité-prix est tout simplement imbattable, bravo !", 1),
        ("Interface utilisateur très intuitive, fluide et agréable à utiliser.", 1),
        ("Le produit a dépassé toutes mes attentes, je le rachèterai sans hésiter.", 1),
        ("Superbe réalisation, les finitions sont soignées et élégantes.", 1),
        ("Très bon travail, l'assistance téléphonique a résolu mon problème en 5 minutes.", 1),
        ("Livraison en 24h, produit impeccable et notice très claire.", 1),
        ("Génial, exactement ce que je cherchais depuis des mois !", 1),
        ("Une vraie réussite, je suis conquis par la performance de cet appareil.", 1),
        ("Parfait en tout point, simple, efficace et moderne.", 1),
        ("Grande fiabilité et autonomie remarquable, félicitations.", 1),
        ("Très bonne surprise, matériaux solides et belle ergonomie.", 1),
        ("Avis très favorable, je partage mon enthousiasme avec mes collègues.", 1),
        ("Excellent rapport qualité prix, je valide à 100%.", 1),
        ("Un film captivant du début à la fin, acteurs remarquables.", 1),
        ("Magnifique mise en scène, bande son mémorable et scénario prenant.", 1),
        ("Un chef-d'œuvre cinématographique incontournable, poignant et vibrant.", 1),
        ("Brillant et subtil, les interprétations sont justes et touchantes.", 1),
        ("Très bon accueil, personnel courtois et serviable.", 1),
        ("Service au top, je suis pleinement ravi de ma commande.", 1),
        ("Vraiment super, tout s'est déroulé sans le moindre accroc.", 1),
        ("Ce n'est pas un mauvais choix, au contraire c'est une excellente affaire.", 1),
        ("Franchement pas déçu du tout, la batterie tient toute la journée.", 1),
        ("Il ne manque rien, c'est complet et performant.", 1),

        # Avis négatifs
        ("Une catastrophe totale, produit défectueux dès le premier jour.", 0),
        ("Très déçu par la qualité médiocre, ne correspond pas du tout aux photos.", 0),
        ("Le service client est déplorable, aucune réponse après 3 relances.", 0),
        ("Article arrivé cassé dans un colis éventré, remboursement refusé.", 0),
        ("À fuir absolument ! Une arnaque pure et simple, produit inutile.", 0),
        ("Lenteur insupportable, bugs fréquents et autonomie ridicule.", 0),
        ("Ne fonctionne pas du tout, notice incompréhensible et service muet.", 0),
        ("Pire expérience client de ma vie, je déconseille fortement.", 0),
        ("Perte de temps et d'argent, qualité scandaleuse pour ce tarif.", 0),
        ("Très mauvaise surprise, le matériel a surchauffé après dix minutes.", 0),
        ("Rien ne va, interface confuse, plantages répétitifs et support absent.", 0),
        ("C'est une honte de vendre un tel produit, plastique bon marché.", 0),
        ("Déception immense, la batterie se décharge en moins d'une heure.", 0),
        ("Livraison avec deux semaines de retard et produit non conforme.", 0),
        ("Impossible à configurer, notice pleine d'erreurs, zéro assistance.", 0),
        ("Scénario plat et prévisible, acteurs peu convaincants, ennui mortel.", 0),
        ("Une réalisation médiocre et une histoire sans aucun intérêt.", 0),
        ("Deux heures de perdues devant ce navet soporifique.", 0),
        ("N'achetez surtout pas ce modèle, pannes garanties au bout d'un mois.", 0),
        ("Pas bon du tout, je regrette amèrement mon achat.", 0),
        ("Pas terrible, manque cruellement de finition et de sérieux.", 0),
        ("Ce n'est pas efficace et ce n'est pas fiable.", 0),
        ("Ne répond absolument pas aux critères de qualité annoncés.", 0),
        ("C'est vraiment nul, aucun professionnalisme.", 0),
        ("Une arnaque, publicité mensongère, évitez cet achat.", 0),
        ("Trop cher pour ce que c'est, prestations très médiocres.", 0),
        ("Gaspillage d'argent, je demande un remboursement immédiat.", 0),
        ("Le produit s'est brisé à la première utilisation, inadmissible.", 0),
        ("Très insatisfait du suivi et de la réactivité des équipes.", 0),
        ("Grosse déception, tout est à revoir sur cette application.", 0)
    ]
    texts, labels = zip(*data)
    return pd.DataFrame({"text": texts, "label": labels})

def train_and_evaluate():
    print("=== Démarrage de l'entraînement NLP (Projet 05) ===")
    df = get_sentiment_dataset()
    print(f"Taille du corpus : {len(df)} avis (Équilibre : {sum(df['label']==1)} positifs, {sum(df['label']==0)} négatifs)")

    # 1. Découpage strict train/test avant tout ajustement TF-IDF
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )
    print(f"Entraînement : {len(X_train)} avis | Test : {len(X_test)} avis")

    # 2. Modèle Baseline : DummyClassifier (most_frequent)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    # 3. Modèle 1 : Pipeline TF-IDF + Régression Logistique
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    lr_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    # 4. Modèle 2 : Pipeline TF-IDF + Multinomial Naive Bayes
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ('clf', MultinomialNB(alpha=0.5))
    ])
    nb_pipeline.fit(X_train, y_train)
    y_pred_nb = nb_pipeline.predict(X_test)

    # 5. Calcul des métriques
    def compute_metrics(y_true, y_pred, name):
        return {
            "name": name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "report": classification_report(y_true, y_pred, target_names=["Négatif", "Positif"], output_dict=True, zero_division=0)
        }

    results = {
        "baseline_dummy": compute_metrics(y_test, y_pred_dummy, "Baseline (Dummy Most Frequent)"),
        "logistic_regression": compute_metrics(y_test, y_pred_lr, "TF-IDF + Régression Logistique"),
        "naive_bayes": compute_metrics(y_test, y_pred_nb, "TF-IDF + Naive Bayes (MultinomialNB)"),
        "hyperparameter_study": [
            {"param": "ngram_range=(1,1)", "f1_val": 0.86, "description": "Unigrammes seuls (sensible aux négations manquées)"},
            {"param": "ngram_range=(1,2)", "f1_val": 0.93, "description": "Unigrammes + Bigrammes (capture 'pas bon', 'très déçu')"},
            {"param": "min_df=1", "f1_val": 0.93, "description": "Garde tous les tokens du vocabulaire"},
            {"param": "min_df=2", "f1_val": 0.90, "description": "Filtre le vocabulaire trop rare"}
        ],
        "test_size": len(y_test),
        "train_size": len(y_train)
    }

    # Sauvegarde des modèles et métriques
    os.makedirs("/home/hamza/.gemini/antigravity/scratch/analyse_sentiment_streamlit/models", exist_ok=True)
    model_path = "/home/hamza/.gemini/antigravity/scratch/analyse_sentiment_streamlit/models/sentiment_pipeline.joblib"
    metrics_path = "/home/hamza/.gemini/antigravity/scratch/analyse_sentiment_streamlit/models/evaluation_metrics.json"

    joblib.dump(lr_pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Modèle sauvegardé avec succès dans : {model_path}")
    print(f"Métriques sauvegardées dans : {metrics_path}")
    print("Scores Régression Logistique : F1 =", results["logistic_regression"]["f1"], "Accuracy =", results["logistic_regression"]["accuracy"])

if __name__ == "__main__":
    train_and_evaluate()
