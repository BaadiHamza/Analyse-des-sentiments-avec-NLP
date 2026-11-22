# Analyse Automatique de Sentiments (NLP) — Application Streamlit

**Module :** IA et ses Applications (Projet N° 05)  
**Binôme 5 :** Hamza BAADI & Nadya OUFDIME  
**Semaine de soutenance :** S10  

---

## 📋 Présentation du Projet
Ce projet consiste à concevoir une chaîne de traitement automatique du langage naturel (NLP) permettant de prédire la polarité d'avis clients (**Positif** ou **Négatif**), de mesurer la fiabilité expérimentale et de déployer une interface web interactive sous **Streamlit** pour la démonstration.

### 🌟 Fonctionnalités de l'Application :
1. **🎯 Démonstrateur en direct :**
   - Zone de saisie d'un avis client libre.
   - Boutons de pré-remplissage avec des exemples typiques (*positif franc, négatif franc, avis ambigu, négation complexe, ironie*).
   - Prédiction de la polarité avec jauge de score de confiance (%).
   - **Message d'avertissement automatique** pour les cas ambigus (score proche de 50%).
   - **Explicabilité TF-IDF :** affichage des termes positifs et négatifs ayant le plus influencé la décision.

2. **📁 Analyse par lot (Fichier CSV) — Bonus :**
   - Importation d'un fichier CSV d'avis clients.
   - Prédiction instantanée sur l'ensemble des commentaires.
   - Statistiques globales et téléchargement du CSV annoté avec prédictions et scores de confiance.

3. **📊 Évaluation & Comparaison des modèles :**
   - Comparaison rigoureuse de 3 modèles :
     - **Baseline :** `DummyClassifier(strategy='most_frequent')`
     - **Modèle 1 :** `Pipeline(TfidfVectorizer, LogisticRegression)`
     - **Modèle 2 :** `Pipeline(TfidfVectorizer, MultinomialNB)`
   - Métriques : Accuracy, Précision, Rappel, F1-score.
   - Matrice de confusion graphique (Seaborn heatmap).
   - Étude de l'impact des hyperparamètres (`ngram_range`, `min_df`).
   - **Analyse concrète de 5 cas d'erreurs linguistiques :**
     - Négation (« pas bon » / « pas mauvais »)
     - Ironie et Sarcasme
     - Mots hors-vocabulaire / fautes (OOV)
     - Avis mixtes / ambigus
     - Polysémie et changement de domaine.

4. **ℹ️ À propos & Méthodologie :**
   - Cadrage du problème, description du corpus, TF-IDF et limites du modèle.

---

## 🚀 Installation et Lancement

### 1. Activer l'environnement virtuel ou installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. (Optionnel) Entraîner et sauvegarder les modèles
Pour ré-entraîner les modèles et générer les métriques d'évaluation :
```bash
python3 train_model.py
```
Cela produit :
- `models/sentiment_pipeline.joblib`
- `models/evaluation_metrics.json`

### 3. Lancer l'application Streamlit
```bash
streamlit run app.py
```
L'interface s'ouvrira automatiquement dans votre navigateur à l'adresse :  
👉 **`http://localhost:8501`**

---

## 📁 Structure du Répertoire
```
analyse_sentiment_streamlit/
├── app.py                      # Application interactive Streamlit
├── train_model.py              # Script d'entraînement, comparaison et sauvegarde
├── sample_reviews.csv          # Fichier CSV de démonstration pour le batch
├── requirements.txt            # Dépendances Python requises
├── README.md                   # Documentation complète
└── models/
    ├── sentiment_pipeline.joblib  # Pipeline sauvegardé (TF-IDF + Régression Logistique)
    └── evaluation_metrics.json   # Métriques précalculées pour l'onglet Évaluation
```
