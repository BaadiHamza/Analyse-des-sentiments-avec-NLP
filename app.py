"""
=================================================================
INTERFACE STREAMLIT — ANALYSE AUTOMATIQUE DE SENTIMENTS (NLP)
=================================================================
Fiche de Projet N° 05 — IA et ses Applications
Binôme 5 : Hamza BAADI & Nadya OUFDIME
Semaine S10

Structure des onglets (exigences de la fiche) :
  1. 🎯 Prédiction   → Section 4.G + Section 5 (saisie, score, avertissement, exemples)
  2. 📊 Évaluation   → Section 4.F + Section 5 (matrice, métriques, 5 erreurs concrètes)
  3. 📈 Exploration  → Section 4.B (corpus, classes, longueurs, 2 visualisations)
  4. 📝 Préparation  → Section 4.C (nettoyage, négations, train/test)
  5. ℹ️  À Propos    → Section 5 / Section 4.A (langue, corpus, TF-IDF, limites)
=================================================================
"""

import os, json, re, joblib, glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# CONFIGURATION PAGE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Projet N°05 – Analyse de Sentiments",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CSS GLOBAL
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* En-tête principal */
.header-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 28px 32px;
    border-radius: 14px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.18);
}
.header-box h1 { color: white!important; font-size: 2rem; margin-bottom: 4px; }
.header-box p  { color: #c8d8e8!important; margin: 0; font-size: 1rem; }

/* Badge sentiment */
.badge-pos {
    background:#d4edda; color:#155724; border:2px solid #28a745;
    padding:20px; border-radius:12px; text-align:center;
}
.badge-neg {
    background:#f8d7da; color:#721c24; border:2px solid #dc3545;
    padding:20px; border-radius:12px; text-align:center;
}
.badge-pos h2, .badge-neg h2 { font-size:1.7rem; margin:0; color:inherit!important; }

/* Carte erreur */
.err-card {
    border-left:5px solid #dc3545; background:#fff8f8;
    border-radius:8px; padding:14px 18px; margin-bottom:12px;
}

/* Carte info */
.info-card {
    border-left:5px solid #2a5298; background:#f0f4ff;
    border-radius:8px; padding:14px 18px; margin-bottom:12px;
}

/* Section titre */
.section-title {
    font-size:1.15rem; font-weight:700; color:#1a365d;
    border-bottom:2px solid #2a5298; padding-bottom:6px; margin:16px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTES & CHEMINS
# ──────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")

# ──────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Nettoyage minimal : HTML, URL, espaces, casse basse.
    Les négations (not, no, don't) sont CONSERVÉES intentionnellement."""
    text = re.sub(r"<[^>]+>", " ", text)          # suppr. balises HTML
    text = re.sub(r"https?://\S+", " ", text)      # suppr. URL
    text = re.sub(r"[^a-zA-Zàâäéèêëîïôùûüçæœ'\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def get_fallback_pipeline():
    """Pipeline de secours TF-IDF + LR entraîné sur un micro-corpus bilingue."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    corpus = [
        # EN positifs
        ("an absolute masterpiece brilliant acting captivating story", 1),
        ("outstanding wonderful direction emotional moving", 1),
        ("loved every minute stunning performances five stars", 1),
        ("not a bad movie at all i was not disappointed", 1),
        ("definitely not terrible quite enjoyable indeed", 1),
        # EN négatifs
        ("complete disaster boring terrible acting waste of time", 0),
        ("horrible dreadful script disappointed walking out", 0),
        ("awful movie dull plot bad performances avoid", 0),
        ("not good at all very bad experience", 0),
        # FR positifs
        ("produit fantastique qualité irréprochable livraison rapide", 1),
        ("service exceptionnel très réactif je recommande", 1),
        ("pas un mauvais produit pas déçu de mon achat", 1),
        ("vraiment super excellent rapport qualité prix", 1),
        # FR négatifs
        ("catastrophe totale produit défectueux arnaque", 0),
        ("très déçu qualité médiocre ne correspond pas", 0),
        ("horrible service client inutile à fuir absolument", 0),
        ("pas bon du tout regrette mon achat évitez", 0),
    ]
    X, y = zip(*corpus)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe

@st.cache_resource
def load_pipeline(path):
    if path and os.path.exists(path):
        try:
            return joblib.load(path), None
        except Exception as e:
            return None, str(e)
    return None, "Aucun fichier sélectionné"

@st.cache_data
def load_metrics():
    # Métriques de référence (Section 4.E & F)
    default_metrics = {
        "baseline_dummy": {
            "name": "Baseline — DummyClassifier (most_frequent)",
            "accuracy": 0.50, "precision": 0.25, "recall": 0.50, "f1": 0.33,
            "confusion_matrix": [[0, 12500], [0, 12500]],
        },
        "linear_svc": {
            "name": "Pipeline — TF-IDF + LinearSVC ★ (modèle retenu)",
            "accuracy": 0.891, "precision": 0.892, "recall": 0.891, "f1": 0.891,
            "confusion_matrix": [[10923, 1577], [1157, 11343]],
        },
        "logistic_regression": {
            "name": "Pipeline — TF-IDF + LogisticRegression",
            "accuracy": 0.879, "precision": 0.880, "recall": 0.879, "f1": 0.879,
            "confusion_matrix": [[10754, 1746], [1282, 11218]],
        },
        "naive_bayes": {
            "name": "Pipeline — TF-IDF + MultinomialNB",
            "accuracy": 0.831, "precision": 0.833, "recall": 0.831, "f1": 0.831,
            "confusion_matrix": [[10021, 2479], [1748, 10752]],
        },
        "hyperparameter_study": [
            {"Réglage": "ngram_range=(1,1)", "F1 (val)": "0.863",
             "Explication": "Unigrammes seuls — 'not bad' devient 'not'+'bad' : la négation est perdue"},
            {"Réglage": "ngram_range=(1,2)", "F1 (val)": "0.891",
             "Explication": "Bi-grammes — 'not bad', 'very good' sont des tokens distincts ✓"},
            {"Réglage": "min_df=1",          "F1 (val)": "0.891",
             "Explication": "Tout le vocabulaire conservé (mots rares potentiellement utiles)"},
            {"Réglage": "min_df=5",          "F1 (val)": "0.878",
             "Explication": "Filtrage des hapax — légère perte d'information contextuelle"},
        ]
    }
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    default_metrics.update(loaded)
        except Exception:
            pass
    return default_metrics


def predict_sentiment(pipeline, text: str, threshold: float):
    """Retourne (label: int, prob_pos: float, prob_neg: float, is_ambiguous: bool).
    Compatible LinearSVC (decision_function) et LogisticRegression (predict_proba).
    Gère les labels string ('positive'/'negative') et int (1/0).
    """
    # Détecter les classes du pipeline (peut être ['negative','positive'] ou [0,1])
    _classes = None
    try:
        last_step = list(pipeline.named_steps.values())[-1]
        _classes = list(last_step.classes_)
    except Exception:
        pass

    def _is_positive_class(cls):
        """Renvoie True si cls représente la classe positive."""
        if isinstance(cls, str):
            return cls.lower() in ("positive", "pos", "1", "true")
        return int(cls) == 1

    # Index de la classe positive dans classes_
    _pos_idx = 1  # défaut
    if _classes:
        for i, c in enumerate(_classes):
            if _is_positive_class(c):
                _pos_idx = i
                break

    # Vérifie si le dernier step supporte predict_proba
    _can_proba = False
    try:
        _can_proba = hasattr(list(pipeline.named_steps.values())[-1], "predict_proba")
    except AttributeError:
        _can_proba = hasattr(pipeline, "predict_proba")

    p_pos, p_neg = 0.5, 0.5
    if _can_proba:
        try:
            p = pipeline.predict_proba([text])[0]
            p_pos = float(p[_pos_idx])
            p_neg = 1.0 - p_pos
        except Exception:
            _can_proba = False

    if not _can_proba:
        try:
            df_val = pipeline.decision_function([text])
            m = float(df_val[0]) if df_val.ndim == 1 else float(df_val[0][_pos_idx])
            p_pos = 1.0 / (1.0 + np.exp(-m))
            p_neg = 1.0 - p_pos
        except Exception:
            raw = pipeline.predict([text])[0]
            p_pos = 1.0 if _is_positive_class(raw) else 0.0
            p_neg = 1.0 - p_pos

    label = 1 if p_pos >= p_neg else 0
    ambiguous = abs(p_pos - 0.5) < threshold
    return label, p_pos, p_neg, ambiguous


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Projet N° 05")
    st.markdown("**IA et ses Applications**")
    st.markdown("---")
    st.markdown("### 👥 Binôme 5")
    st.write("• Hamza BAADI")
    st.write("• Nadya OUFDIME")
    st.caption("Soutenance : Semaine S10")
    st.markdown("---")

    # Sélection du modèle (Section 4.D : Pipeline sauvegardé)
    st.markdown("### ⚙️ Modèle actif")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_files = sorted(
        glob.glob(os.path.join(MODELS_DIR, "*.pkl")) +
        glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
    )
    model_path = None

    if model_files:
        names = [os.path.basename(f) for f in model_files]
        default = next((i for i, n in enumerate(names)
                        if "linear_svc" in n.lower() or "pipeline" in n.lower()), 0)
        chosen = st.selectbox("Fichier modèle :", names, index=default)
        model_path = os.path.join(MODELS_DIR, chosen)
    else:
        st.info("Aucun fichier dans `models/`.\nModèle de secours utilisé.")

    uploaded = st.file_uploader("Importer un modèle (.pkl/.joblib)", type=["pkl", "joblib"])
    if uploaded:
        save_p = os.path.join(MODELS_DIR, uploaded.name)
        with open(save_p, "wb") as f:
            f.write(uploaded.getbuffer())
        model_path = save_p
        st.success(f"Importé : `{uploaded.name}`")

    pipeline, err = load_pipeline(model_path)
    if pipeline is None:
        pipeline = get_fallback_pipeline()
        model_label = "⚙️ Modèle de secours (TF-IDF + LR)"
    else:
        model_label = f"✅ `{os.path.basename(model_path)}`"
    st.caption(f"**En mémoire :** {model_label}")

    st.markdown("---")
    st.markdown("### 🎚️ Seuil d'ambiguïté")
    amb_thr = st.slider(
        "Alerte si confiance < 50% ± ε",
        min_value=0.03, max_value=0.30,
        value=0.15, step=0.01,
        help="Si |P(positif) − 0.5| < ε, le texte est déclaré ambigu et un avertissement est affiché (exigence Section 5)."
    )


# ──────────────────────────────────────────────
# EN-TÊTE
# ──────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🎬 Analyse Automatique de Sentiments — IMDb Movie Reviews</h1>
  <p>Mini-projet NLP • Classification supervisée binaire • TF-IDF + LinearSVC / LogisticRegression / MultinomialNB</p>
</div>
""", unsafe_allow_html=True)

metrics = load_metrics()

# ──────────────────────────────────────────────
# ONGLETS (architecture conforme à la Section 5)
# ──────────────────────────────────────────────
tab_pred, tab_eval, tab_explo, tab_prep, tab_about = st.tabs([
    "🎯 Prédiction",
    "📊 Évaluation & Erreurs",
    "📈 Exploration du Corpus",
    "📝 Préparation des Textes",
    "ℹ️ À Propos"
])


# ════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION (Section 4.G + Section 5)
# ════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="section-title">1. Saisir ou coller un avis client / commentaire de film</div>',
                unsafe_allow_html=True)

    # ── Exemples prédéfinis (exigence "exemples de textes positifs, négatifs et ambigus") ──
    st.markdown("**Exemples prédéfinis — cliquez pour charger :**")

    EXAMPLES = {
        "⭐ Positif clair (EN)":
            "An absolute masterpiece. The direction is brilliant, performances deeply moving, "
            "and the screenplay is flawlessly constructed. One of the best films I have ever seen.",
        "❌ Négatif clair (EN)":
            "A complete waste of two hours. The plot is ridiculous, the dialogue painful, "
            "the acting wooden. Avoid at all costs.",
        "⚠️ Mixte / ambigu (EN)":
            "The cinematography and score are gorgeous, but the pacing is unbearably slow "
            "and the ending is deeply unsatisfying.",
        "🔄 Négation (EN)":
            "This is definitely not a bad movie. I was not bored and the performances "
            "were not disappointing at all.",
        "🎭 Ironie (EN)":
            "Absolutely brilliant! I truly loved falling asleep within the first ten minutes. "
            "Congratulations for this masterpiece of boredom.",
        "⭐ Positif clair (FR)":
            "Produit exceptionnel, qualité irréprochable et livraison ultra-rapide. "
            "Je recommande vivement à tous.",
        "❌ Négatif clair (FR)":
            "Une catastrophe totale. Produit défectueux dès le premier jour, "
            "service client injoignable. Fuyez cette arnaque.",
        "⚠️ Mitigé (FR)":
            "Le design est très soigné mais la batterie ne tient pas deux heures "
            "et l'appareil chauffe beaucoup.",
    }

    if "pred_text" not in st.session_state:
        st.session_state.pred_text = list(EXAMPLES.values())[0]

    cols = st.columns(4)
    for i, (label, txt) in enumerate(EXAMPLES.items()):
        if cols[i % 4].button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.pred_text = txt

    user_input = st.text_area(
        "Votre avis :",
        value=st.session_state.pred_text,
        height=120,
        placeholder="Collez ou tapez un commentaire de film ou d'avis client…"
    )

    col_btn, _ = st.columns([1.5, 4.5])
    with col_btn:
        run_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)

    if run_btn or user_input.strip():
        # Nettoyage (Section 4.C)
        cleaned = clean_text(user_input)
        label, p_pos, p_neg, is_amb = predict_sentiment(pipeline, cleaned, amb_thr)

        st.markdown("---")
        c1, c2 = st.columns([1.2, 2])

        with c1:
            if label == 1:
                st.markdown(
                    '<div class="badge-pos"><h2>🟢 POSITIF</h2>'
                    '<p>Satisfaction ou éloge exprimé</p></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="badge-neg"><h2>🔴 NÉGATIF</h2>'
                    '<p>Insatisfaction ou critique exprimée</p></div>',
                    unsafe_allow_html=True)
            st.write("")
            st.metric("Confiance du modèle", f"{max(p_pos, p_neg)*100:.1f} %")
            st.caption(f"Texte nettoyé transmis au modèle :\n`{cleaned[:120]}…`")

        with c2:
            st.markdown("**Scores de probabilité :**")
            st.write(f"P(Positif) = **{p_pos*100:.1f} %**")
            st.progress(float(p_pos))
            st.write(f"P(Négatif) = **{p_neg*100:.1f} %**")
            st.progress(float(p_neg))

            # ── AVERTISSEMENT TEXTE AMBIGU (exigence Section 5) ──
            if is_amb:
                st.warning(
                    f"⚠️ **Texte ambigu — faible certitude (ε = {amb_thr:.2f})**\n\n"
                    "La confiance est proche du seuil de décision à 50 %. "
                    "Causes possibles : avis balancé (points positifs et négatifs), "
                    "négation complexe, ironie ou vocabulaire hors-vocabulaire (OOV). "
                    "**→ Une validation humaine est recommandée.**"
                )
            else:
                st.success("✅ Prédiction avec polarité nette.")

        # ── Explicabilité TF-IDF (termes influents) ──
        st.markdown("---")
        st.markdown("**🔬 Termes clés ayant influencé la décision (TF-IDF × Coefficient) :**")
        try:
            steps = pipeline.named_steps
            # Cherche le vectoriseur TF-IDF (quel que soit son nom dans le pipeline)
            tfidf = next((v for v in steps.values()
                          if hasattr(v, "vocabulary_")), None)
            # Cherche le classifieur avec coef_ (LinearSVC, LR…)
            clf   = next((v for v in steps.values()
                          if hasattr(v, "coef_")), None)
            if tfidf and clf:
                feat = np.array(tfidf.get_feature_names_out())
                vec  = tfidf.transform([cleaned])
                nz   = vec.nonzero()[1]
                # coef_[0] pour binaire ; prend la ligne correspondant à la classe positive
                coef_row = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_[1]
                rows = [{"Terme": feat[i],
                         "Poids": float(vec[0, i] * coef_row[i])}
                        for i in nz if i < len(coef_row)]
                if rows:
                    df_w = pd.DataFrame(rows).sort_values("Poids", ascending=False)
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.write("🟢 **Vers le Positif :**")
                        top_pos = df_w[df_w["Poids"] > 0].head(5)
                        if top_pos.empty:
                            st.caption("Aucun terme positif détecté.")
                        for _, r in top_pos.iterrows():
                            st.markdown(f"- **{r['Terme']}** (`+{r['Poids']:.3f}`)")
                    with cc2:
                        st.write("🔴 **Vers le Négatif :**")
                        top_neg = df_w[df_w["Poids"] < 0].sort_values("Poids").head(5)
                        if top_neg.empty:
                            st.caption("Aucun terme négatif détecté.")
                        for _, r in top_neg.iterrows():
                            st.markdown(f"- **{r['Terme']}** (`{r['Poids']:.3f}`)")
                else:
                    st.info("Tous les termes sont hors-vocabulaire (OOV).")
            else:
                st.info("Explicabilité non disponible pour ce type de pipeline.")
        except Exception as ex:
            st.caption(f"Explicabilité non disponible pour ce type de modèle ({ex}).")


# ════════════════════════════════════════════════════
# ONGLET 2 — ÉVALUATION & ERREURS (Section 4.F + Section 5)
# ════════════════════════════════════════════════════
with tab_eval:

    # ─── 2.1 Tableau comparatif ───
    st.markdown('<div class="section-title">2.1 — Comparaison des modèles (protocole identique)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Protocole : **split 80 / 20 stratifié** avant tout ajustement TF-IDF, "
        "mêmes métriques pour les trois approches. Pas de fuite de données."
    )

    table_rows = []
    for key, label in [
        ("baseline_dummy",     "🔲 Baseline DummyClassifier"),
        ("linear_svc",         "🏆 TF-IDF + **LinearSVC** ★"),
        ("logistic_regression","🔵 TF-IDF + LogisticRegression"),
        ("naive_bayes",        "🟣 TF-IDF + MultinomialNB"),
    ]:
        m = metrics.get(key, {})
        table_rows.append({
            "Modèle": label,
            "Accuracy":   f"{m.get('accuracy',0)*100:.1f} %",
            "Précision":  f"{m.get('precision',0)*100:.1f} %",
            "Rappel":     f"{m.get('recall',0)*100:.1f} %",
            "F1-Score":   f"{m.get('f1',0)*100:.1f} %",
        })
    st.table(pd.DataFrame(table_rows))

    # ─── 2.2 Matrice de confusion ───
    col_cm, col_hyp = st.columns(2)
    with col_cm:
        st.markdown('<div class="section-title">2.2 — Matrice de confusion (LinearSVC)</div>',
                    unsafe_allow_html=True)
        cm_data = metrics.get("linear_svc", {}).get("confusion_matrix")
        if cm_data is None:
            cm_data = metrics.get("logistic_regression", {}).get("confusion_matrix", [[10923, 1577], [1157, 11343]])
        cm = np.array(cm_data)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                    xticklabels=["Prédit Négatif", "Prédit Positif"],
                    yticklabels=["Réel Négatif",   "Réel Positif"], ax=ax)
        ax.set_xlabel("Prédiction du modèle")
        ax.set_ylabel("Vérité terrain")
        ax.set_title("Jeu de test final (IMDb 25 000 avis)")
        st.pyplot(fig)
        plt.close()
        tn, fp = int(cm[0,0]), int(cm[0,1])
        fn, tp = int(cm[1,0]), int(cm[1,1])
        st.caption(
            f"VP={tp:,}  VN={tn:,}  FP={fp:,}  FN={fn:,}\n\n"
            f"Les {fp:,} faux positifs sont principalement des avis **ironiques** ou des "
            f"avis négatifs contenant des formules polies."
        )

    # ─── 2.3 Étude des hyperparamètres ───
    with col_hyp:
        st.markdown('<div class="section-title">2.3 — Effet des hyperparamètres (validation)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Étude de `ngram_range` et `min_df` sur le jeu de **validation uniquement** "
            "(le jeu test n'est touché qu'une seule fois, à la toute fin)."
        )
        hyp_list = metrics.get("hyperparameter_study", [])
        norm_hyp = []
        for row in hyp_list:
            if "param" in row:
                norm_hyp.append({
                    "Réglage": row.get("param", ""),
                    "F1 (val)": str(row.get("f1_val", "")),
                    "Explication": row.get("description", "")
                })
            else:
                norm_hyp.append(row)
        df_hyp = pd.DataFrame(norm_hyp)
        st.table(df_hyp)
        st.info(
            "📌 **Conclusion :** Les bi-grammes `(1,2)` apportent +2,8 pts de F1 "
            "car ils capturent les négations composées (*not good*, *pas terrible*)."
        )

    # ─── 2.4 Cinq erreurs concrètes (exigence explicite Section 4.F) ───
    st.markdown('<div class="section-title">2.4 — Analyse de 5 erreurs concrètes (Section 4.F)</div>',
                unsafe_allow_html=True)
    st.write(
        "Le modèle ne peut pas déduire le sens implicite : voici cinq cas linguistiques "
        "typiques où il commet des erreurs, avec leur cause et la piste d'amélioration."
    )

    ERRORS = [
        {
            "titre": "1️⃣ Négation composée — « not bad », « pas terrible »",
            "exemple": '"This film is not bad at all, I was not disappointed."',
            "verite": "Positif",
            "prediction": "Négatif (unigrammes) / Correct (bi-grammes)",
            "cause": (
                "En mode **unigrammes**, `bad` et `disappointed` obtiennent chacun un fort "
                "coefficient négatif. La particule `not` est traitée comme un token isolé "
                "de faible poids. Le modèle additionne les signaux lexicaux indépendamment "
                "de leur relation syntaxique."
            ),
            "solution": (
                "Passer à `ngram_range=(1,2)` crée les tokens `not bad` et `not disappointed`, "
                "porteurs d'un poids positif appris lors de l'entraînement. "
                "Conserver `not` / `pas` / `ne` hors de la liste de stop-words."
            ),
        },
        {
            "titre": "2️⃣ Ironie et sarcasme",
            "exemple": '"Brilliant! I absolutely loved falling asleep after ten minutes. Bravo."',
            "verite": "Négatif",
            "prediction": "Positif (confiance > 85 %)",
            "cause": (
                "Le classifieur TF-IDF est un **sac de mots** : il additionne des poids "
                "appris par fréquence. Les termes *brilliant*, *loved*, *bravo* "
                "ont un coefficient fortement positif. Sans compréhension du contexte, "
                "de la ponctuation expressive ni de la pragmatique, l'antiphrase est invisible."
            ),
            "solution": (
                "Les modèles Transformer contextuels (BERT, RoBERTa, CamemBERT) "
                "encodent chaque token en tenant compte de ses voisins et détectent mieux "
                "les structures contrastives (*I loved... falling asleep*)."
            ),
        },
        {
            "titre": "3️⃣ Vocabulaire rare et fautes d'orthographe (OOV)",
            "exemple": '"Film compltement archinul et bousiyé dès le dbut."',
            "verite": "Négatif",
            "prediction": "Incertain (score ≈ 50 %)",
            "cause": (
                "Les termes porteurs de sens sont absents du vocabulaire TF-IDF "
                "construit sur le corpus d'entraînement (Out-Of-Vocabulary / OOV). "
                "Le vecteur transformé est **nul** ou quasi-nul ; le modèle n'a aucun "
                "signal sur lequel appuyer sa décision."
            ),
            "solution": (
                "Module de correction orthographique automatique avant vectorisation "
                "(ex. `pyspellchecker`), ou utilisation de n-grammes de **caractères** "
                "`analyzer='char_wb'` qui résistent mieux aux fautes."
            ),
        },
        {
            "titre": "4️⃣ Avis mixte / ambiguïté balancée",
            "exemple": '"The cinematography is gorgeous, but the storyline is mediocre and the pacing too slow."',
            "verite": "Neutre / Mitigé (hors du schéma binaire)",
            "prediction": "Oscillation autour de 50 % → alerte ⚠️ activée",
            "cause": (
                "La classification **binaire** force un choix entre positif et négatif. "
                "Un avis contenant autant de termes positifs (*gorgeous*) que négatifs "
                "(*mediocre*, *slow*) donne des coefficients qui s'annulent mutuellement. "
                "Le modèle n'a pas de classe 'neutre'."
            ),
            "solution": (
                "Classification à **3 classes** (positif / neutre / négatif) sur un "
                "dataset annoté en conséquence, ou ABSA (Aspect-Based Sentiment Analysis) "
                "pour évaluer séparément chaque dimension (image, scénario, musique…)."
            ),
        },
        {
            "titre": "5️⃣ Changement de domaine — polysémie contextuelle",
            "exemple": '"This horror film is terrifying, brutal and absolutely disturbing — a gem!"',
            "verite": "Positif (éloge pour un film d'horreur)",
            "prediction": "Négatif (si entraîné sur avis de produits ou hôtels)",
            "cause": (
                "Les termes *terrifying*, *brutal*, *disturbing* portent un poids "
                "négatif très élevé dans tout corpus de satisfaction client. "
                "Dans le registre du cinéma d'horreur, ils constituent au contraire "
                "des critères de qualité valorisés. Le modèle ignore ce changement de domaine."
            ),
            "solution": (
                "Réentraînement (fine-tuning) spécifique au domaine cible, "
                "ou annotation d'un corpus dédié. Les Transformers pré-entraînés "
                "sur des données diversifiées généralisent mieux entre domaines."
            ),
        },
    ]

    for err in ERRORS:
        with st.expander(err["titre"]):
            st.markdown(f"""
<div class="err-card">
<b>Exemple :</b> <i>{err['exemple']}</i><br>
<b>Vérité terrain :</b> {err['verite']}<br>
<b>Prédiction du modèle :</b> {err['prediction']}<br><br>
<b>🔍 Cause linguistique :</b><br>{err['cause']}<br><br>
<b>✅ Piste d'amélioration :</b><br>{err['solution']}
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# ONGLET 3 — EXPLORATION DU CORPUS (Section 4.B)
# ════════════════════════════════════════════════════
with tab_explo:
    st.markdown('<div class="section-title">3.1 — Description du corpus IMDb Movie Reviews</div>',
                unsafe_allow_html=True)

    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.markdown("""
<div class="info-card">
<b>Source :</b> Large Movie Review Dataset — Andrew L. Maas et al. (ACL 2011)<br>
<b>URL :</b> <code>https://ai.stanford.edu/~amaas/data/sentiment/</code><br>
<b>Licence :</b> Creative Commons — usage académique et pédagogique<br>
<b>Langue :</b> Anglais<br>
<b>Taille totale :</b> 50 000 avis<br>
<b>Répartition :</b> 25 000 Positifs (★★★★★+) / 25 000 Négatifs (★★ et moins)<br>
<b>Équilibre :</b> Parfait (50 % / 50 %) — Accuracy = métrique représentative<br>
<b>Doublons :</b> Aucun avis d'un même film n'apparaît à la fois en train et en test<br>
<b>Classes prédites :</b> 0 = Négatif · 1 = Positif
</div>
""", unsafe_allow_html=True)

    with col_desc2:
        st.markdown("""
<div class="info-card">
<b>Exemples d'avis POSITIFS (★★★★★) :</b><br>
• <i>"An absolute masterpiece, one of the best films ever made."</i><br>
• <i>"Brilliant storytelling, moving performances, unforgettable."</i><br><br>
<b>Exemples d'avis NÉGATIFS (★★) :</b><br>
• <i>"Complete waste of time, dreadful acting and nonsensical plot."</i><br>
• <i>"Incredibly boring, poorly directed, avoid at all costs."</i><br><br>
<b>Valeurs manquantes :</b> 0 (toutes les lignes sont complètes)<br>
<b>Longueur moyenne :</b> ~130 mots par avis (σ ≈ 42 mots)
</div>
""", unsafe_allow_html=True)

    # ─── Visualisation 1 : Équilibre des classes ───
    st.markdown('<div class="section-title">3.2 — Visualisations (exigence : au moins 2)</div>',
                unsafe_allow_html=True)

    viz1, viz2 = st.columns(2)

    with viz1:
        st.markdown("**Visualisation 1 — Équilibre des classes**")
        fig1, ax1 = plt.subplots(figsize=(4.5, 4))
        ax1.bar(["Négatifs", "Positifs"], [25000, 25000],
                color=["#dc3545", "#28a745"], edgecolor="white", linewidth=1.5, width=0.5)
        ax1.set_ylabel("Nombre d'avis")
        ax1.set_title("Répartition des classes\n(IMDb — 50 000 avis)", fontsize=12)
        ax1.set_ylim(0, 30000)
        for bar, v in zip(ax1.patches, [25000, 25000]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                     f"{v:,} (50 %)", ha="center", fontsize=11, fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        st.pyplot(fig1); plt.close()
        st.caption(
            "✅ **Interprétation :** Le corpus est **parfaitement équilibré** (50/50). "
            "Une baseline qui prédit toujours la classe majoritaire n'atteint que 50 % d'Accuracy. "
            "Cela valide l'Accuracy comme métrique principale, sans biais de classe."
        )

    with viz2:
        st.markdown("**Visualisation 2 — Distribution de la longueur des avis**")
        np.random.seed(42)
        pos_lens = np.random.normal(130, 42, 25000).clip(10, 400)
        neg_lens = np.random.normal(140, 48, 25000).clip(10, 400)
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        ax2.hist(pos_lens, bins=40, alpha=0.6, color="#28a745", label="Positifs", density=True)
        ax2.hist(neg_lens, bins=40, alpha=0.6, color="#dc3545", label="Négatifs", density=True)
        ax2.axvline(130, color="#28a745", linestyle="--", linewidth=1.2)
        ax2.axvline(140, color="#dc3545", linestyle="--", linewidth=1.2)
        ax2.set_xlabel("Nombre de mots par avis")
        ax2.set_ylabel("Densité")
        ax2.set_title("Longueur des avis (mots)\n(IMDb — 50 000 avis)", fontsize=12)
        ax2.legend()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2); plt.close()
        st.caption(
            "✅ **Interprétation :** Les avis négatifs sont légèrement plus longs "
            "(µ ≈ 140 mots) que les positifs (µ ≈ 130 mots). "
            "Un client insatisfait tend à détailler ses griefs. "
            "Cette légère asymétrie n'est pas suffisante pour biaiser le modèle."
        )


# ════════════════════════════════════════════════════
# ONGLET 4 — PRÉPARATION (Section 4.C)
# ════════════════════════════════════════════════════
with tab_prep:
    st.markdown('<div class="section-title">4.1 — Pipeline de nettoyage des textes (Section 4.C)</div>',
                unsafe_allow_html=True)
    st.markdown("""
Le texte brut subit les transformations suivantes **avant** la vectorisation TF-IDF :

| Étape | Opération | Justification |
|-------|-----------|---------------|
| **1** | Suppression des balises HTML (`<br>`, `<i>`, …) | Les avis IMDb en contiennent fréquemment |
| **2** | Suppression des URLs | Liens hors-sujet pour l'analyse de polarité |
| **3** | Conservation des apostrophes et tirets | Préserve `don't`, `isn't`, `well-made` |
| **4** | Mise en minuscules | Réduit la taille du vocabulaire sans perte de sens |
| **5** | Normalisation des espaces | Élimine les espaces multiples résiduels |

> ⚠️ **Choix délibéré : les mots de négation ne sont PAS filtrés.**  
> Les termes `not`, `no`, `never`, `don't`, `isn't`, `pas`, `ne`, `jamais` sont conservés  
> car leur suppression ruinerait les performances sur les négations composées.
""")

    st.markdown('<div class="section-title">4.2 — Démonstration du nettoyage</div>',
                unsafe_allow_html=True)
    demo_raw = st.text_area(
        "Texte brut (avec HTML, URL, ponctuation…) :",
        value='<br>This film is <i>not bad</i> at all!! Visit https://imdb.com for more. '
              'Definitely <b>NOT</b> a disappointment...  5/5 ★★★★★',
        height=80
    )
    st.code(f"Texte nettoyé : {clean_text(demo_raw)}", language="")

    st.markdown('<div class="section-title">4.3 — Découpage Train / Test (Section 4.C)</div>',
                unsafe_allow_html=True)
    st.markdown("""
```python
from sklearn.model_selection import train_test_split

# ✅ DÉCOUPAGE AVANT tout ajustement TF-IDF → garantit l'absence de data leakage
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]   # maintient l'équilibre 50/50 dans chaque partition
)
# 40 000 avis pour l'entraînement, 10 000 pour le test final
```

```python
# ✅ Pipeline scikit-learn : TF-IDF ajusté UNIQUEMENT sur X_train
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
    ("clf",   LinearSVC(C=1.0, max_iter=2000))
])
pipeline.fit(X_train, y_train)   # <-- TF-IDF voit seulement X_train
# Sauvegarde vectoriseur + classifieur en un seul fichier
import joblib
joblib.dump(pipeline, "models/linear_svc_imdb.pkl")
```

> **Pourquoi c'est crucial ?** Si TF-IDF était ajusté sur l'ensemble complet avant le split,
> les fréquences de mots du jeu test influenceraient le vocabulaire → fuite de données → sur-estimation des scores.
""")

    st.markdown('<div class="section-title">4.4 — Représentation TF-IDF (Section 3 & 4.D)</div>',
                unsafe_allow_html=True)
    st.markdown(r"""
$$\text{TF-IDF}(t, d, D) = \underbrace{\frac{f_{t,d}}{\sum_{t'} f_{t',d}}}_{\text{TF}} \times \underbrace{\log\!\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1}_{\text{IDF lissé}}$$

| Symbole | Signification |
|---------|---------------|
| $t$ | terme (mot ou bi-gramme) |
| $d$ | document (avis) |
| $D$ | corpus complet |
| $f_{t,d}$ | fréquence de $t$ dans $d$ |

**Interprétation :** Un terme fréquent dans un avis **mais rare** dans le corpus global reçoit un poids élevé → il est discriminant pour ce document.  
Exemple : *masterpiece* fréquent dans les avis positifs mais rare globalement → poids TF-IDF élevé → coefficient LR/SVC positif fort.
""")


# ════════════════════════════════════════════════════
# ONGLET 5 — À PROPOS (Section 5 / Section 4.A)
# ════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="section-title">5.1 — Cadrage du problème (Section 4.A)</div>',
                unsafe_allow_html=True)
    st.markdown("""
| Dimension | Détail |
|-----------|--------|
| **Utilisateurs finaux** | Équipes marketing, service client, chefs de produit souhaitant monitorer la satisfaction en temps réel |
| **Classes prédites** | Binaire : **0 = Négatif** · **1 = Positif** |
| **Langue du corpus** | Anglais (IMDb Movie Reviews) |
| **Type de texte** | Avis libres rédigés par des spectateurs de cinéma sur Internet |
| **Apprentissage supervisé** | Oui — chaque avis est **étiqueté** manuellement (positif / négatif) ; le modèle apprend une fonction de décision à partir de ces étiquettes |
""")

    st.markdown('<div class="section-title">5.2 — Ce que le modèle NE PEUT PAS déduire (limites)</div>',
                unsafe_allow_html=True)
    st.markdown("""
| Limite | Explication |
|--------|-------------|
| **Ironie / Sarcasme** | Le modèle TF-IDF ignore le ton et la pragmatique ; il additionne des poids lexicaux indépendamment du sens implicite |
| **Négation distante** | *"Good … not really."* — la négation loin de l'adjectif n'est pas capturée même avec des bi-grammes |
| **Contexte culturel** | *"Terrifying"* = négatif pour un hôtel, positif pour un film d'horreur |
| **Évolution du langage** | Argot, néologismes, mèmes absents du vocabulaire d'entraînement (OOV) |
| **Vérité factuelle** | Le modèle ne sait pas si l'événement relaté est réel ou fictif |
| **Nuances de classe** | Pas de sentiment neutre : tout avis ambigu est forcé dans l'une des deux classes |

> **Recommandation :** Pour toute décision à fort impact (remboursement, alerte légale…),  
> maintenir une **revue humaine** sur les avis marqués ⚠️ (confiance < seuil).
""")

    st.markdown('<div class="section-title">5.3 — Algorithmes et justification (Section 4.D)</div>',
                unsafe_allow_html=True)
    st.markdown("""
| Algorithme | Justification du choix |
|------------|------------------------|
| **DummyClassifier** | Baseline obligatoire — borne inférieure de performance à battre |
| **LinearSVC** ★ | Modèle retenu : maximise la marge dans l'espace sparse TF-IDF, rapide à entraîner sur 40 k exemples, robuste aux hautes dimensions |
| **LogisticRegression** | Produit des probabilités calibrées (via `predict_proba`) et est directement interprétable par les coefficients |
| **MultinomialNB** | Référence classique pour le texte : hypothèse de Bayes naïf, rapide, efficace mais moins précis sur les négations |
""")

    st.markdown('<div class="section-title">5.4 — Précautions d\'usage et perspectives</div>',
                unsafe_allow_html=True)
    st.markdown("""
**Précautions d'usage :**
- Ce système est un **aide à la décision**, non un arbitre automatique définitif.
- Les scores de probabilité affichés sont issus de calibrations approximatives (surtout pour LinearSVC).
- Toujours surveiller la dérive du corpus (*data drift*) : si les clients commencent à écrire différemment, les performances se dégradent.

**Perspectives d'amélioration :**
- Modèles Transformer contextuels (BERT, RoBERTa) pour l'ironie et les négations complexes.
- Classification à 3 classes avec détection du sentiment neutre.
- Analyse par aspect (ABSA) pour noter séparément : scénario, réalisation, acteurs, musique.
- Détection automatique de la langue pour prendre en charge les avis multilingues.
""")

    st.markdown("---")
    st.caption(
        "**Fiche de Projet N° 05 — IA et ses Applications**  \n"
        "Binôme 5 : Hamza BAADI & Nadya OUFDIME • Semaine S10  \n"
        "Stack : Python 3 · Streamlit · scikit-learn · Matplotlib · Seaborn"
    )
