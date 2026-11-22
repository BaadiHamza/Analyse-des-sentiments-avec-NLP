#!/bin/bash
set -e

# Se positionner dans le répertoire du projet
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "🚀 Mini-Projet NLP - Analyse de Sentiments (Binôme 5)"
echo "   Hamza BAADI & Nadya OUFDIME"
echo "=========================================================="
echo ""

echo "📦 [1/2] Installation des dépendances requises (Streamlit, Scikit-Learn, Pandas)..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt

echo ""
echo "🌟 [2/2] Démarrage de l'application Streamlit..."
streamlit run app.py
