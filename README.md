# Plateforme de prédiction Coupe du Monde — version Django

## Installation

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Puis ouvrir http://127.0.0.1:8000/

## Structure

- `wc_prediction/` : configuration du projet Django (settings, urls)
- `predictor/` : l'application
  - `ml.py` : chargement du modèle (`model_best.joblib`, un `LogisticRegression`
    scikit-learn), du `scaler`, et des dernières statistiques connues de
    85 équipes (`platform_teams.json`) ; construit les features et calcule
    les probabilités (identiques à celles du notebook et du widget JS)
  - `forms.py` : formulaire de sélection des deux équipes
  - `views.py` : vue `predict_view` (GET affiche le formulaire avec une
    prédiction par défaut Portugal-Espagne, POST calcule la prédiction
    choisie)
  - `templates/predictor/predict.html` : interface (thème sombre, barres
    de probabilité)
  - `artifacts/` : le modèle et les données exportés depuis le notebook
    d'entraînement (`model_best.joblib`, `scaler.joblib`, `features.json`,
    `platform_teams.json`)

## Notes

- `DEBUG = True` et `ALLOWED_HOSTS = ['*']` sont réglés pour un usage
  local/démo. À ajuster avant tout déploiement en production (clé secrète,
  `DEBUG = False`, hôtes autorisés, serveur WSGI/ASGI de prod).
- Le modèle retenu (régression logistique) est le même que celui comparé
  aux 2 autres modèles (Random Forest, XGBoost) dans le notebook
  `Examen_ML_Prediction_CM.ipynb`.
