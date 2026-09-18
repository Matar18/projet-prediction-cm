"""
Chargement des artefacts ML (modèle + scaler + stats équipes) et calcul
de la prédiction. Isolé dans un module dédié pour que la vue reste simple.
"""
import json
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model_best.joblib")
    _patch_sklearn_compat(model)
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    with open(ARTIFACTS_DIR / "features.json") as f:
        meta = json.load(f)
    with open(ARTIFACTS_DIR / "platform_teams.json") as f:
        teams = json.load(f)
    return model, scaler, meta, teams


def _patch_sklearn_compat(model):
    """Le modèle a été exporté (pickle) avec une version de scikit-learn plus
    récente que celle installée en prod/dev. Certains attributs internes
    posés au fit() peuvent manquer une fois rechargés sur une version plus
    ancienne (ex: `multi_class`, retiré du LogisticRegression en 1.8),
    ce qui fait planter predict_proba(). On restaure les valeurs par défaut
    manquantes ici plutôt que d'exiger une version exacte de scikit-learn.
    """
    defaults = {"multi_class": "auto"}
    for attr, default in defaults.items():
        if not hasattr(model, attr):
            setattr(model, attr, default)


def team_names():
    _, _, _, teams = load_artifacts()
    return sorted(teams.keys())


def build_features(home_stats, away_stats, neutral, feature_list):
    h, a = home_stats, away_stats
    raw = {
        "rank_diff": a["rank"] - h["rank"],
        "points_diff": h["total_points"] - a["total_points"],
        "rank_trend_diff": h["rank_trend3"] - a["rank_trend3"],
        "form5_winrate_diff": h["form5_winrate"] - a["form5_winrate"],
        "form10_winrate_diff": h["form10_winrate"] - a["form10_winrate"],
        "goal_avg_diff": (h["form5_avg_gf"] - h["form5_avg_ga"]) - (a["form5_avg_gf"] - a["form5_avg_ga"]),
        "career_winrate_diff": h["career_winrate"] - a["career_winrate"],
        "experience_diff": h["career_matches"] - a["career_matches"],
        "home_rank": h["rank"], "away_rank": a["rank"],
        "home_total_points": h["total_points"], "away_total_points": a["total_points"],
        "home_form5_winrate": h["form5_winrate"], "away_form5_winrate": a["form5_winrate"],
        "is_neutral": int(neutral), "is_world_cup": 1, "is_qualifier": 0, "is_friendly": 0,
    }
    return np.array([[raw[f] for f in feature_list]])


def predict_match(home, away, neutral=True):
    """Retourne un dict avec les probabilités et les stats des 2 équipes."""
    model, scaler, meta, teams = load_artifacts()
    feature_list = meta["features"]
    best_name = meta["best_model"]

    h, a = teams[home], teams[away]
    X = build_features(h, a, neutral, feature_list)
    X_in = scaler.transform(X) if best_name == "Logistic Regression" else X

    proba = model.predict_proba(X_in)[0]  # ordre des classes : 0=away,1=draw,2=home
    return {
        "home": home, "away": away,
        "home_stats": h, "away_stats": a,
        "proba_home_win": round(float(proba[2]) * 100, 1),
        "proba_draw": round(float(proba[1]) * 100, 1),
        "proba_away_win": round(float(proba[0]) * 100, 1),
        "model_name": best_name,
    }
