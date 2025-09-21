from __future__ import annotations

import json
import gzip
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Prédiction de match de football", layout="centered")

API_KEY = st.secrets.get("API_KEY", "")
API_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2025

# Model + artefacts (place them at repo root)
MODEL_CANDIDATES = ["xgb_model_v2.json.gz", "xgb_model_v2.json"]
FEATURES_PATH = "xgb_features_v2.json"
LABELMAP_PATH = "xgb_labelmap_v2.json"

LEAGUES = {
    "Premier League (Angleterre)": 39,
    "Ligue 1 (France)": 61,
    "La Liga (Espagne)": 140,
    "Serie A (Italie)": 135,
    "Bundesliga (Allemagne)": 78,
}


# ===============================
# UTILS
# ===============================
def _safe_get(d, path, default=None):
    cur = d if isinstance(d, dict) else {}
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace("%", "")
        return float(s)
    except Exception:
        return float(default)


def _rate_limit_sleep():
    time.sleep(0.25)


# ===============================
# LOAD ARTEFACTS
# ===============================
def _find_model_path() -> str:
    for p in MODEL_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        f"Modèle introuvable. Placez l'un de ces fichiers à la racine: {MODEL_CANDIDATES}"
    )


@st.cache_resource
def _load_booster(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            raw: bytes = f.read()
        booster.load_model(bytearray(raw))  # IMPORTANT: bytes
    else:
        booster.load_model(path)
    return booster


@st.cache_resource
def load_artifacts():
    model_path = _find_model_path()
    booster = _load_booster(model_path)

    # feature order
    if Path(FEATURES_PATH).exists():
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        feature_order = obj["feature_names"] if isinstance(obj, dict) and "feature_names" in obj else obj
    else:
        # try reading from booster (may be None depending on export)
        feature_order = booster.feature_names
        if not feature_order:
            raise FileNotFoundError(
                f"{FEATURES_PATH} manquant et le modèle ne contient pas les noms de colonnes.\n"
                "➡️ Uploadez xgb_features_v2.json"
            )

    # label map
    if Path(LABELMAP_PATH).exists():
        with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    else:
        label_map = {"away": 0, "draw": 1, "home": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    return booster, feature_order, label_map, inv_label_map, model_path


booster, FEATURE_ORDER, LABEL_MAP, INV_LABEL_MAP, MODEL_FILE = load_artifacts()


# ===============================
# API CALLS
# ===============================
@st.cache_data(show_spinner=False)
def get_upcoming_matches(league_id: int):
    url = f"{API_URL}/fixtures?league={league_id}&season={SEASON}&next=10"
    r = requests.get(url, headers=HEADERS, timeout=25)
    if not r.ok:
        return []
    return r.json().get("response", [])


@st.cache_data(show_spinner=False)
def get_name_to_id_mapping(league_id: int) -> Dict[str, int]:
    url = f"{API_URL}/teams?league={league_id}&season={SEASON}"
    r = requests.get(url, headers=HEADERS, timeout=25)
    data = r.json().get("response", []) if r.ok else []
    return {t["team"]["name"]: t["team"]["id"] for t in data}


@st.cache_data(show_spinner=False)
def get_team_statistics(team_id: int, league_id: int):
    url = f"{API_URL}/teams/statistics?team={team_id}&season={SEASON}&league={league_id}"
    r = requests.get(url, headers=HEADERS, timeout=25)
    return r.json().get("response", {}) if r.ok else {}


@st.cache_data(show_spinner=False)
def get_recent_fixtures(team_id: int, count: int = 5):
    url = f"{API_URL}/fixtures?team={team_id}&season={SEASON}&last={count}"
    r = requests.get(url, headers=HEADERS, timeout=25)
    return r.json().get("response", []) if r.ok else []


@st.cache_data(show_spinner=False)
def get_injuries(team_id: int):
    url = f"{API_URL}/injuries?team={team_id}&season={SEASON}"
    r = requests.get(url, headers=HEADERS, timeout=25)
    data = r.json().get("response", []) if r.ok else []
    return len(data)


# ===============================
# FEATURE HELPERS
# ===============================
def _form_wld_from_fixtures(fixtures: List[dict], team_id: int) -> Tuple[int, int, int]:
    w = d = l = 0
    for fx in fixtures:
        winner_home = _safe_get(fx, ["teams", "home", "winner"], False)
        winner_away = _safe_get(fx, ["teams", "away", "winner"], False)
        home_id = _safe_get(fx, ["teams", "home", "id"], None)
        away_id = _safe_get(fx, ["teams", "away", "id"], None)

        if winner_home is True and home_id == team_id:
            w += 1
        elif winner_away is True and away_id == team_id:
            w += 1
        elif winner_home is False and home_id == team_id:
            l += 1
        elif winner_away is False and away_id == team_id:
            l += 1
        else:
            d += 1
    return w, d, l


def _goals_scored_conceded(fixtures: List[dict], team_id: int) -> Tuple[int, int]:
    scored = conceded = 0
    for fx in fixtures:
        gh = _safe_get(fx, ["goals", "home"], 0)
        ga = _safe_get(fx, ["goals", "away"], 0)
        if _safe_get(fx, ["teams", "home", "id"], None) == team_id:
            scored += _to_float(gh); conceded += _to_float(ga)
        else:
            scored += _to_float(ga); conceded += _to_float(gh)
    return int(scored), int(conceded)


def _avg_stat_generic(stats: dict, field: str, side: str = "total") -> float:
    d = stats if isinstance(stats, dict) else {}
    if field == "shots_on_target":
        v = _safe_get(d, ["shots", "on", side], 0)  # adapté à la structure API courante
        return _to_float(v, 0.0)
    if field == "possession":
        v = _safe_get(d, ["ball", "possession", side], 0)  # ex "54%"
        val = _to_float(v, 0.0)
        return val / (100.0 if val > 1.0 else 1.0)
    if field == "corners":
        v = _safe_get(d, ["corners", "total", side], 0)
        return _to_float(v, 0.0)
    return 0.0


def _add_fixture_meta_columns(row: Dict[str, float], feature_order: List[str], match_raw: dict,
                              league_id: int, home_id: int, away_id: int):
    """Remplit les colonnes meta si elles sont dans FEATURE_ORDER."""
    fx = match_raw.get("fixture", {})
    lg = match_raw.get("league", {})
    t  = match_raw.get("teams", {})

    meta = {
        "fixture.id": _safe_get(fx, ["id"], 0),
        "fixture.timestamp": _safe_get(fx, ["timestamp"], 0),
        "fixture.periods.first": _safe_get(fx, ["periods", "first"], 0),
        "fixture.periods.second": _safe_get(fx, ["periods", "second"], 0),
        "fixture.venue.id": _safe_get(fx, ["venue", "id"], 0),
        "fixture.status.elapsed": _safe_get(fx, ["status", "elapsed"], 0),

        "league.id": _safe_get(lg, ["id"], league_id),
        "league.season": _safe_get(lg, ["season"], SEASON),

        "teams.home.id": _safe_get(t, ["home", "id"], home_id),
        "teams.away.id": _safe_get(t, ["away", "id"], away_id),
    }

    for k, v in meta.items():
        if k in row:
            row[k] = float(_to_float(v, 0.0))


def build_features_for_pair(
    home_name: str,
    away_name: str,
    league_id: int,
    feature_order: List[str],
    match_raw: dict | None = None,
) -> pd.DataFrame:

    name2id = get_name_to_id_mapping(league_id)
    if home_name not in name2id or away_name not in name2id:
        st.error("Équipe introuvable dans la ligue sélectionnée.")
        st.stop()

    home_id = name2id[home_name]
    away_id = name2id[away_name]

    fx_home = get_recent_fixtures(home_id, 5); _rate_limit_sleep()
    fx_away = get_recent_fixtures(away_id, 5); _rate_limit_sleep()

    h_w, _, _ = _form_wld_from_fixtures(fx_home, home_id)
    a_w, _, _ = _form_wld_from_fixtures(fx_away, away_id)

    h_scored_5, h_conceded_5 = _goals_scored_conceded(fx_home, home_id)
    a_scored_5, a_conceded_5 = _goals_scored_conceded(fx_away, away_id)

    home_absents = get_injuries(home_id); _rate_limit_sleep()
    away_absents = get_injuries(away_id); _rate_limit_sleep()

    home_stats = get_team_statistics(home_id, league_id); _rate_limit_sleep()
    away_stats = get_team_statistics(away_id, league_id)

    shots_home_last5 = _avg_stat_generic(home_stats, "shots_on_target")
    shots_away_last5 = _avg_stat_generic(away_stats, "shots_on_target")
    poss_home_last5  = _avg_stat_generic(home_stats, "possession")
    poss_away_last5  = _avg_stat_generic(away_stats, "possession")
    corners_home_last5 = _avg_stat_generic(home_stats, "corners")
    corners_away_last5 = _avg_stat_generic(away_stats, "corners")

    # goal_diff proxy (buts marqués moyens home vs away)
    gfor_home = _safe_get(home_stats, ["goals", "for", "average", "home"], 0)
    gfor_away = _safe_get(away_stats, ["goals", "for", "average", "away"], 0)
    goal_diff = _to_float(gfor_home, 0.0) - _to_float(gfor_away, 0.0)

    team_ids = list(name2id.values())
    home_enc = int(team_ids.index(home_id))
    away_enc = int(team_ids.index(away_id))

    base = {
        "home_team_enc": home_enc,
        "away_team_enc": away_enc,
        "home_advantage": 1,
        "goal_diff": float(goal_diff),

        "home_last5_form": float(h_w),
        "away_last5_form": float(a_w),

        "home_last5_goals_scored": float(h_scored_5),
        "away_last5_goals_scored": float(a_scored_5),
        "home_last5_goals_conceded": float(h_conceded_5),
        "away_last5_goals_conceded": float(a_conceded_5),

        "home_absents": float(home_absents),
        "away_absents": float(away_absents),

        "shots_on_target_home_last5": float(shots_home_last5),
        "shots_on_target_away_last5": float(shots_away_last5),
        "possession_home_last5": float(poss_home_last5),
        "possession_away_last5": float(poss_away_last5),
        "corners_home_last5": float(corners_home_last5),
        "corners_away_last5": float(corners_away_last5),
    }

    # Start with zeros for every expected feature
    row = {col: 0.0 for col in feature_order}
    # fill with our computed values
    for k, v in base.items():
        if k in row:
            row[k] = v

    # NEW: fill meta if present in training features
    if match_raw is not None:
        _add_fixture_meta_columns(row, feature_order, match_raw, league_id, home_id, away_id)

    X = pd.DataFrame([row], columns=feature_order)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X


# ===============================
# UI
# ===============================
st.title("⚽ Prédiction de match de football")

selected_league = st.selectbox("Choisis une ligue", list(LEAGUES.keys()))
LEAGUE_ID = LEAGUES[selected_league]

matches_raw = get_upcoming_matches(LEAGUE_ID)
if not matches_raw:
    st.warning("Aucun match à venir pour cette ligue.")
    st.stop()

options = []
for m in matches_raw:
    fixture = m["fixture"]; teams = m["teams"]
    label = f"{teams['home']['name']} vs {teams['away']['name']} ({fixture['date'][:10]})"
    options.append({
        "label": label,
        "home": teams["home"]["name"],
        "away": teams["away"]["name"],
        "fixture_id": fixture["id"],
        "date": fixture["date"][:10],
        "raw": m,  # 🔁 on garde tout l'objet pour les meta
    })

selected = st.selectbox("Choisis un match à venir", options, format_func=lambda x: x["label"])

st.caption(f"Modèle chargé : {MODEL_FILE}")
st.markdown("### Détails du match")
st.write(f"- 🏠 Domicile : **{selected['home']}**")
st.write(f"- ✈️ Extérieur : **{selected['away']}**")
st.write(f"- 🏆 Ligue : **{selected_league}**")
st.write(f"- 📅 Date : **{selected['date']}**")

# ===============================
# PREDICTION
# ===============================
if st.button("Prédire le résultat"):
    with st.spinner("Construction des features…"):
        X_match = build_features_for_pair(
            selected["home"], selected["away"], LEAGUE_ID, FEATURE_ORDER, match_raw=selected["raw"]
        )

    st.markdown("### Données utilisées pour la prédiction")
    st.dataframe(X_match)

    try:
        dmat = xgb.DMatrix(X_match, feature_names=FEATURE_ORDER)
        proba = booster.predict(dmat)[0]

        label_human = {"away": "Victoire extérieure", "draw": "Match nul", "home": "Victoire à domicile"}

        st.markdown("### Probabilités")
        st.write({
            "Victoire extérieure (away)": round(float(proba[LABEL_MAP.get("away", 0)]), 3),
            "Match nul (draw)": round(float(proba[LABEL_MAP.get("draw", 1)]), 3),
            "Victoire à domicile (home)": round(float(proba[LABEL_MAP.get("home", 2)]), 3),
        })

        pred_idx = int(np.argmax(proba))
        inv = {v: k for k, v in LABEL_MAP.items()}
        pred_label = inv.get(pred_idx, "home")
        confidence = float(proba[pred_idx])
        texte = label_human.get(pred_label, pred_label)

        if confidence >= 0.65:
            st.success(f"✅ Prédiction confiante : **{texte}** ({confidence:.2%})")
        else:
            st.warning(f"ℹ️ Prédiction : **{texte}** (confiance : {confidence:.2%})")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
