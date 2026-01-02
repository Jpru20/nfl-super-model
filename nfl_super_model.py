import joblib
import pandas as pd
import numpy as np
import requests
import injury_manager
import nfl_data_py as nfl
import xgboost as xgb
from datetime import datetime, timedelta
import pytz
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
ODDS_API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae"
CURRENT_SEASON = 2025
TRENCH_MODEL_FILE = "nfl_model_trench.json"

# --- NEW: WEEK 18 RESTING/TANKING LIST ---
# Add team abbreviations here (e.g. ['BAL', 'PHI']) to penalize them for resting starters
TEAMS_RESTING_STARTERS = ['LAC','NYG','LV','GB'] 

# ==========================================
# OPTIONAL SIM ENHANCEMENT DATA
# ==========================================
COACH_TENDENCIES_FILE = "data/team_coaching_tendencies.csv"
REDZONE_RATES_FILE = "data/team_redzone_rates.csv"
TURNOVER_RATES_FILE = "data/team_turnover_rates.csv"
BLOWOUT_RATES_FILE = "data/team_blowout_rates.csv"

def _load_optional_csv(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except:
        pass
    return pd.DataFrame()

# Defaults (League Averages)
_DEFAULT_RZ_TD_RATE = 0.58
_DEFAULT_RZ_REACH_RATE = 0.35
_DEFAULT_TOV_RATE = 0.11
_DEFAULT_COACH_AGG = 0.0
_DEFAULT_NOHUDDLE = 0.0
_DEFAULT_LEAD_PACE_MOD = 0.85
_DEFAULT_TRAIL_AGG_MOD = 1.05

# Load Data
coach_df = _load_optional_csv(COACH_TENDENCIES_FILE)
rz_df = _load_optional_csv(REDZONE_RATES_FILE)
tov_df = _load_optional_csv(TURNOVER_RATES_FILE)
bo_df = _load_optional_csv(BLOWOUT_RATES_FILE)

# Build Lookup Dicts
_COACH = {}
_RZ = {}
_TOV = {}
_BO = {}

if not coach_df.empty and "team" in coach_df.columns:
    for _, r in coach_df.iterrows():
        tm = str(r.get("team", "")).strip()
        if not tm: continue
        _COACH[tm] = {"agg": float(r.get("agg", _DEFAULT_COACH_AGG)), "no_huddle": float(r.get("no_huddle", _DEFAULT_NOHUDDLE))}

if not rz_df.empty and "team" in rz_df.columns:
    for _, r in rz_df.iterrows():
        tm = str(r.get("team", "")).strip()
        if not tm: continue
        _RZ[tm] = {"rz_reach": float(r.get("rz_reach_rate", _DEFAULT_RZ_REACH_RATE)), "rz_td": float(r.get("rz_td_rate", _DEFAULT_RZ_TD_RATE))}

if not tov_df.empty and "team" in tov_df.columns:
    for _, r in tov_df.iterrows():
        tm = str(r.get("team", "")).strip()
        if not tm: continue
        _TOV[tm] = {"tov_off": float(r.get("tov_off_rate", _DEFAULT_TOV_RATE)), "tov_def": float(r.get("tov_def_rate", _DEFAULT_TOV_RATE))}

if not bo_df.empty and "team" in bo_df.columns:
    for _, r in bo_df.iterrows():
        tm = str(r.get("team", "")).strip()
        if not tm: continue
        _BO[tm] = {
            "lead_pace_mod": float(r.get("lead_pace_mod", _DEFAULT_LEAD_PACE_MOD)),
            "trail_agg_mod": float(r.get("trail_agg_mod", _DEFAULT_TRAIL_AGG_MOD))
        }

# Helper Functions
def _coach(team: str): return _COACH.get(team, {"agg": _DEFAULT_COACH_AGG, "no_huddle": _DEFAULT_NOHUDDLE})
def _rz(team: str): return _RZ.get(team, {"rz_reach": _DEFAULT_RZ_REACH_RATE, "rz_td": _DEFAULT_RZ_TD_RATE})
def _tov(team: str): return _TOV.get(team, {"tov_off": _DEFAULT_TOV_RATE, "tov_def": _DEFAULT_TOV_RATE})
def _bo(team: str): return _BO.get(team, {"lead_pace_mod": _DEFAULT_LEAD_PACE_MOD, "trail_agg_mod": _DEFAULT_TRAIL_AGG_MOD})


def get_live_odds():
    print("   [1/5] Fetching Live NFL Odds (Thurs-Mon Window)...")
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"api_key": ODDS_API_KEY, "regions": "us", "markets": "spreads,totals", "oddsFormat": "american", "bookmakers": "betmgm,draftkings"}
    
    try:
        res = requests.get(url, params=params).json()
        matchups = []
        now = datetime.now(pytz.utc)
        max_future = now + timedelta(days=6, hours=12)
        
        team_map = {
            "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
            "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
            "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
            "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
            "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
            "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
            "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
            "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
        }

        for game in res:
            commence_str = game.get("commence_time")
            if not commence_str: continue
            game_dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
            if game_dt > max_future or game_dt < (now - timedelta(hours=4)): continue

            h, a = team_map.get(game["home_team"]), team_map.get(game["away_team"])
            if not h or not a: continue

            spread, total, book_used = 0.0, 45.0, "Unk"
            if game.get("bookmakers"):
                target_books = ["betmgm", "draftkings"]
                available_books = sorted(game["bookmakers"], key=lambda x: target_books.index(x["key"]) if x["key"] in target_books else 99)
                for book in available_books:
                    current_spread, current_total = None, None
                    for m in book.get("markets", []):
                        if m["key"] == "spreads":
                            for o in m.get("outcomes", []):
                                if o["name"] == game["home_team"]: current_spread = o["point"]
                        if m["key"] == "totals":
                            outs = m.get("outcomes", [])
                            if outs: current_total = outs[0]["point"]
                    if current_spread is not None and current_total is not None:
                        spread, total, book_used = current_spread, current_total, book.get("title", book.get("key", "Unk"))
                        break
            
            local = game_dt.astimezone(pytz.timezone("US/Eastern"))
            matchups.append({"home": h, "away": a, "spread": float(spread), "total": float(total), "book": book_used, "time": local.hour, "thurs": 1 if local.weekday() == 3 else 0})

        print(f"   -> Found {len(matchups)} games for the current week.")
        return matchups
    except Exception as e:
        print(f"   [Error] Odds API failed: {e}")
        return []

def get_live_injuries():
    try: return injury_manager.get_injury_data()
    except Exception as e: 
        print(f"   ⚠️ Injury Manager failed: {e}")
        return {}

def load_super_brains():
    print("   [2/5] Loading Models (Standard + Trench)...")
    try:
        model_s = joblib.load("nfl_model_blind.pkl")
        model_t = joblib.load("nfl_model_total.pkl")
        elo = joblib.load("current_elo_state_blind.pkl")
        vals = pd.read_csv("player_values_blind.csv")
        ctx = pd.read_csv("team_context_blind.csv")
        units = pd.read_csv("unit_stats_blind.csv")
        feats = joblib.load("model_features_blind.pkl")
        trench_model = None
        if os.path.exists(TRENCH_MODEL_FILE):
            trench_model = xgb.XGBClassifier()
            trench_model.load_model(TRENCH_MODEL_FILE)
        else:
            print("   ⚠️ Trench Model not found. Running without Trench adjustments.")
        return model_s, model_t, elo, vals, ctx, units, feats, trench_model
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return None

def load_monte_carlo_data():
    print("   [3/5] Loading PBP for Monte Carlo...")
    try: return nfl.import_pbp_data([CURRENT_SEASON - 1, CURRENT_SEASON])
    except: return pd.DataFrame()

def get_drive_stats(pbp):
    if pbp.empty: return pd.DataFrame()
    real_drives = pbp[pbp["play_type"].isin(["pass", "run", "field_goal", "punt"])]
    drives = real_drives.groupby(["game_id", "fixed_drive", "posteam", "defteam"])["fixed_drive_result"].first().reset_index()
    def get_points(res): return 7 if res == "Touchdown" else 3 if res == "Field goal" else 0
    drives["points"] = drives["fixed_drive_result"].apply(get_points)
    off = drives.groupby("posteam").agg(off_ppd=("points", "mean"), off_std=("points", "std"), drives=("fixed_drive", "count"), games=("game_id", "nunique")).reset_index()
    off["pace"] = off["drives"] / off["games"]
    def_ = drives.groupby("defteam").agg(def_ppd=("points", "mean"), def_std=("points", "std")).reset_index()
    return off.merge(def_, left_on="posteam", right_on="defteam").set_index("posteam")

def run_simulation(home, away, stats, h_loss, a_loss, n_sims=2000):
    if home not in stats.index or away not in stats.index: return 0, 0
    h, a = stats.loc[home], stats.loc[away]
    
    h_ppd = (h["off_ppd"] - h_loss / 11 + a["def_ppd"]) / 2 + 0.15
    a_ppd = (a["off_ppd"] - a_loss / 11 + h["def_ppd"]) / 2
    pace = (h["pace"] + a["pace"]) / 2

    # --- LOAD REAL METRICS ---
    hc, ac = _coach(home), _coach(away)
    hrz, arz = _rz(home), _rz(away)
    ht, at = _tov(home), _tov(away)
    hbo, abo = _bo(home), _bo(away)

    base_pace = max(6.0, pace) * (1.0 + 0.05 * (hc["no_huddle"] - ac["no_huddle"]))
    h_sd = float(h["off_std"]) if pd.notna(h["off_std"]) else 1.0
    a_sd = float(a["off_std"]) if pd.notna(a["off_std"]) else 1.0

    h_scores, a_scores = np.zeros(n_sims), np.zeros(n_sims)
    lam_drives = max(8.0, base_pace) / 2.0

    for i in range(n_sims):
        h_drives = max(6, int(np.random.poisson(lam_drives)))
        a_drives = max(6, int(np.random.poisson(lam_drives)))
        hs, as_ = 0, 0
        total_drives = h_drives + a_drives
        if total_drives <= 0: continue

        poss_home = (np.random.rand() < 0.5)
        h_taken, a_taken = 0, 0

        for d in range(total_drives):
            q_frac = d / max(1, total_drives - 1)
            in_q4 = q_frac >= 0.75
            margin = hs - as_
            h_ppd_mod, a_ppd_mod = h_ppd, a_ppd
            
            # --- REAL-DATA DRIVEN BLOWOUT LOGIC ---
            if in_q4:
                # Leading Big (Turtle Logic)
                if margin >= 17: h_ppd_mod *= hbo["lead_pace_mod"]
                elif margin <= -17: a_ppd_mod *= abo["lead_pace_mod"]
                
                # Trailing Big (Comeback/Fight Logic)
                elif margin <= -14: h_ppd_mod *= hbo["trail_agg_mod"]
                elif margin >= 14: a_ppd_mod *= abo["trail_agg_mod"]

            # General Aggression
            if margin < 0: h_ppd_mod *= (1.0 + 0.03 * max(-1.0, min(1.0, hc["agg"])))
            if margin > 0: a_ppd_mod *= (1.0 + 0.03 * max(-1.0, min(1.0, ac["agg"])))

            if poss_home and h_taken < h_drives:
                tov_p = 0.5 * (ht["tov_off"] + at["tov_def"])
                if margin <= -10: tov_p *= 1.10
                if margin >= 10: tov_p *= 0.95
                tov_p = max(0.02, min(0.25, tov_p))

                if np.random.rand() < tov_p:
                    as_ += int(np.random.choice([0, 3, 7], p=[0.25, 0.40, 0.35]))
                    h_taken += 1; poss_home = False; continue

                reach = max(0.10, min(0.65, hrz["rz_reach"] + 0.06 * (h_ppd_mod - 1.8)))
                if np.random.rand() < reach:
                    td_p = max(0.35, min(0.75, hrz["rz_td"] + 0.03 * (h_ppd_mod - 1.8)))
                    hs += 7 if np.random.rand() < td_p else 3
                else:
                    if np.random.rand() < max(0.02, min(0.20, 0.06 * h_ppd_mod)): hs += int(np.random.choice([3, 7], p=[0.60, 0.40]))

                h_taken += 1; poss_home = False

            elif (not poss_home) and a_taken < a_drives:
                tov_p = 0.5 * (at["tov_off"] + ht["tov_def"])
                if margin >= 10: tov_p *= 1.10
                if margin <= -10: tov_p *= 0.95
                tov_p = max(0.02, min(0.25, tov_p))

                if np.random.rand() < tov_p:
                    hs += int(np.random.choice([0, 3, 7], p=[0.25, 0.40, 0.35]))
                    a_taken += 1; poss_home = True; continue

                reach = max(0.10, min(0.65, arz["rz_reach"] + 0.06 * (a_ppd_mod - 1.8)))
                if np.random.rand() < reach:
                    td_p = max(0.35, min(0.75, arz["rz_td"] + 0.03 * (a_ppd_mod - 1.8)))
                    as_ += 7 if np.random.rand() < td_p else 3
                else:
                    if np.random.rand() < max(0.02, min(0.20, 0.06 * a_ppd_mod)): as_ += int(np.random.choice([3, 7], p=[0.60, 0.40]))

                a_taken += 1; poss_home = True

            elif h_taken >= h_drives and a_taken < a_drives: poss_home = False
            elif a_taken >= a_drives and h_taken < h_drives: poss_home = True
            else: break

        h_scores[i] = hs + np.random.normal(0, max(0.5, h_sd) * 0.25)
        a_scores[i] = as_ + np.random.normal(0, max(0.5, a_sd) * 0.25)

    return np.mean(h_scores - a_scores), np.mean(h_scores + a_scores)

def get_trench_probability(home, away, elo_diff, rest_diff, trench_model):
    if trench_model is None: return 0.5, 0.0, 0.0
    try:
        # --- FIX: Clean data before processing to handle "<5" ---
        def clean_val(x):
            if isinstance(x, str):
                x = x.replace("<", "").replace(">", "").strip()
            return float(x)

        ol_df = pd.read_csv("data/ol_value_ratings.csv")
        dl_df = pd.read_csv("data/dl_value_ratings.csv")
        
        # Apply cleaning
        if "OL_Value_EPA" in ol_df.columns: ol_df["OL_Value_EPA"] = ol_df["OL_Value_EPA"].apply(clean_val)
        if "pressure_generated" in dl_df.columns: dl_df["pressure_generated"] = dl_df["pressure_generated"].apply(clean_val)
        
        map_fix = {"ARZ": "ARI", "WSH": "WAS", "HST": "HOU", "BLT": "BAL", "CLV": "CLE", "SL": "LA"}
        if "team" in ol_df.columns: ol_df["team"] = ol_df["team"].replace(map_fix)
        if "team" in dl_df.columns: dl_df["team"] = dl_df["team"].replace(map_fix)

        h_ol_df = ol_df[ol_df["team"] == home]
        a_ol_df = ol_df[ol_df["team"] == away]
        h_dl_df = dl_df[dl_df["team"] == home]
        a_dl_df = dl_df[dl_df["team"] == away]

        if h_ol_df.empty or a_ol_df.empty or h_dl_df.empty or a_dl_df.empty: return 0.5, 0.0, 0.0

        h_ol = h_ol_df.nlargest(5, "OL_Value_EPA")["OL_Value_EPA"].sum()
        a_ol = a_ol_df.nlargest(5, "OL_Value_EPA")["OL_Value_EPA"].sum()
        h_dl = h_dl_df.nlargest(4, "pressure_generated")["pressure_generated"].sum()
        a_dl = a_dl_df.nlargest(4, "pressure_generated")["pressure_generated"].sum()
        mismatch_home, mismatch_away = h_ol - a_dl, a_ol - h_dl

        feats_df = pd.DataFrame([[elo_diff, rest_diff, 2.0, h_ol, a_ol, h_dl, a_dl, mismatch_home, mismatch_away]], 
                                columns=["elo_diff", "rest_diff", "hfa_mod", "home_ol_rating", "visitor_ol_rating", "home_dl_rating", "visitor_dl_rating", "mismatch_home_ol_vs_vis_dl", "mismatch_vis_ol_vs_home_dl"])
        return trench_model.predict_proba(feats_df)[0][1], mismatch_home, mismatch_away
    except Exception as e:
        print(f"   ⚠️ Trench model error ({home} vs {away}): {e}")
        return 0.5, 0.0, 0.0

def explain_matchup(row, home, away, h_lost, a_lost, impact, pred_total, vegas_total, resting_note):
    spread_reasons, total_reasons = [], []
    pass_edge = row["pass_mismatch"].values[0]
    if pass_edge > 0.15: spread_reasons.append(f"{home} Air Edge")
    elif pass_edge < -0.15: spread_reasons.append(f"{away} Air Edge")
    if row["body_clock_disadvantage"].values[0] == 1: spread_reasons.append(f"⚠️ {away} Travel Spot")
    if abs(impact) > 1.5: spread_reasons.append(f"Injuries {abs(impact):.1f}pts ({'Hurt' if impact < 0 else 'Boosted'} {home if impact < 0 else away})")
    if resting_note: spread_reasons.append(f"🛑 {resting_note}")
    
    total_diff = pred_total - vegas_total
    if total_diff > 3: total_reasons.append(f"Model sees Shootout (Ov {pred_total:.0f})")
    elif total_diff < -3: total_reasons.append(f"Model sees Defense (Un {pred_total:.0f})")
    
    full_text = []
    if spread_reasons: full_text.append(f"SPR: {', '.join(spread_reasons)}")
    if total_reasons: full_text.append(f"TOT: {', '.join(total_reasons)}")
    return " | ".join(full_text) if full_text else "Statistical Edge Only."

def get_divisions():
    return {
        "AFC_N": ["BAL", "CIN", "CLE", "PIT"], "AFC_S": ["HOU", "IND", "JAX", "TEN"], "AFC_E": ["BUF", "MIA", "NE", "NYJ"], "AFC_W": ["DEN", "KC", "LV", "LAC"],
        "NFC_N": ["CHI", "DET", "GB", "MIN"], "NFC_S": ["ATL", "CAR", "NO", "TB"], "NFC_E": ["DAL", "NYG", "PHI", "WAS"], "NFC_W": ["ARI", "LA", "SF", "SEA"]
    }

def run_consensus():
    print("\nInitializing NFL Consensus Model (Trench-Adjusted, Week 18 Logic)...")
    brains = load_super_brains()
    if not brains: return
    model_s, model_t, elo, vals, ctx, units, feats, trench_model = brains
    matchups = get_live_odds()
    if not matchups: return
    
    live_injuries = get_live_injuries()
    pbp = load_monte_carlo_data()
    drive_stats = get_drive_stats(pbp)
    units = units[units["season"] == CURRENT_SEASON]

    def get_lost_epa(player_dict):
        if not player_dict: return 0.0, []
        total, details = 0.0, []
        weights = {"OUT": 1.0, "DOUBTFUL": 0.75, "QUESTIONABLE": 0.5}
        for p, status in player_dict.items():
            matches = vals[vals["full_name"].astype(str).str.contains(p.split()[-1], case=False, na=False)]
            w = weights.get(status, 0.5)
            if not matches.empty:
                val = matches["value_per_game"].max(); impact = val * w
                total += impact; details.append(f"{p} ({status}): -{impact:.2f}")
            else:
                impact = 0.5 * w; total += impact; details.append(f"{p} ({status})*: -{impact:.2f}")
        return total, details

    print("\n" + "=" * 210)
    # --- FIX: Adjusted column headers for better alignment ---
    print(f"{'MATCHUP':<12} | {'XGB':<10} | {'SIM':<10} | {'TRNCH':<6} | {'CONSENSUS':<10} | {'VEGAS':<6} | {'BOOK':<8} | {'PICK':<12} | {'EDGE':<5} | {'CONF':<4} | {'AGREE':<6} | {'TOT PICK':<9} | {'ANALYSIS'}")
    print("=" * 210)
    injury_audit = []

    for m in matchups:
        home, away, spread, total_line, book = m["home"], m["away"], m["spread"], m["total"], m.get("book", "Unk")
        h_elo, a_elo = elo.get(home, 1500), elo.get(away, 1500)

        try:
            h_u, a_u = units[units["team"] == home].iloc[0], units[units["team"] == away].iloc[0]
            pass_mis = h_u["off_pass_epa"] - a_u["def_pass_epa"]
            run_mis = h_u["off_run_epa"] - a_u["def_run_epa"]
            st_mis = h_u["special_teams_epa"] - a_u["special_teams_epa"]
            edsr_mis = h_u["edsr"] - a_u["edsr"]
        except: pass_mis, run_mis, st_mis, edsr_mis = 0, 0, 0, 0

        bc = 1 if (away in ["SEA", "SF", "LAC", "LA", "LV", "ARI", "DEN"] and m["time"] == 13) else 0
        div_check = 1 if any(home in d and away in d for d in get_divisions().values()) else 0
        h_form = ctx[ctx["team"] == home]["form_rating"].values[0] if not ctx[ctx["team"] == home].empty else 0
        a_form = ctx[ctx["team"] == away]["form_rating"].values[0] if not ctx[ctx["team"] == away].empty else 0

        row = pd.DataFrame([{
            "elo_diff": h_elo - a_elo, "home_elo": h_elo, "away_elo": a_elo, "coaching_mismatch": 0, "h_proe": 0, "a_proe": 0, "injury_value_diff": 0,
            "is_thursday": m["thurs"], "div_game": div_check, "body_clock_disadvantage": bc, "rest_diff": 0, "home_form": h_form, "away_form": a_form,
            "pass_mismatch": pass_mis, "run_mismatch": run_mis, "st_mismatch": st_mis, "edsr_mismatch": edsr_mis, "luck_mismatch": 0,
        }])

        h_loss, h_dets = get_lost_epa(live_injuries.get(home, {}))
        a_loss, a_dets = get_lost_epa(live_injuries.get(away, {}))
        if h_dets: injury_audit.append(f"[{home}] Impact -{h_loss:.1f}: " + ", ".join(h_dets))
        if a_dets: injury_audit.append(f"[{away}] Impact -{a_loss:.1f}: " + ", ".join(a_dets))

        net_impact = a_loss - h_loss
        row["injury_value_diff"] = net_impact
        
        xgb_raw = model_s.predict(row[feats])[0] + net_impact
        xgb_total = model_t.predict(row[feats])[0] - ((h_loss + a_loss) * 0.3)
        sim_raw, sim_total = run_simulation(home, away, drive_stats, h_loss * 0.3, a_loss * 0.3)
        trench_prob, tm_1, tm_2 = get_trench_probability(home, away, h_elo - a_elo, 0, trench_model)
        trench_points = max(min((trench_prob - 0.5) * 10, 4.0), -4.0)

        con_raw = (xgb_raw * 0.4) + (sim_raw * 0.4) + (trench_points * 0.2)
        
        # --- NEW: TANKING / RESTING PENALTY ---
        tank_mod = 0.0
        resting_note = ""
        if home in TEAMS_RESTING_STARTERS:
            tank_mod -= 7.0
            resting_note = f"{home} Resting Starters"
        if away in TEAMS_RESTING_STARTERS:
            tank_mod += 7.0
            resting_note = f"{away} Resting Starters"
        con_raw += tank_mod
        # --------------------------------------

        con_total = (xgb_total + sim_total) / 2
        vegas_expect = spread * -1
        diff, edge = con_raw - vegas_expect, abs(con_raw - vegas_expect)
        
        two_agree = (xgb_raw > vegas_expect) == (sim_raw > vegas_expect)
        if edge >= 4.0 and two_agree: conf = "HIGH"
        elif edge >= 4.0: conf = "MED"
        elif edge >= 2.0 and two_agree: conf = "MED"
        else: conf = "LOW"

        pick = f"{home} {spread}" if diff > 0 else (f"{away} +{abs(spread)}" if spread < 0 else f"{away} {spread}")
        pick_tot = f"Ov {total_line}" if (con_total - total_line) > 0 else f"Un {total_line}"
        agree = "🔥 ALL" if (two_agree and ((trench_prob > 0.52) if diff > 0 else (trench_prob < 0.48))) else ("✅" if two_agree else "⚠️")

        def fmt_margin(val): return f"{home} -{val:.1f}" if val > 0 else f"{away} -{abs(val):.1f}"
        
        # --- FIX: Formatting adjustments for alignment ---
        matchup_str = f"{away} @ {home}"
        print(f"{matchup_str:<12} | {fmt_margin(xgb_raw):<10} | {fmt_margin(sim_raw):<10} | {trench_prob:<6.0%} | {fmt_margin(con_raw):<10} | {spread:<6} | {book.split()[0] if book else 'Unk':<8} | {pick:<12} | {edge:<5.1f} | {conf:<4} | {agree:<6} | {pick_tot:<9} | {explain_matchup(row, home, away, h_loss, a_loss, net_impact, con_total, total_line, resting_note)}")

    if injury_audit:
        print("\n" + "=" * 50 + "\nINJURY AUDIT (Impact > 0)\n" + "=" * 50)
        team_audits = {}
        for log in injury_audit:
            team_audits.setdefault(log.split("]")[0][1:], []).append(log.split(": ", 1)[1] if ": " in log else log)
        for team, logs in team_audits.items():
            print(f"\n--- {team} ---"); [print(f"   {l}") for l in logs]

if __name__ == "__main__":
    run_consensus()
