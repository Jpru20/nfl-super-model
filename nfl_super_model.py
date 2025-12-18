import joblib
import pandas as pd
import numpy as np
import requests
import injury_manager
import nfl_data_py as nfl
import xgboost as xgb
import os
import io
from datetime import datetime, timedelta
import pytz

# ==========================================
# 1. CONFIGURATION
# ==========================================
ODDS_API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae"
CURRENT_SEASON = 2025
TRENCH_MODEL_FILE = 'nfl_model_trench.json'

def get_live_odds():
    print("   [1/5] Fetching Live NFL Odds (Thurs-Mon Window)...")
    url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
    
    params = {
        'api_key': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'spreads,totals',
        'oddsFormat': 'american',
        'bookmakers': 'betmgm,draftkings'
    }
    
    try:
        res = requests.get(url, params=params).json()
        matchups = []
        
        now = datetime.now(pytz.utc)
        max_future = now + timedelta(days=6, hours=12) 
        
        team_map = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        
        for game in res:
            commence_str = game.get('commence_time')
            if not commence_str: continue
            
            game_dt = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
            
            if game_dt > max_future: continue 
            if game_dt < (now - timedelta(hours=4)): continue

            h = team_map.get(game['home_team'])
            a = team_map.get(game['away_team'])
            if not h or not a: continue
            
            spread = 0.0
            total = 45.0 
            book_used = "Unk"
            
            if game['bookmakers']:
                target_books = ['betmgm', 'draftkings']
                available_books = sorted(game['bookmakers'], key=lambda x: target_books.index(x['key']) if x['key'] in target_books else 99)
                
                for book in available_books:
                    current_spread = None
                    current_total = None
                    
                    for m in book['markets']:
                        if m['key'] == 'spreads':
                            for o in m['outcomes']:
                                if o['name'] == game['home_team']: current_spread = o['point']
                        if m['key'] == 'totals':
                            current_total = m['outcomes'][0]['point']
                    
                    if current_spread is not None and current_total is not None:
                        spread = current_spread
                        total = current_total
                        book_used = book['title']
                        break
            
            local = game_dt.astimezone(pytz.timezone('US/Eastern'))
            is_thurs = 1 if local.weekday() == 3 else 0
            
            matchups.append({
                'home': h, 'away': a, 
                'spread': float(spread), 'total': float(total),
                'book': book_used, 'time': local.hour, 'thurs': is_thurs
            })
            
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
    print("   [2/5] Loading All Models (Standard + Trench)...")
    try:
        model_s = joblib.load('nfl_model_blind.pkl')
        model_t = joblib.load('nfl_model_total.pkl')
        elo = joblib.load('current_elo_state_blind.pkl')
        ctx = pd.read_csv('team_context_blind.csv')
        units = pd.read_csv('unit_stats_blind.csv')
        feats = joblib.load('model_features_blind.pkl')
        vals = pd.read_csv('player_values_blind.csv')
        
        trench_model = None
        if os.path.exists(TRENCH_MODEL_FILE):
            trench_model = xgb.XGBClassifier()
            trench_model.load_model(TRENCH_MODEL_FILE)
        else:
            print("   ⚠️ Trench Model not found. Running in Standard Mode.")

        return model_s, model_t, elo, vals, ctx, units, feats, trench_model
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return None

def load_monte_carlo_data():
    print("   [3/5] Loading PBP for Monte Carlo...")
    try:
        pbp = nfl.import_pbp_data([CURRENT_SEASON-1, CURRENT_SEASON])
        return pbp
    except:
        return pd.DataFrame()

# ==========================================
# 2. ENGINES
# ==========================================
def get_drive_stats(pbp):
    if pbp.empty: return pd.DataFrame()
    real_drives = pbp[pbp['play_type'].isin(['pass','run','field_goal','punt'])]
    drives = real_drives.groupby(['game_id', 'fixed_drive', 'posteam', 'defteam'])['fixed_drive_result'].first().reset_index()
    
    def get_points(res):
        if res == 'Touchdown': return 7
        if res == 'Field goal': return 3
        return 0
    
    drives['points'] = drives['fixed_drive_result'].apply(get_points)
    
    off = drives.groupby('posteam').agg(
        off_ppd=('points', 'mean'), off_std=('points', 'std'),
        drives=('fixed_drive', 'count'), games=('game_id', 'nunique')
    ).reset_index()
    off['pace'] = off['drives'] / off['games']
    
    def_ = drives.groupby('defteam').agg(def_ppd=('points', 'mean'), def_std=('points', 'std')).reset_index()
    
    return off.merge(def_, left_on='posteam', right_on='defteam').set_index('posteam')

def run_simulation(home, away, stats, h_loss, a_loss, n_sims=2000):
    if home not in stats.index or away not in stats.index: return 0, 0
    h = stats.loc[home]; a = stats.loc[away]
    
    h_ppd_adj = h['off_ppd'] - (h_loss / 11.0)
    a_ppd_adj = a['off_ppd'] - (a_loss / 11.0)
    
    h_ppd = (h_ppd_adj + a['def_ppd']) / 2 + 0.15
    a_ppd = (a_ppd_adj + h['def_ppd']) / 2
    pace = (h['pace'] + a['pace']) / 2
    
    h_scores = np.random.normal(h_ppd * pace, h['off_std'] * np.sqrt(pace), n_sims)
    a_scores = np.random.normal(a_ppd * pace, a['off_std'] * np.sqrt(pace), n_sims)
    
    margin = np.mean(h_scores - a_scores)
    total = np.mean(h_scores + a_scores)
    
    return margin, total

def get_trench_probability(home, away, elo_diff, rest_diff, trench_model):
    if not trench_model: return 0.5, 0, 0
    try:
        ol_df = pd.read_csv('data/ol_value_ratings.csv')
        dl_df = pd.read_csv('data/dl_value_ratings.csv')
        map_fix = {'ARZ': 'ARI', 'WSH': 'WAS', 'HST': 'HOU', 'BLT': 'BAL', 'CLV': 'CLE', 'SL': 'LA'}
        if 'team' in ol_df.columns: ol_df['team'] = ol_df['team'].replace(map_fix)
        if 'team' in dl_df.columns: dl_df['team'] = dl_df['team'].replace(map_fix)
        
        h_ol_df = ol_df[ol_df['team'] == home]
        v_ol_df = ol_df[ol_df['team'] == away]
        h_dl_df = dl_df[dl_df['team'] == home]
        v_dl_df = dl_df[dl_df['team'] == away]
        
        if h_ol_df.empty or v_ol_df.empty or h_dl_df.empty or v_dl_df.empty: return 0.5, 0, 0

        h_ol = h_ol_df.nlargest(5, 'OL_Value_EPA')['OL_Value_EPA'].sum()
        v_ol = v_ol_df.nlargest(5, 'OL_Value_EPA')['OL_Value_EPA'].sum()
        h_dl = h_dl_df.nlargest(4, 'pressure_generated')['pressure_generated'].sum()
        v_dl = v_dl_df.nlargest(4, 'pressure_generated')['pressure_generated'].sum()
        
        mismatch_1 = h_ol - v_dl
        mismatch_2 = v_ol - h_dl
        
        feats = pd.DataFrame([[elo_diff, rest_diff, 2.0, h_ol, v_ol, h_dl, v_dl, mismatch_1, mismatch_2]], 
                             columns=['elo_diff', 'rest_diff', 'hfa_mod', 'home_ol_rating', 'visitor_ol_rating', 'home_dl_rating', 'visitor_dl_rating', 'mismatch_home_ol_vs_vis_dl', 'mismatch_vis_ol_vs_home_dl'])
        
        return trench_model.predict_proba(feats)[0][1], mismatch_1, mismatch_2
    except: return 0.5, 0, 0

# ==========================================
# 3. ANALYSIS & OUTPUT
# ==========================================
def explain_matchup_super(row, home, away, h_loss, a_loss, impact, pred_total, vegas_total, trench_prob, tm_1, tm_2):
    reasons = []
    if trench_prob > 0.60: reasons.append(f"🛡️ {home} Trench")
    elif trench_prob < 0.40: reasons.append(f"🛡️ {away} Trench")
    if tm_1 < -0.25: reasons.append(f"🚨 {home} OL")
    
    pass_edge = row['pass_mismatch'].values[0]
    if pass_edge > 0.15: reasons.append(f"{home} Air")
    elif pass_edge < -0.15: reasons.append(f"{away} Air")
    
    if row['body_clock_disadvantage'].values[0] == 1: reasons.append(f"⚠️ Travel")
    if abs(impact) > 1.5: reasons.append(f"Inj {abs(impact):.1f}")
        
    total_diff = pred_total - vegas_total
    if total_diff > 3.5: reasons.append(f"High Tot")
    elif total_diff < -3.5: reasons.append(f"Low Tot")
    
    if not reasons: return "-"
    return "|".join(reasons)

def run_super_consensus():
    print("\nInitializing SUPER CONSENSUS MODEL (Standard + Trench)...")
    brains = load_super_brains()
    if not brains: return
    model_s, model_t, elo, vals, ctx, units, feats, trench_model = brains
    
    matchups = get_live_odds()
    if not matchups: return
    live_injuries = get_live_injuries()
    pbp = load_monte_carlo_data()
    drive_stats = get_drive_stats(pbp)
    units = units[units['season'] == CURRENT_SEASON]

    def get_lost_epa(player_dict):
        if not player_dict: return 0.0
        total = 0.0
        weights = {'OUT': 1.0, 'DOUBTFUL': 0.75, 'QUESTIONABLE': 0.5}
        for p, status in player_dict.items():
            matches = vals[vals['full_name'].astype(str).str.contains(p.split()[-1], case=False, na=False)]
            w = weights.get(status, 0.5)
            if not matches.empty: total += matches['value_per_game'].max() * w
            else: total += 0.5 * w
        return total

    print("\n" + "="*175)
    # RESTORED: All columns including XGB, SIM, and TOT PICK
    print(f"{'MATCHUP':<12} | {'XGB':<6} | {'SIM':<6} | {'TRNCH':<5} | {'CONSENSUS':<10} | {'VEGAS':<6} | {'PICK':<12} | {'EDGE':<4} | {'AGREE':<6} | {'TOT PICK':<9} | {'ANALYSIS'}")
    print("="*175)
    
    for m in matchups:
        home, away, spread, total_line, book = m['home'], m['away'], m['spread'], m['total'], m.get('book', 'Unk')
        time, thurs = m['time'], m['thurs']
        
        h_elo = elo.get(home, 1500); a_elo = elo.get(away, 1500)
        try:
            h_u = units[units['team'] == home].iloc[0]
            a_u = units[units['team'] == away].iloc[0]
            pass_mis = h_u['off_pass_epa'] - a_u['def_pass_epa']
            run_mis = h_u['off_run_epa'] - a_u['def_run_epa']
            st_mis = h_u['special_teams_epa'] - a_u['special_teams_epa']
            edsr_mis = h_u['edsr'] - a_u['edsr']
        except: pass_mis, run_mis, st_mis, edsr_mis = 0,0,0,0

        try: h_form = ctx[ctx['team']==home]['form_rating'].values[0]
        except: h_form = 0
        try: a_form = ctx[ctx['team']==away]['form_rating'].values[0]
        except: a_form = 0
        
        west = ['SEA','SF','LAC','LA','LV','ARI','DEN']
        bc = 1 if (away in west and time==13) else 0
        
        row = pd.DataFrame([{
            'elo_diff': h_elo - a_elo, 'home_elo': h_elo, 'away_elo': a_elo,
            'coaching_mismatch': 0, 'h_proe': 0, 'a_proe': 0,
            'injury_value_diff': 0, 'is_thursday': thurs, 'div_game': 0,
            'body_clock_disadvantage': bc, 'rest_diff': 0,
            'home_form': h_form, 'away_form': a_form,
            'pass_mismatch': pass_mis, 'run_mismatch': run_mis,
            'st_mismatch': st_mis, 'edsr_mismatch': edsr_mis, 'luck_mismatch': 0
        }])
        
        h_loss = get_lost_epa(live_injuries.get(home, {}))
        a_loss = get_lost_epa(live_injuries.get(away, {}))
        net_impact = a_loss - h_loss
        row['injury_value_diff'] = net_impact
        
        # --- FIXED: APPLY DAMPING FACTOR (0.3) ---
        # XGB Prediction
        xgb_raw = model_s.predict(row[feats])[0] + net_impact
        # Restore Total Logic (XGB + Sim) with Dampening
        xgb_total = model_t.predict(row[feats])[0] - ((h_loss + a_loss) * 0.3)
        
        # Sim Prediction with Dampening
        sim_raw, sim_total = run_simulation(home, away, drive_stats, h_loss * 0.3, a_loss * 0.3)
        
        # Trench
        trench_prob, tm_1, tm_2 = get_trench_probability(home, away, h_elo - a_elo, 0, trench_model)
        trench_points = max(min((trench_prob - 0.5) * 10, 4.0), -4.0)
        
        # Consensus Spread (Super Consensus)
        con_raw = (xgb_raw * 0.4) + (sim_raw * 0.4) + ((spread * -1 + trench_points) * 0.2)
        
        # Consensus Total
        con_total = (xgb_total + sim_total) / 2
        
        diff = con_raw - (spread * -1)
        edge = abs(diff)
        
        if diff > 0: pick = f"{home} {spread}"
        else: pick = f"{away} +{abs(spread)}" if spread < 0 else f"{away} {spread}"
        
        # Total Logic
        diff_tot = con_total - total_line
        pick_tot = f"Ov {total_line}" if diff_tot > 0 else f"Un {total_line}"
        
        xgb_side = xgb_raw > (spread * -1)
        sim_side = sim_raw > (spread * -1)
        trn_side = trench_prob > 0.52 if diff > 0 else trench_prob < 0.48
        
        agree = "✅"
        if xgb_side != sim_side: agree = "⚠️"
        if agree == "✅" and trn_side: agree = "🔥 ALL"
        
        def fmt(val): return f"{home} -{val:.1f}" if val > 0 else f"{away} -{abs(val):.1f}"
        
        trn_disp = f"{trench_prob:.0%}"
        analysis = explain_matchup_super(row, home, away, h_loss, a_loss, net_impact, con_total, total_line, trench_prob, tm_1, tm_2)
        
        # RESTORED OUTPUT FORMAT
        print(f"{away} @ {home:<5} | {fmt(xgb_raw):<6} | {fmt(sim_raw):<6} | {trn_disp:<5} | {fmt(con_raw):<10} | {spread:<6} | {pick:<12} | {edge:.1f}  | {agree:<6} | {pick_tot:<9} | {analysis}")

if __name__ == "__main__":
    run_super_consensus()
