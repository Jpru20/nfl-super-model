import nfl_data_py as nfl
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_YEARS = [2022, 2023, 2024, 2025]

def load_all_data(years):
    print(f"Loading NFL data...")
    
    # 1. PBP (Robust Brute Force Load)
    print("1. Play-by-Play...", end=" ")
    try:
        pbp = nfl.import_pbp_data(years)
        print("Success (Library).")
    except:
        print("Library failed. Attempting Direct GitHub Download...")
        dfs = []
        for y in years:
            try:
                url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.csv.gz"
                df = pd.read_csv(url, compression='gzip', low_memory=False)
                dfs.append(df)
            except: pass
        pbp = pd.concat(dfs) if dfs else pd.DataFrame()

    # 2. Schedule
    print("2. Schedules...", end=" ")
    try:
        schedule = nfl.import_schedules(years)
        print("Success.")
    except:
        try:
            url = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
            schedule = pd.read_csv(url)
            schedule = schedule[schedule['season'].isin(years)]
            print("Direct Download Success.")
        except: schedule = pd.DataFrame()

    # 3. Injuries & Depth (Safe Load)
    print("3. Context Files...", end=" ")
    try: injuries = nfl.import_injuries(years)
    except: injuries = pd.DataFrame(columns=['season','week','team','report_status','full_name'])
    
    try: depth_charts = nfl.import_depth_charts(years)
    except: depth_charts = pd.DataFrame()
    print("Done.")
        
    return pbp, schedule, injuries, depth_charts

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

def calculate_sharp_metrics(pbp_df):
    print("Calculating Sharp Metrics (EDSR, Special Teams, Luck)...")
    if pbp_df.empty: return pd.DataFrame(), pd.DataFrame()
    
    # 1. SPECIAL TEAMS EPA
    st_plays = pbp_df[pbp_df['play_type'].isin(['field_goal', 'punt', 'kickoff', 'extra_point'])]
    st_epa = st_plays.groupby(['season', 'posteam'])['epa'].mean().reset_index().rename(columns={'epa': 'special_teams_epa'})
    
    # 2. EARLY DOWN SUCCESS RATE (EDSR)
    early_downs = pbp_df[
        (pbp_df['down'].isin([1, 2])) & 
        (pbp_df['play_type'].isin(['pass', 'run']))
    ].copy()
    early_downs['success'] = np.where(early_downs['epa'] > 0, 1, 0)
    edsr = early_downs.groupby(['season', 'posteam'])['success'].mean().reset_index().rename(columns={'success': 'edsr'})
    
    # 3. TURNOVER DIFFERENTIAL
    turnovers = pbp_df.groupby(['season', 'week', 'posteam'])[['fumble_lost', 'interception']].sum().reset_index()
    turnovers['turnovers_committed'] = turnovers['fumble_lost'] + turnovers['interception']
    
    # Merge ST and EDSR
    sharp_stats = st_epa.merge(edsr, on=['season', 'posteam'])
    return sharp_stats, turnovers[['season', 'week', 'posteam', 'turnovers_committed']]

def calculate_advanced_context(schedule_df, turnovers_df):
    print("Calculating Travel, Neutral Form, and Rest...")
    df = schedule_df.copy()
    
    west_teams = ['SEA', 'SF', 'LAC', 'LA', 'LV', 'ARI', 'DEN']
    east_teams = ['NE', 'NYJ', 'NYG', 'BUF', 'PHI', 'WAS', 'BAL', 'PIT', 'CLE', 'CIN', 'MIA', 'TB', 'JAX', 'ATL', 'CAR', 'DET', 'IND']
    df['gametime'] = df['gametime'].astype(str)
    df['body_clock_disadvantage'] = np.where(
        (df['away_team'].isin(west_teams)) & (df['home_team'].isin(east_teams)) & (df['gametime'].str.startswith('13')), 1, 0)

    if 'rest_days' in df.columns:
        df['home_rest'] = df.groupby('home_team')['rest_days'].fillna(7)
        df['away_rest'] = df.groupby('away_team')['rest_days'].fillna(7)
    else:
        df['home_rest'] = 7; df['away_rest'] = 7
    df['rest_diff'] = df['home_rest'] - df['away_rest']

    HFA_ADJUSTMENT = 1.5
    df['home_margin'] = (df['home_score'] - df['away_score']) - HFA_ADJUSTMENT
    df['away_margin'] = (df['away_score'] - df['home_score']) + HFA_ADJUSTMENT
    
    if not turnovers_df.empty:
        df = df.merge(turnovers_df, left_on=['season', 'week', 'home_team'], right_on=['season', 'week', 'posteam'], how='left').rename(columns={'turnovers_committed': 'h_turnovers'}).drop(columns=['posteam'])
        df = df.merge(turnovers_df, left_on=['season', 'week', 'away_team'], right_on=['season', 'week', 'posteam'], how='left').rename(columns={'turnovers_committed': 'a_turnovers'}).drop(columns=['posteam'])
        df['h_turnovers'] = df['h_turnovers'].fillna(1.0)
        df['a_turnovers'] = df['a_turnovers'].fillna(1.0)
    else:
        df['h_turnovers'] = 1.0
        df['a_turnovers'] = 1.0

    home_games = df[['season', 'week', 'home_team', 'home_margin', 'h_turnovers']].rename(columns={'home_team': 'team', 'home_margin': 'margin', 'h_turnovers': 'to'})
    away_games = df[['season', 'week', 'away_team', 'away_margin', 'a_turnovers']].rename(columns={'away_team': 'team', 'away_margin': 'margin', 'a_turnovers': 'to'})
    all_games = pd.concat([home_games, away_games]).sort_values(['team', 'season', 'week'])
    
    all_games['form_rating'] = all_games.groupby(['team', 'season'])['margin'].transform(lambda x: x.ewm(span=5).mean().shift(1))
    all_games['rolling_to'] = all_games.groupby(['team', 'season'])['to'].transform(lambda x: x.ewm(span=5).mean().shift(1))
    
    df = df.merge(all_games[['season', 'week', 'team', 'form_rating', 'rolling_to']], left_on=['season', 'week', 'home_team'], right_on=['season', 'week', 'team'], how='left').rename(columns={'form_rating': 'home_form', 'rolling_to': 'h_rolling_to'})
    df = df.merge(all_games[['season', 'week', 'team', 'form_rating', 'rolling_to']], left_on=['season', 'week', 'away_team'], right_on=['season', 'week', 'team'], how='left').rename(columns={'form_rating': 'away_form', 'rolling_to': 'a_rolling_to'})
    
    for col in ['home_form', 'away_form', 'h_rolling_to', 'a_rolling_to']:
        df[col] = df[col].fillna(0)
    
    return df

def calculate_player_values(pbp_df):
    if pbp_df.empty: return pd.DataFrame()
    pass_epa = pbp_df.groupby(['season', 'passer_player_name'])['epa'].sum().reset_index().rename(columns={'passer_player_name': 'full_name'})
    rush_epa = pbp_df.groupby(['season', 'rusher_player_name'])['epa'].sum().reset_index().rename(columns={'rusher_player_name': 'full_name'})
    rec_epa = pbp_df.groupby(['season', 'receiver_player_name'])['epa'].sum().reset_index().rename(columns={'receiver_player_name': 'full_name'})
    all_epa = pd.concat([pass_epa, rush_epa, rec_epa])
    player_values = all_epa.groupby(['season', 'full_name'])['epa'].sum().reset_index()
    player_values['value_per_game'] = player_values['epa'] / 17
    player_values['value_per_game'] = player_values['value_per_game'].clip(lower=0)
    return player_values

def calculate_injury_impact(schedule_df, injury_df, depth_chart_df, player_values_df):
    if injury_df.empty: return schedule_df.assign(injury_value_diff=0)
    active = injury_df[injury_df['report_status'].isin(['Out', 'Injured Reserve'])].copy()
    if not depth_chart_df.empty:
        active = active.merge(depth_chart_df[['season','club_code','full_name','depth_team']], left_on=['season','team','full_name'], right_on=['season','club_code','full_name'], how='left')
        active = active[active['depth_team'].isin(['1','2',1,2])]
    active = active.merge(player_values_df, on=['season','full_name'], how='left')
    active['value_per_game'] = active['value_per_game'].fillna(0.5)
    lost = active.groupby(['season','week','team'])['value_per_game'].sum().reset_index(name='lost_epa')
    df = schedule_df.merge(lost, left_on=['season','week','home_team'], right_on=['season','week','team'], how='left').fillna({'lost_epa':0}).rename(columns={'lost_epa':'h_loss'})
    df = df.merge(lost, left_on=['season','week','away_team'], right_on=['season','week','team'], how='left').fillna({'lost_epa':0}).rename(columns={'lost_epa':'a_loss'})
    df['injury_value_diff'] = df['a_loss'] - df['h_loss']
    return df

def calculate_coaching(pbp_df):
    if pbp_df.empty: return pd.DataFrame()
    neutral = pbp_df[(pbp_df['wp']>=0.2) & (pbp_df['wp']<=0.8) & (pbp_df['down'].isin([1,2]))]
    avg = neutral['pass'].mean()
    tm = neutral.groupby(['season','posteam'])['pass'].mean().reset_index(name='rate')
    tm['proe'] = tm['rate'] - avg
    return tm

def calculate_elo(schedule_df):
    print("Calculating Calibrated Elo (Low HFA)...")
    df = schedule_df.copy()
    elos = {t:1500 for t in pd.concat([df['home_team'], df['away_team']]).unique()}
    h_elos, a_elos = [], []
    HFA_ELO_POINTS = 20
    for _, r in df.sort_values('gameday').iterrows():
        h, a = r['home_team'], r['away_team']
        h_elos.append(elos.get(h,1500)); a_elos.append(elos.get(a,1500))
        if pd.notna(r['home_score']):
            s_diff = r['home_score'] - r['away_score']
            res = 1 if s_diff > 0 else 0 if s_diff < 0 else 0.5
            dr = elos.get(h,1500) + HFA_ELO_POINTS - elos.get(a,1500)
            e = 1 / (1 + 10**(-dr/400))
            shift = 20 * np.log(abs(s_diff)+1) * (res - e)
            elos[h] += shift; elos[a] -= shift
    df['home_elo'] = h_elos; df['away_elo'] = a_elos
    joblib.dump(elos, 'current_elo_state_blind.pkl')
    return df

def calculate_unit_stats(pbp_df):
    if pbp_df.empty: return pd.DataFrame()
    plays = pbp_df[pbp_df['play_type'].isin(['pass', 'run'])]
    
    # Calculate Offense
    off = plays.groupby(['season', 'posteam', 'play_type'])['epa'].mean().unstack().reset_index()
    off.columns = ['season', 'team', 'off_pass_epa', 'off_run_epa']
    
    # Calculate Defense
    defs = plays.groupby(['season', 'defteam', 'play_type'])['epa'].mean().unstack().reset_index()
    defs.columns = ['season', 'team', 'def_pass_epa', 'def_run_epa']
    
    # Merge and Fill NaN
    merged = off.merge(defs, on=['season', 'team'], how='outer')
    return merged.fillna(0.0)

def run_pipeline():
    pbp, sched, inj, depth = load_all_data(TRAIN_YEARS)
    
    sharp_stats, turnovers = calculate_sharp_metrics(pbp)
    sched = calculate_advanced_context(sched, turnovers)
    elo_df = calculate_elo(sched)
    p_val = calculate_player_values(pbp)
    df = calculate_injury_impact(elo_df, inj, depth, p_val)
    coach = calculate_coaching(pbp)
    units = calculate_unit_stats(pbp)
    
    # Merge Sharp Stats into Units
    units = units.merge(sharp_stats, left_on=['season','team'], right_on=['season','posteam'], how='left').drop(columns=['posteam'])
    units.to_csv("unit_stats_blind.csv", index=False)
    
    df = df.merge(coach, left_on=['season','home_team'], right_on=['season','posteam'], how='left').rename(columns={'proe':'h_proe'})
    df = df.merge(coach, left_on=['season','away_team'], right_on=['season','posteam'], how='left').rename(columns={'proe':'a_proe'})
    
    # --- FIXED MERGE WITH RENAMING ---
    df = df.merge(units, left_on=['season','home_team'], right_on=['season','team']).rename(columns={
        'off_pass_epa':'h_off_pass', 'def_pass_epa':'h_def_pass', 
        'off_run_epa':'h_off_run', 'def_run_epa':'h_def_run',
        'edsr': 'h_edsr', 'special_teams_epa': 'h_st_epa' # <--- RENAMED CORRECTLY
    })
    
    df = df.merge(units, left_on=['season','away_team'], right_on=['season','team'], suffixes=(None, '_dup')).rename(columns={
        'off_pass_epa':'a_off_pass', 'def_pass_epa':'a_def_pass', 
        'off_run_epa':'a_off_run', 'def_run_epa':'a_def_run',
        'edsr': 'a_edsr', 'special_teams_epa': 'a_st_epa' # <--- RENAMED CORRECTLY
    })
    
    # Fill NAs
    for c in ['h_edsr','a_edsr','h_st_epa','a_st_epa']: 
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            df[c] = 0

    df['pass_mismatch'] = df['h_off_pass'] - df['a_def_pass']
    df['run_mismatch'] = df['h_off_run'] - df['a_def_run']
    df['st_mismatch'] = df['h_st_epa'] - df['a_st_epa'] 
    df['edsr_mismatch'] = df['h_edsr'] - df['a_edsr']   
    df['luck_mismatch'] = df['h_rolling_to'] - df['a_rolling_to'] 
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['coaching_mismatch'] = df['h_proe'] - df['a_proe']
    df['is_thursday'] = np.where(df['weekday'] == 'Thursday', 1, 0)
    df['target_spread'] = df['home_score'] - df['away_score']
    df['target_total'] = df['home_score'] + df['away_score']
    
    if 'div_game' not in df.columns: df['div_game'] = 0
    
    latest_form = df.sort_values('week').groupby('home_team')['home_form'].last().reset_index()
    latest_form.columns = ['team', 'form_rating']
    latest_form.to_csv("team_context_blind.csv", index=False)
    p_val.to_csv("player_values_blind.csv", index=False)
    coach.to_csv("team_tendencies_blind.csv", index=False)

    df = df[df['week'] < 18]
    train = df.dropna(subset=['target_spread', 'h_proe', 'pass_mismatch'])
    
    # === TRAINING SPREAD MODEL ===
    print("Training SPREAD Model (Balanced Sharp)...")
    features = [
        'elo_diff', 'home_elo', 'away_elo',
        'coaching_mismatch', 'h_proe', 'a_proe',
        'injury_value_diff', 'is_thursday', 'div_game',
        'body_clock_disadvantage', 'rest_diff',
        'home_form', 'away_form',
        'pass_mismatch', 'run_mismatch',
        'st_mismatch', 'edsr_mismatch', 'luck_mismatch'
    ]
    
    monotone = {'elo_diff': 1, 'edsr_mismatch': 1}
    
    X_train, X_test, y_train, y_test = train_test_split(train[features], train['target_spread'], test_size=0.15, random_state=42)
    
    model_spread = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.01, max_depth=5, 
        early_stopping_rounds=50, reg_lambda=1.0, 
        monotone_constraints=monotone
    )
    model_spread.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    joblib.dump(model_spread, 'nfl_model_blind.pkl')
    
    # === TRAINING TOTAL MODEL ===
    print("Training TOTAL Model...")
    X_train_tot, X_test_tot, y_train_tot, y_test_tot = train_test_split(train[features], train['target_total'], test_size=0.15, random_state=42)
    model_total = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.01, max_depth=5, early_stopping_rounds=50, reg_lambda=1.0)
    model_total.fit(X_train_tot, y_train_tot, eval_set=[(X_test_tot, y_test_tot)], verbose=False)
    joblib.dump(model_total, 'nfl_model_total.pkl')
    
    imp = model_spread.get_booster().get_score(importance_type='gain')
    total = sum(imp.values())
    print("\nSPREAD BRAIN PRIORITY:")
    for k, v in sorted(imp.items(), key=lambda x:x[1], reverse=True):
        print(f"{k:<20}: {(v/total)*100:.1f}%")

    joblib.dump(features, 'model_features_blind.pkl')
    print("Training Complete.")

if __name__ == "__main__":
    run_pipeline()
