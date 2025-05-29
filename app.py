import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import joblib
from datetime import datetime, timedelta
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import uuid
from collections import Counter
from itertools import product

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
MODEL_FILE = os.path.join(tempfile.gettempdir(), "baccarat_rf_model.joblib")
SCALER_FILE = os.path.join(tempfile.gettempdir(), "baccarat_scaler.joblib")
SHOE_SIZE = 100
HISTORY_LIMIT = 100
STOP_LOSS_DEFAULT = 1.0
WIN_LIMIT = 1.5
PARLAY_TABLE = {i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
    (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
    (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
], 1)}
FOUR_TIER_TABLE = {1: {'step1': 1, 'step2': 3}, 2: {'step1': 7, 'step2': 21},
                   3: {'step1': 50, 'step2': 150}, 4: {'step1': 350, 'step2': 1050}}
FOUR_TIER_MIN_BANKROLL = sum(FOUR_TIER_TABLE[tier][step] for tier in FOUR_TIER_TABLE for step in FOUR_TIER_TABLE[tier])
FLATBET_LEVELUP_TABLE = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
FLATBET_LEVELUP_MIN_BANKROLL = sum(FLATBET_LEVELUP_TABLE[level] * 5 for level in FLATBET_LEVELUP_TABLE)
FLATBET_LEVELUP_THRESHOLDS = {1: -5.0, 2: -10.0, 3: -20.0, 4: -40.0, 5: -40.0}
GRID = [
    [0, 1, 2, 3, 4, 4, 3, 2, 1], [1, 0, 1, 3, 4, 4, 4, 3, 2], [2, 1, 0, 2, 3, 4, 5, 4, 3],
    [3, 3, 2, 0, 2, 4, 5, 6, 5], [4, 4, 3, 2, 0, 2, 5, 7, 7], [4, 4, 4, 4, 2, 0, 3, 7, 9],
    [3, 4, 5, 5, 5, 3, 0, 5, 9], [2, 3, 4, 6, 7, 7, 5, 0, 8], [1, 2, 3, 5, 7, 9, 9, 8, 0],
    [1, 1, 2, 3, 5, 8, 11, 15, 15], [0, 0, 1, 2, 4, 8, 15, 15, 30]
]
GRID_MIN_BANKROLL = max(max(row) for row in GRID) * 5
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Moon", "FourTier", "FlatbetLevelUp", "Grid", "OscarGrind", "1222"]
OUTCOME_MAPPING = {'P': 0, 'B': 1, 'T': 2}
REVERSE_MAPPING = {0: 'P', 1: 'B', 2: 'T'}
_1222_MIN_BANKROLL = 10
START_BETTING_HAND = 6
CONFIDENCE_THRESHOLD = 60.0

# --- CSS Styling ---
def apply_css():
    st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; padding: 20px; background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    h1 { color: #1a3c6e; font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem; }
    h2 { color: #2c5282; font-size: 1.5rem; font-weight: 600; margin: 1.5rem 0 1rem; }
    .stButton > button { background: #1a3c6e; color: white; border: none; border-radius: 8px; padding: 10px; font-size: 14px; width: 100%; }
    .stButton > button:hover { background: #2b6cb0; transform: translateY(-2px); box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .result-button-player { background: linear-gradient(to bottom, #3182ce, #2b6cb0); }
    .result-button-player:hover { background: linear-gradient(to bottom, #63b3ed, #3182ce); }
    .result-button-banker { background: linear-gradient(to bottom, #e53e3e, #c53030); }
    .result-button-banker:hover { background: linear-gradient(to bottom, #fc8181, #e53e3e); }
    .result-button-tie { background: linear-gradient(to bottom, #38a169, #2f855a); }
    .result-button-tie:hover { background: linear-gradient(to bottom, #68d391, #38a169); }
    .result-button-undo { background: linear-gradient(to bottom, #718096, #5a667f); }
    .result-button-undo:hover { background: linear-gradient(to bottom, #a0aec0, #718096); }
    .stNumberInput input, .stSelectbox select { border-radius: 8px; border: 1px solid #e2e8f0; padding: 10px; font-size: 14px; }
    .st-expander { border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 1rem; }
    .bead-plate { background: #edf2f7; padding: 10px; border-radius: 8px; overflow-x: auto; }
    </style>
    """, unsafe_allow_html=True)

# --- Session Tracking ---
def track_user_session() -> int:
    return 1

# --- AI Prediction ---
def extract_ngrams(sequence, n):
    return [''.join(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

def get_shoe_features(sequence):
    if not sequence:
        return [0] * 21
    valid_sequence = [r for r in sequence if r in ['P', 'B', 'T']]
    total_hands = len(valid_sequence)
    
    counts = Counter(valid_sequence)
    p_ratio = counts.get('P', 0) / total_hands if total_hands > 0 else 0
    b_ratio = counts.get('B', 0) / total_hands if total_hands > 0 else 0
    t_ratio = counts.get('T', 0) / total_hands if total_hands > 0 else 0
    
    bigrams = extract_ngrams(valid_sequence, 2)
    bigram_counts = Counter(bigrams)
    bigram_features = [bigram_counts.get(combo, 0) / len(bigrams) if bigrams else 0 
                       for combo in ['PP', 'BB', 'PB', 'BP', 'PT', 'BT']]
    
    trigrams = extract_ngrams(valid_sequence, 3)
    trigram_counts = Counter(trigrams)
    trigram_features = [trigram_counts.get(combo, 0) / len(trigrams) if trigrams else 0 
                        for combo in ['PPP', 'BBB', 'PBP', 'BPB']]
    
    fourgrams = extract_ngrams(valid_sequence, 4)
    fourgram_counts = Counter(fourgrams)
    fourgram_features = [fourgram_counts.get(combo, 0) / len(fourgrams) if fourgrams else 0 
                         for combo in ['PPPP', 'BBBB', 'PBPB', 'BPBP']]
    
    streak_length = 0
    current_outcome = valid_sequence[-1] if valid_sequence else None
    for outcome in reversed(valid_sequence):
        if outcome == current_outcome:
            streak_length += 1
        else:
            break
    chop_count = sum(1 for i in range(1, len(valid_sequence)) if valid_sequence[i] != valid_sequence[i-1])
    chop_ratio = chop_count / (total_hands - 1) if total_hands > 1 else 0
    
    return [p_ratio, b_ratio, t_ratio] + bigram_features + trigram_features + fourgram_features + [streak_length / 10.0, chop_ratio]

def train_ml_model(sequence):
    if len(sequence) < 5:
        return None, None
    X, y = [], []
    for i in range(4, len(sequence)):
        window = sequence[i-4:i]
        next_outcome = sequence[i]
        features = [OUTCOME_MAPPING[window[j]] for j in range(4)] + [
            st.session_state.time_before_last.get(k, len(sequence) + 1) / (len(sequence) + 1)
            for k in ['P', 'B']
        ] + [st.session_state.current_streak / 10.0, st.session_state.current_chop_count / 10.0,
             st.session_state.bets_won / max(st.session_state.bets_placed, 1)]
        shoe_features = get_shoe_features(sequence[:i])
        X.append(features + shoe_features)
        y.append(OUTCOME_MAPPING[next_outcome])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return model, scaler

def load_ml_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            return model, scaler
        except:
            return None, None
    return None, None

def predict_next_outcome(sequence, model, scaler):
    if len(sequence) < 4 or model is None:
        return 'P', 0.5
    window = sequence[-4:]
    features = [OUTCOME_MAPPING[window[j]] for j in range(4)] + [
        st.session_state.time_before_last.get(k, len(sequence) + 1) / (len(sequence) + 1)
        for k in ['P', 'B']
    ] + [st.session_state.current_streak / 10.0, st.session_state.current_chop_count / 10.0,
         st.session_state.bets_won / max(st.session_state.bets_placed, 1)]
    shoe_features = get_shoe_features(sequence)
    X_scaled = scaler.transform([features + shoe_features])
    probs = model.predict_proba(X_scaled)[0]
    predicted_idx = np.argmax(probs)
    return REVERSE_MAPPING[predicted_idx], probs[predicted_idx]

# --- Session State ---
def initialize_session_state():
    defaults = {
        'bankroll': 0.0, 'base_bet': 0.0, 'initial_bankroll': 0.0, 'peak_bankroll': 0.0, 'sequence': [], 
        'bet_history': [], 'pending_bet': None, 'bets_placed': 0, 'bets_won': 0, 't3_level': 1, 
        't3_results': [], 'money_management': 'T3', 'stop_loss_percentage': STOP_LOSS_DEFAULT, 
        'stop_loss_enabled': True, 'win_limit': WIN_LIMIT, 'shoe_completed': False, 
        'safety_net_enabled': True, 'safety_net_percentage': 0.02,
        'advice': f"Need {START_BETTING_HAND} results to start betting", 'parlay_step': 1, 'parlay_wins': 0,
        'parlay_using_base': True, 'parlay_step_changes': 0, 'parlay_peak_step': 1, 'moon_level': 1,
        'moon_level_changes': 0, 'moon_peak_level': 1, 'target_profit_option': 'Profit %',
        'target_profit_percentage': 0.0, 'target_profit_units': 0.0, 'four_tier_level': 1,
        'four_tier_step': 1, 'four_tier_losses': 0, 'flatbet_levelup_level': 1,
        'flatbet_levelup_net_loss': 0.0, 'grid_pos': [0, 0], 'oscar_cycle_profit': 0.0,
        'oscar_current_bet_level': 1, 'current_streak': 0, 'current_streak_type': None,
        'longest_streak': 0, 'longest_streak_type': None, 'current_chop_count': 0, 'longest_chop': 0,
        'ml_model': None, 'ml_scaler': None, 'ai_mode': False, 'level_1222': 1, 
        'next_bet_multiplier_1222': 1, 'rounds_1222': 0, 'level_start_bankroll_1222': 0.0,
        'last_positions': {'P': [], 'B': [], 'T': []}, 'time_before_last': {'P': 0, 'B': 0, 'T': 0},
        'prediction_accuracy': {'P': 0.0, 'B': 0.0, 'T': 0.0, 'total': 0.0}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.ml_model is None and st.session_state.ml_scaler is None:
        st.session_state.ml_model, st.session_state.ml_scaler = load_ml_model()

def reset_session():
    setup_values = {k: st.session_state[k] for k in [
        'bankroll', 'base_bet', 'initial_bankroll', 'peak_bankroll', 'money_management', 'stop_loss_percentage',
        'stop_loss_enabled', 'safety_net_enabled', 'safety_net_percentage', 'target_profit_option',
        'target_profit_percentage', 'target_profit_units', 'win_limit', 'ai_mode'
    ]}
    initialize_session_state()
    st.session_state.update(setup_values)
    st.session_state.update({
        'sequence': [], 'bet_history': [], 'pending_bet': None, 'bets_placed': 0, 'bets_won': 0,
        't3_level': 1, 't3_results': [], 'shoe_completed': False, 
        'advice': f"Need {START_BETTING_HAND} results to start betting", 'parlay_step': 1,
        'parlay_wins': 0, 'parlay_using_base': True, 'parlay_step_changes': 0, 'parlay_peak_step': 1,
        'moon_level': 1, 'moon_level_changes': 0, 'moon_peak_level': 1, 'four_tier_level': 1,
        'four_tier_step': 1, 'four_tier_losses': 0, 'flatbet_levelup_level': 1, 'flatbet_levelup_net_loss': 0.0,
        'grid_pos': [0, 0], 'oscar_cycle_profit': 0.0, 'oscar_current_bet_level': 1,
        'current_streak': 0, 'current_streak_type': None, 'longest_streak': 0, 'longest_streak_type': None,
        'current_chop_count': 0, 'longest_chop': 0, 'level_1222': 1, 'next_bet_multiplier_1222': 1,
        'rounds_1222': 0, 'level_start_bankroll_1222': setup_values.get('bankroll', 0.0),
        'last_positions': {'P': [], 'B': [], 'T': []}, 'time_before_last': {'P': 0, 'B': 0, 'T': 0},
        'prediction_accuracy': {'P': 0.0, 'B': 0.0, 'T': 0.0, 'total': 0.0}
    })
    st.session_state.ml_model, st.session_state.ml_scaler = load_ml_model()

# --- Betting Logic ---
def calculate_bet_amount(bet_selection: str) -> float:
    try:
        if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
            return st.session_state.base_bet
        if st.session_state.money_management == 'Flatbet':
            return st.session_state.base_bet
        elif st.session_state.money_management == 'T3':
            return st.session_state.base_bet * st.session_state.t3_level
        elif st.session_state.money_management == 'Parlay16':
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            return st.session_state.base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
        elif st.session_state.money_management == 'Moon':
            return st.session_state.base_bet * st.session_state.moon_level
        elif st.session_state.money_management == 'FourTier':
            step_key = 'step1' if st.session_state.four_tier_step == 1 else 'step2'
            return st.session_state.base_bet * FOUR_TIER_TABLE[st.session_state.four_tier_level][step_key]
        elif st.session_state.money_management == 'FlatbetLevelUp':
            return st.session_state.base_bet * FLATBET_LEVELUP_TABLE[st.session_state.flatbet_levelup_level]
        elif st.session_state.money_management == 'Grid':
            return st.session_state.base_bet * GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]]
        elif st.session_state.money_management == 'OscarGrind':
            return st.session_state.base_bet * st.session_state.oscar_current_bet_level
        elif st.session_state.money_management == '1222':
            return st.session_state.base_bet * st.session_state.level_1222 * st.session_state.next_bet_multiplier_1222
        return 0.0
    except:
        return 0.0

def simulate_shoe_result():
    return random.choices(['P', 'B', 'T'], weights=[0.4586, 0.4460, 0.0954], k=1)[0]

def place_result(result: str):
    try:
        if st.session_state.stop_loss_enabled and st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage:
            if not st.session_state.safety_net_enabled:
                reset_session()
                st.warning(f"Stop-loss triggered at {st.session_state.stop_loss_percentage*100:.0f}%. Game reset.")
                return
        if st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.safety_net_percentage and st.session_state.safety_net_enabled:
            reset_session()
            st.info(f"Safety net triggered at {st.session_state.safety_net_percentage*100:.0f}%. Game reset.")
        if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
            reset_session()
            st.success(f"Win limit reached at {st.session_state.win_limit*100:.0f}%. Game reset.")
            return
        profit = st.session_state.bankroll - st.session_state.initial_bankroll
        if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0 and profit >= st.session_state.initial_bankroll * st.session_state.target_profit_percentage:
            reset_session()
            st.success(f"Target profit reached: ${profit:.2f} ({st.session_state.target_profit_percentage*100:.0f}%). Game reset.")
            return
        if st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0 and profit >= st.session_state.target_profit_units:
            reset_session()
            st.success(f"Target profit reached: ${profit:.2f} (Target: ${st.session_state.target_profit_units:.2f}). Game reset.")
            return

        previous_state = {
            'bankroll': st.session_state.bankroll, 't3_level': st.session_state.t3_level, 't3_results': st.session_state.t3_results.copy(),
            'parlay_step': st.session_state.parlay_step, 'parlay_wins': st.session_state.parlay_wins, 'parlay_using_base': st.session_state.parlay_using_base,
            'parlay_step_changes': st.session_state.parlay_step_changes, 'parlay_peak_step': st.session_state.parlay_peak_step,
            'moon_level': st.session_state.moon_level, 'moon_level_changes': st.session_state.moon_level_changes, 'moon_peak_level': st.session_state.moon_peak_level,
            'four_tier_level': st.session_state.four_tier_level, 'four_tier_step': st.session_state.four_tier_step, 'four_tier_losses': st.session_state.four_tier_losses,
            'flatbet_levelup_level': st.session_state.flatbet_levelup_level, 'flatbet_levelup_net_loss': st.session_state.flatbet_levelup_net_loss,
            'bets_placed': st.session_state.bets_placed, 'bets_won': st.session_state.bets_won, 'pending_bet': st.session_state.pending_bet, 
            'shoe_completed': st.session_state.shoe_completed, 'grid_pos': st.session_state.grid_pos.copy(),
            'oscar_cycle_profit': st.session_state.oscar_cycle_profit, 'oscar_current_bet_level': st.session_state.oscar_current_bet_level,
            'current_streak': st.session_state.current_streak, 'current_streak_type': st.session_state.current_streak_type,
            'longest_streak': st.session_state.longest_streak, 'longest_streak_type': st.session_state.longest_streak_type,
            'current_chop_count': st.session_state.current_chop_count, 'longest_chop': st.session_state.longest_chop,
            'level_1222': st.session_state.level_1222, 'next_bet_multiplier_1222': st.session_state.next_bet_multiplier_1222,
            'rounds_1222': st.session_state.rounds_1222, 'level_start_bankroll_1222': st.session_state.level_start_bankroll_1222,
            'last_positions': st.session_state.last_positions.copy(), 'time_before_last': st.session_state.time_before_last.copy(),
            'prediction_accuracy': st.session_state.prediction_accuracy.copy()
        }

        if result in ['P', 'B']:
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']] + [result]
            if len(valid_sequence) == 1 or st.session_state.current_streak_type != result:
                st.session_state.current_streak = 1
                st.session_state.current_streak_type = result
            else:
                st.session_state.current_streak += 1
            if st.session_state.current_streak > st.session_state.longest_streak:
                st.session_state.longest_streak = st.session_state.current_streak
                st.session_state.longest_streak_type = result
            if len(valid_sequence) > 1 and valid_sequence[-2] != result:
                st.session_state.current_chop_count += 1
            else:
                st.session_state.current_chop_count = 0
            if st.session_state.current_chop_count > st.session_state.longest_chop:
                st.session_state.longest_chop = st.session_state.current_chop_count
        else:
            st.session_state.current_streak = 0
            st.session_state.current_streak_type = None
            if st.session_state.current_chop_count > st.session_state.longest_chop:
                st.session_state.longest_chop = st.session_state.current_chop_count
            st.session_state.current_chop_count = 0

        bet_amount = 0
        bet_selection = None
        bet_outcome = None
        confidence = 0.0  # Default confidence
        if st.session_state.pending_bet and result in ['P', 'B']:
            bet_amount, bet_selection = st.session_state.pending_bet
            st.session_state.bets_placed += 1
            if result == bet_selection:
                winnings = bet_amount * (0.95 if bet_selection == 'B' else 1.0)
                st.session_state.bankroll += winnings
                st.session_state.bets_won += 1
                bet_outcome = 'win'
                if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                    if st.session_state.money_management == 'T3':
                        if not st.session_state.t3_results:
                            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                        st.session_state.t3_results.append('W')
                    elif st.session_state.money_management == 'Parlay16':
                        st.session_state.parlay_wins += 1
                        if st.session_state.parlay_wins == 2:
                            old_step = st.session_state.parlay_step
                            st.session_state.parlay_step = 1
                            st.session_state.parlay_wins = 0
                            st.session_state.parlay_using_base = True
                            if old_step != st.session_state.parlay_step:
                                st.session_state.parlay_step_changes += 1
                            st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                        else:
                            st.session_state.parlay_using_base = False
                    elif st.session_state.money_management == 'Moon':
                        st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                    elif st.session_state.money_management == 'FourTier':
                        st.session_state.four_tier_level = 1
                        st.session_state.four_tier_step = 1
                        st.session_state.four_tier_losses = 0
                        st.session_state.shoe_completed = True
                        st.session_state.advice = "Win recorded. Reset for a new shoe."
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += winnings / st.session_state.base_bet
                    elif st.session_state.money_management == 'Grid':
                        st.session_state.grid_pos[1] += 1
                        if st.session_state.grid_pos[1] >= len(GRID[0]):
                            st.session_state.grid_pos[1] = 0
                            if st.session_state.grid_pos[0] < len(GRID) - 1:
                                st.session_state.grid_pos[0] += 1
                        if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                            st.session_state.grid_pos = [0, 0]
                    elif st.session_state.money_management == 'OscarGrind':
                        st.session_state.oscar_cycle_profit += winnings
                        if st.session_state.oscar_cycle_profit >= st.session_state.base_bet:
                            st.session_state.oscar_current_bet_level = 1
                            st.session_state.oscar_cycle_profit = 0.0
                        else:
                            next_bet_level = st.session_state.oscar_current_bet_level + 1
                            potential_winnings = st.session_state.base_bet * next_bet_level * (0.95 if bet_selection == 'B' else 1.0)
                            if st.session_state.oscar_cycle_profit + potential_winnings > st.session_state.base_bet:
                                next_bet_level = max(1, int((st.session_state.base_bet - st.session_state.oscar_cycle_profit) / (st.session_state.base_bet * (0.95 if bet_selection == 'B' else 1.0)) + 0.99))
                            st.session_state.oscar_current_bet_level = next_bet_level
                    elif st.session_state.money_management == '1222':
                        st.session_state.next_bet_multiplier_1222 = 2
            else:
                st.session_state.bankroll -= bet_amount
                bet_outcome = 'loss'
                if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                    if st.session_state.money_management == 'T3':
                        st.session_state.t3_results.append('L')
                    elif st.session_state.money_management == 'Parlay16':
                        st.session_state.parlay_wins = 0
                        old_step = st.session_state.parlay_step
                        st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                        st.session_state.parlay_using_base = True
                        if old_step != st.session_state.parlay_step:
                            st.session_state.parlay_step_changes += 1
                        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                    elif st.session_state.money_management == 'Moon':
                        old_level = st.session_state.moon_level
                        st.session_state.moon_level += 1
                        if old_level != st.session_state.moon_level:
                            st.session_state.moon_level_changes += 1
                        st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                    elif st.session_state.money_management == 'FourTier':
                        st.session_state.four_tier_losses += 1
                        if st.session_state.four_tier_losses == 1:
                            st.session_state.four_tier_step = 2
                        elif st.session_state.four_tier_losses >= 2:
                            st.session_state.four_tier_level = min(st.session_state.four_tier_level + 1, 4)
                            st.session_state.four_tier_step = 1
                            st.session_state.four_tier_losses = 0
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss -= bet_amount / st.session_state.base_bet
                        current_level = st.session_state.flatbet_levelup_level
                        if current_level < 5 and st.session_state.flatbet_levelup_net_loss <= FLATBET_LEVELUP_THRESHOLDS[current_level]:
                            st.session_state.flatbet_levelup_level = min(st.session_state.flatbet_levelup_level + 1, 5)
                            st.session_state.flatbet_levelup_net_loss = 0.0
                    elif st.session_state.money_management == 'Grid':
                        st.session_state.grid_pos[0] += 1
                        if st.session_state.grid_pos[0] >= len(GRID):
                            st.session_state.grid_pos = [0, 0]
                        if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                            st.session_state.grid_pos = [0, 0]
                    elif st.session_state.money_management == '1222':
                        st.session_state.next_bet_multiplier_1222 = 1
            if st.session_state.money_management == 'T3' and len(st.session_state.t3_results) == 3:
                wins = st.session_state.t3_results.count('W')
                losses = st.session_state.t3_results.count('L')
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1 if wins > losses else st.session_state.t3_level + 1 if losses > wins else st.session_state.t3_level)
                st.session_state.t3_results = []
            if st.session_state.money_management == '1222' and bet_amount > 0:
                st.session_state.rounds_1222 += 1
                if st.session_state.rounds_1222 >= 5:
                    if st.session_state.bankroll >= st.session_state.peak_bankroll:
                        st.session_state.level_1222 = 1
                        st.session_state.next_bet_multiplier_1222 = 1
                        st.session_state.rounds_1222 = 0
                        st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
                    elif st.session_state.bankroll > st.session_state.level_start_bankroll_1222:
                        st.session_state.level_1222 = max(1, st.session_state.level_1222 - 1)
                        st.session_state.next_bet_multiplier_1222 = 1
                        st.session_state.rounds_1222 = 0
                        st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
                    else:
                        st.session_state.level_1222 += 1
                        st.session_state.next_bet_multiplier_1222 = 1
                        st.session_state.rounds_1222 = 0
                        st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
            st.session_state.peak_bankroll = max(st.session_state.peak_bankroll, st.session_state.bankroll)
            st.session_state.pending_bet = None

        if result in ['P', 'B', 'T']:
            st.session_state.sequence.append(result)
            current_position = len(st.session_state.sequence)
            st.session_state.last_positions[result].append(current_position)
            if len(st.session_state.last_positions[result]) > 2:
                st.session_state.last_positions[result].pop(0)
            for outcome in ['P', 'B', 'T']:
                if len(st.session_state.last_positions[outcome]) >= 2:
                    st.session_state.time_before_last[outcome] = current_position - st.session_state.last_positions[outcome][-2]
                else:
                    st.session_state.time_before_last[outcome] = current_position + 1

        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
        if len(valid_sequence) >= 5:
            st.session_state.ml_model, st.session_state.ml_scaler = train_ml_model(valid_sequence)

        if st.session_state.bet_history and st.session_state.bet_history[-1].get('Bet_Selection'):
            last_bet = st.session_state.bet_history[-1]
            predicted = last_bet['Bet_Selection']
            actual = result
            if predicted == actual:
                total_preds = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection') == predicted)
                correct_preds = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection') == predicted and h['Result'] == predicted)
                st.session_state.prediction_accuracy[predicted] = (correct_preds / total_preds * 100) if total_preds > 0 else 0.0
                total_bets = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection'))
                total_correct = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection') and h['Result'] == h['Bet_Selection'])
                st.session_state.prediction_accuracy['total'] = (total_correct / total_bets * 100) if total_bets > 0 else 0.0

        st.session_state.bet_history.append({
            "Result": result, "Bet_Amount": bet_amount, "Bet_Selection": bet_selection, "Bet_Outcome": bet_outcome,
            "Money_Management": st.session_state.money_management, "AI_Prediction": st.session_state.advice,
            "Confidence": f"{confidence:.1f}%", "Previous_State": previous_state
        })
        if len(st.session_state.bet_history) > HISTORY_LIMIT:
            st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

        if len(valid_sequence) < START_BETTING_HAND:
            st.session_state.pending_bet = None
            st.session_state.advice = f"Need {START_BETTING_HAND - len(valid_sequence)} more results to start betting"
        elif len(valid_sequence) >= START_BETTING_HAND and result in ['P', 'B']:
            if len(st.session_state.sequence) >= SHOE_SIZE:
                st.session_state.shoe_completed = True
            if st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
                st.session_state.pending_bet = None
                st.session_state.advice = "Shoe completed. AI-only betting stopped."
                return
            bet_selection, confidence = predict_next_outcome(valid_sequence, st.session_state.ml_model, st.session_state.ml_scaler)
            confidence *= 100
            strategy_used = 'AI'

            if confidence >= CONFIDENCE_THRESHOLD and bet_selection in ['P', 'B']:
                bet_amount = calculate_bet_amount(bet_selection)
                if bet_amount <= st.session_state.bankroll:
                    st.session_state.pending_bet = (bet_amount, bet_selection)
                    strategy_info = f"{st.session_state.money_management}"
                    if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                        strategy_info = "Safety Net (Flatbet)"
                    elif st.session_state.money_management == 'T3':
                        strategy_info += f" Level {st.session_state.t3_level}"
                    elif st.session_state.money_management == 'Parlay16':
                        strategy_info += f" Step {st.session_state.parlay_step}/16"
                    elif st.session_state.money_management == 'Moon':
                        strategy_info += f" Level {st.session_state.moon_level}"
                    elif st.session_state.money_management == 'FourTier':
                        strategy_info += f" Level {st.session_state.four_tier_level} Step {st.session_state.four_tier_step}"
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        strategy_info += f" Level {st.session_state.flatbet_levelup_level}"
                    elif st.session_state.money_management == 'Grid':
                        strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                    elif st.session_state.money_management == 'OscarGrind':
                        strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level}"
                    elif st.session_state.money_management == '1222':
                        strategy_info += f" Level {st.session_state.level_1222}, Rounds {st.session_state.rounds_1222}, Bet: {st.session_state.next_bet_multiplier_1222 * st.session_state.level_1222}u"
                    st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%)"
                else:
                    st.session_state.pending_bet = None
                    st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (low confidence: {confidence:.1f}% or Tie)"
    except Exception as e:
        st.error(f"Error in place_result: {str(e)}")

def run_simulation():
    for _ in range(SHOE_SIZE):
        if st.session_state.shoe_completed:
            break
        result = simulate_shoe_result()
        place_result(result)
        time.sleep(0.01)
    st.session_state.shoe_completed = True
    st.rerun()

def render_setup_form():
    with st.expander("Setup Session", expanded=not st.session_state.initial_bankroll):
        with st.form("setup_form"):
            bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
            base_bet = st.number_input("Base Bet ($)", min_value=0.0, value=10.0, step=1.0)
            money_management = st.selectbox("Money Management Strategy", STRATEGIES)
            stop_loss_enabled = st.checkbox("Enable Stop Loss", value=True)
            stop_loss_percentage = st.number_input("Stop Loss Percentage", min_value=0.0, max_value=100.0, value=STOP_LOSS_DEFAULT * 100, step=5.0) / 100
            safety_net_enabled = st.checkbox("Enable Safety Net", value=True)
            safety_net_percentage = st.number_input("Safety Net Percentage", min_value=0.0, max_value=100.0, value=2.0, step=1.0) / 100
            win_limit = st.number_input("Win Limit (Multiple of Bankroll)", min_value=1.0, value=WIN_LIMIT, step=0.5)
            target_mode = st.selectbox("Target Profit Mode", ["None", "Profit %", "Units"])
            target_value = 0.0
            if target_mode == "Profit %":
                target_value = st.number_input("Target Profit (%)", min_value=0.0, value=10.0, step=5.0)
            elif target_mode == "Units":
                target_value = st.number_input("Target Profit (Units)", min_value=0.0, value=100.0, step=10.0)
            ai_mode = st.checkbox("Enable AI Auto-Play", value=False)
            min_bankroll = {
                "T3": base_bet * 3, "Flatbet": base_bet * 5, "Parlay16": base_bet * 190,
                "Moon": base_bet * 10, "FourTier": base_bet * FOUR_TIER_MIN_BANKROLL,
                "FlatbetLevelUp": base_bet * FLATBET_LEVELUP_MIN_BANKROLL, "Grid": base_bet * GRID_MIN_BANKROLL,
                "OscarGrind": base_bet * 10, "1222": base_bet * _1222_MIN_BANKROLL
            }
            submitted = st.form_submit_button("Start Session")
            if submitted:
                if bankroll < min_bankroll[money_management]:
                    st.error(f"Bankroll must be at least ${min_bankroll[money_management]:.2f} for {money_management}.")
                elif base_bet <= 0:
                    st.error("Base bet must be greater than 0.")
                else:
                    reset_session()
                    st.session_state.update({
                        'bankroll': bankroll, 'base_bet': base_bet, 'initial_bankroll': bankroll,
                        'peak_bankroll': bankroll, 'money_management': money_management, 'stop_loss_enabled': stop_loss_enabled,
                        'stop_loss_percentage': stop_loss_percentage, 'safety_net_enabled': safety_net_enabled,
                        'safety_net_percentage': safety_net_percentage, 'win_limit': win_limit,
                        'target_profit_option': target_mode, 'target_profit_percentage': target_value / 100 if target_mode == "Profit %" else 0.0,
                        'target_profit_units': target_value if target_mode == "Units" else 0.0, 'ai_mode': ai_mode,
                        'level_start_bankroll_1222': bankroll
                    })
                    st.success(f"Session started with {money_management}! AI Auto-Play: {'Enabled' if ai_mode else 'Off'}")
                    if ai_mode:
                        run_simulation()

def render_result_input():
    with st.expander("Enter Result", expanded=True):
        if st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
            st.success(f"Shoe of {SHOE_SIZE} hands completed!")
        elif st.session_state.shoe_completed:
            st.info("Continuing with safety net.")
        cols = st.columns(4)
        with cols[0]:
            if st.button("Player", key="player_btn", disabled=(st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0 or st.session_state.ai_mode):
                place_result("P")
                st.rerun()
        with cols[1]:
            if st.button("Banker", key="banker_btn", disabled=(st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0 or st.session_state.ai_mode):
                place_result("B")
                st.rerun()
        with cols[2]:
            if st.button("Tie", key="tie_btn", disabled=(st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0 or st.session_state.ai_mode):
                place_result("T")
                st.rerun()
        with cols[3]:
            if st.button("Undo Last", key="undo_btn", disabled=not st.session_state.bet_history or (st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0 or st.session_state.ai_mode):
                if not st.session_state.sequence:
                    st.warning("No results to undo.")
                else:
                    last_bet = st.session_state.bet_history.pop()
                    st.session_state.sequence.pop()
                    for key, value in last_bet["Previous_State"].items():
                        st.session_state[key] = value
                    if last_bet["Bet_Amount"] > 0:
                        st.session_state.bets_placed -= 1
                        if last_bet["Bet_Outcome"] == 'win':
                            st.session_state.bankroll -= last_bet["Bet_Amount"] * (0.95 if last_bet["Bet_Selection"] == 'B' else 1.0)
                            st.session_state.bets_won -= 1
                    last_result = last_bet["Result"]
                    if last_result in st.session_state.last_positions and st.session_state.last_positions[last_result]:
                        st.session_state.last_positions[last_result].pop()
                    current_position = len(st.session_state.sequence)
                    for outcome in ['P', 'B', 'T']:
                        if len(st.session_state.last_positions[outcome]) >= 2:
                            st.session_state.time_before_last[outcome] = current_position - st.session_state.last_positions[outcome][-2]
                        else:
                            st.session_state.time_before_last[outcome] = current_position + 1
                    valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
                    if len(valid_sequence) < START_BETTING_HAND:
                        st.session_state.pending_bet = None
                        st.session_state.advice = f"Need {START_BETTING_HAND - len(valid_sequence)} more results to start betting"
                    else:
                        if len(valid_sequence) >= 5:
                            st.session_state.ml_model, st.session_state.ml_scaler = train_ml_model(valid_sequence)
                        if len(st.session_state.sequence) >= SHOE_SIZE:
                            st.session_state.shoe_completed = True
                        if st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
                            st.session_state.pending_bet = None
                            st.session_state.advice = "Shoe completed. AI-only betting paused."
                            st.rerun()
                        bet_selection, confidence = predict_next_outcome(valid_sequence, st.session_state.ml_model, st.session_state.ml_scaler)
                        confidence *= 100
                        strategy_used = 'AI'
                        if confidence >= CONFIDENCE_THRESHOLD and bet_selection in ['P', 'B']:
                            bet_amount = calculate_bet_amount(bet_selection)
                            if bet_amount <= st.session_state.bankroll:
                                st.session_state.pending_bet = (bet_amount, bet_selection)
                                strategy_info = f"{st.session_state.money_management}"
                                if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                                    strategy_info = "Safety Net (Flatbet)"
                                elif st.session_state.money_management == 'T3':
                                    strategy_info += f" Level {st.session_state.t3_level}"
                                elif st.session_state.money_management == 'Parlay16':
                                    strategy_info += f" Step {st.session_state.parlay_step}/16"
                                elif st.session_state.money_management == 'Moon':
                                    strategy_info += f" Level {st.session_state.moon_level}"
                                elif st.session_state.money_management == 'FourTier':
                                    strategy_info += f" Level {st.session_state.four_tier_level} Step {st.session_state.four_tier_step}"
                                elif st.session_state.money_management == 'FlatbetLevelUp':
                                    strategy_info += f" Level {st.session_state.flatbet_levelup_level}"
                                elif st.session_state.money_management == 'Grid':
                                    strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                                elif st.session_state.money_management == 'OscarGrind':
                                    strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level}"
                                elif st.session_state.money_management == '1222':
                                    strategy_info += f" Level {st.session_state.level_1222}, Rounds {st.session_state.rounds_1222}, Bet: {st.session_state.next_bet_multiplier_1222 * st.session_state.level_1222}u"
                                st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%)"
                            else:
                                st.session_state.pending_bet = None
                                st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
                        else:
                            st.session_state.pending_bet = None
                            st.session_state.advice = f"Skip betting (low confidence: {confidence:.1f}% or Tie)"
                    st.success("Undone last action.")
                    st.rerun()
        if st.session_state.shoe_completed and st.button("Reset and Start New Shoe", key="new_shoe_btn"):
            reset_session()
            st.session_state.shoe_completed = False
            st.rerun()

def render_bead_plate():
    with st.expander("Bead Plate", expanded=True):
        st.markdown("**Bead Plate**")
        sequence = st.session_state.sequence[-84:]
        grid = [['' for _ in range(14)] for _ in range(6)]
        for i, result in enumerate(sequence):
            if result in ['P', 'B', 'T']:
                col = i // 6
                row = i % 6
                if col < 14:
                    color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                    grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'
        for row in grid:
            st.markdown(' '.join(row), unsafe_allow_html=True)

def render_prediction():
    with st.expander("Prediction", expanded=True):
        if st.session_state.bankroll == 0:
            st.info("Start a session with bankroll and base bet.")
        elif st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
            st.info("Session ended. Reset to start a new session.")
        else:
            advice = st.session_state.advice
            text_color = '#3182ce' if ' on P ' in advice else '#e53e3e' if ' on B ' in advice else '#2d3748'
            st.markdown(f'<p style="font-size:18px; font-weight:bold; color:{text_color};">{advice}</p>', unsafe_allow_html=True)

def render_status():
    with st.expander("Session Status", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
            st.markdown(f"**Profit**: ${st.session_state.bankroll - st.session_state.initial_bankroll:.2f}")
            st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
            st.markdown(f"**Stop Loss**: {'On' if st.session_state.stop_loss_enabled else 'Off'}, {st.session_state.stop_loss_percentage*100:.0f}%")
            target = f"{st.session_state.target_profit_percentage*100:.0f}%" if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0 else f"${st.session_state.target_profit_units:.2f}" if st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0 else "None"
            st.markdown(f"**Target Profit**: {target}")
        with col2:
            st.markdown(f"**Safety Net**: {'On' if st.session_state.safety_net_enabled else 'Off'}")
            st.markdown(f"**Hands Played**: {len(st.session_state.sequence)}")
            st.markdown(f"**AI Mode**: {'On' if st.session_state.ai_mode else 'Off'}")
            strategy_info = f"{st.session_state.money_management}"
            if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                strategy_info = "Safety Net (Flatbet)"
            elif st.session_state.money_management == 'T3':
                strategy_info += f" (Level {st.session_state.t3_level})"
            elif st.session_state.money_management == 'Parlay16':
                strategy_info += f" (Step {st.session_state.parlay_step}/16)"
            elif st.session_state.money_management == 'Moon':
                strategy_info += f" (Level {st.session_state.moon_level})"
            elif st.session_state.money_management == 'FourTier':
                strategy_info += f" (Level {st.session_state.four_tier_level}, Step {st.session_state.four_tier_step})"
            elif st.session_state.money_management == 'FlatbetLevelUp':
                strategy_info += f" (Level {st.session_state.flatbet_levelup_level})"
            elif st.session_state.money_management == 'Grid':
                strategy_info += f" (Grid {st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
            elif st.session_state.money_management == 'OscarGrind':
                strategy_info += f" (Bet Level {st.session_state.oscar_current_bet_level})"
            elif st.session_state.money_management == '1222':
                strategy_info += f" (Level {st.session_state.level_1222}, Rounds {st.session_state.rounds_1222})"
            st.markdown(f"**Strategy**: {strategy_info}")
            st.markdown(f"**Bets Placed**: {st.session_state.bets_placed}")
            st.markdown(f"**Bets Won**: {st.session_state.bets_won}")
            tbl_display = {k: f"{v}" if v <= len(st.session_state.sequence) else "N/A" for k, v in st.session_state.time_before_last.items()}
            st.markdown(
                f"**Time Before Last**:<br>P: {tbl_display['P']} hands<br>B: {tbl_display['B']} hands<br>T: {tbl_display['T']} hands",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Streak**: {st.session_state.current_streak} ({st.session_state.current_streak_type or 'None'})<br>"
                f"**Longest Streak**: {st.session_state.longest_streak} ({st.session_state.longest_streak_type or 'None'})<br>"
                f"**Chop**: {st.session_state.current_chop_count}<br>**Longest Chop**: {st.session_state.longest_chop}",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Prediction Accuracy**:<br>"
                f"Player: {st.session_state.prediction_accuracy['P']:.1f}%<br>"
                f"Banker: {st.session_state.prediction_accuracy['B']:.1f}%<br>"
                f"Tie: {st.session_state.prediction_accuracy['T']:.1f}%<br>"
                f"Overall: {st.session_state.prediction_accuracy['total']:.1f}%",
                unsafe_allow_html=True
            )

def render_history():
    with st.expander("Bet History", expanded=True):
        if not st.session_state.bet_history:
            st.write("No history available.")
        else:
            n = st.slider("Show last N bets", 5, 50, 10)
            st.dataframe([
                {
                    "Result": h["Result"], "Bet": h["Bet_Selection"] if h.get("Bet_Selection") else "-",
                    "Amount": f"${h['Bet_Amount']:.2f}" if h["Bet_Amount"] > 0 else "-",
                    "Outcome": h["Bet_Outcome"] if h["Bet_Outcome"] else "-", "AI_Prediction": h["AI_Prediction"],
                    "Confidence": h["Confidence"]
                } for h in st.session_state.bet_history[-n:]
            ], use_container_width=True)

def render_insights():
    with st.expander("AI Insights", expanded=False):
        if not st.session_state.sequence:
            st.write("No data available for insights.")
        else:
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
            counts = Counter(valid_sequence)
            total = len(valid_sequence)
            st.markdown(f"**Shoe Composition**:<br>"
                        f"Player: {counts.get('P', 0)} ({counts.get('P', 0)/total*100:.1f}%)<br>"
                        f"Banker: {counts.get('B', 0)} ({counts.get('B', 0)/total*100:.1f}%)<br>"
                        f"Tie: {counts.get('T', 0)} ({counts.get('T', 0)/total*100:.1f}%)",
                        unsafe_allow_html=True)
            bigrams = Counter(extract_ngrams(valid_sequence, 2))
            st.markdown("**Top Bigrams**:<br>" + "<br>".join(
                [f"{k}: {v} ({v/len(bigrams)*100:.1f}%)" for k, v in bigrams.most_common(3)] if bigrams else ["No bigrams available"]
            ), unsafe_allow_html=True)
            trigrams = Counter(extract_ngrams(valid_sequence, 3))
            st.markdown("**Top Trigrams**:<br>" + "<br>".join(
                [f"{k}: {v} ({v/len(trigrams)*100:.1f}%)" for k, v in trigrams.most_common(3)] if trigrams else ["No trigrams available"]
            ), unsafe_allow_html=True)

# --- Main ---
def main():
    st.set_page_config(layout="wide", page_title="MANG BACCARAT GROUP")
    apply_css()
    st.title("MANG BACCARAT GROUP")
    initialize_session_state()
    render_setup_form()
    render_result_input()
    render_bead_plate()
    render_prediction()
    render_status()
    render_insights()
    render_history()

if __name__ == "__main__":
    main()
