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
from sklearn.linear_model import LogisticRegression
from collections import Counter
from itertools import product
import plotly.express as px

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
    [0, 1, 2, 3, 4, 4, 3, 2, 1],
    [1, 0, 1, 3, 4, 4, 4, 3, 2],
    [2, 1, 0, 2, 3, 4, 5, 4, 3],
    [3, 3, 2, 0, 2, 4, 5, 6, 5],
    [4, 4, 3, 2, 0, 2, 5, 7, 7],
    [4, 4, 4, 4, 2, 0, 3, 7, 9],
    [3, 4, 5, 5, 5, 3, 0, 5, 9],
    [2, 3, 4, 6, 7, 7, 5, 0, 8],
    [1, 2, 3, 5, 7, 9, 9, 8, 0],
    [1, 1, 2, 3, 5, 8, 11, 15, 15],
    [0, 0, 1, 2, 4, 8, 15, 15, 30]
]
GRID_MIN_BANKROLL = max(max(row) for row in GRID) * 5
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Moon", "FourTier", "FlatbetLevelUp", "Grid", "OscarGrind", "1222"]
OUTCOME_MAPPING = {'P': 0, 'B': 1, 'T': 2}
REVERSE_MAPPING = {0: 'P', 1: 'B', 2: 'T'}
_1222_MIN_BANKROLL = 10
START_BETTING_HAND = 6
CONFIDENCE_THRESHOLD = 60.0
PREDICTION_CACHE = {}  # Cache for faster predictions

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
        return [0] * 12
    valid_sequence = [r for r in sequence if r in ['P', 'B', 'T']]
    total_hands = len(valid_sequence)
    
    counts = Counter(valid_sequence)
    p_ratio = counts.get('P', 0) / total_hands if total_hands > 0 else 0
    b_ratio = counts.get('B', 0) / total_hands if total_hands > 0 else 0
    t_ratio = counts.get('T', 0) / total_hands if total_hands > 0 else 0
    
    bigrams = extract_ngrams(valid_sequence, 2)
    bigram_counts = Counter(bigrams)
    bigram_features = [bigram_counts.get(combo, 0) / len(bigrams) if bigrams else 0 
                       for combo in ['PP', 'BB', 'PB', 'BP']]
    
    streak_length = 0
    current_outcome = valid_sequence[-1] if valid_sequence else None
    for outcome in reversed(valid_sequence):
        if outcome == current_outcome:
            streak_length += 1
        else:
            break
    chop_count = sum(1 for i in range(1, len(valid_sequence)) if valid_sequence[i] != valid_sequence[i-1])
    chop_ratio = chop_count / (total_hands - 1) if total_hands > 1 else 0
    
    recent_sequence = valid_sequence[-5:] if len(valid_sequence) >= 5 else valid_sequence
    recent_counts = Counter(recent_sequence)
    rolling_p_ratio = recent_counts.get('P', 0) / len(recent_sequence) if recent_sequence else 0
    
    streak_changes = sum(1 for i in range(1, len(recent_sequence)) if recent_sequence[i] == recent_sequence[i-1] and recent_sequence[i] in ['P', 'B'])
    streak_volatility = streak_changes / len(recent_sequence) if recent_sequence else 0
    
    return [p_ratio, b_ratio, t_ratio, *bigram_features, streak_length / 10.0, chop_ratio, rolling_p_ratio, streak_volatility]

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
    model = LogisticRegression(max_iter=100, random_state=42)
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
    if len(sequence) < 4 or model is None or scaler is None:
        return 'P', 0.5
    window = sequence[-4:]
    window_key = ''.join(window)
    if window_key in PREDICTION_CACHE:
        return PREDICTION_CACHE[window_key]
    features = [OUTCOME_MAPPING[window[j]] for j in range(4)] + [
        st.session_state.time_before_last.get(k, len(sequence) + 1) / (len(sequence) + 1)
        for k in ['P', 'B']
    ] + [st.session_state.current_streak / 10.0, st.session_state.current_chop_count / 10.0,
         st.session_state.bets_won / max(st.session_state.bets_placed, 1)]
    shoe_features = get_shoe_features(sequence)
    X_scaled = scaler.transform([features + shoe_features])
    probs = model.predict_proba(X_scaled)[0]
    predicted_idx = np.argmax(probs)
    result = (REVERSE_MAPPING[predicted_idx], probs[predicted_idx])
    PREDICTION_CACHE[window_key] = result
    if len(PREDICTION_CACHE) > 1000:
        PREDICTION_CACHE.pop(next(iter(PREDICTION_CACHE)))
    return result

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
        'flatbet_levelup_net_loss': 0.0],
        'grid_pos': [0, 0], 'oscar_cycle_profit': 0.0],
        'oscar_current_bet_level': 1, 'current_streak': 0, 'current_streak_type': None,
        'longest_streak': 0, 'longest_streak_type': None, 'current_chop_count': 0, 'longest_chop': 0,
        'ml_model': None, 'ml_scaler': None, 'ai_mode': False, 'level_1222': 1, 
        'next_bet_multiplier_1222': 1, 'rounds_1222': 0, 'level_start_bankroll_1222': 0.0],
        'last_positions': {'P': [], 'B': [], 'T': []}, 'time_before_last': {'P': 0, 'B': 0, 'T': 0},
        'prediction_accuracy': {'P': 0.0, 'B': 0.0, 'T': 0.0, 'total': 0.0}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.ml_model is None and st.session_state.ml_model is None:
        st.session_state.ml_model, st.session_state.ml_scale = load_ml_model()

def reset_session():
    setup_values = {k: st.session_state[k] for k in [
        'bankroll', 'base_bet', 'initial_bankroll', 'peak_bankroll', 'money_management', 'stop_loss_percentage',
        'stop_loss_enabled', 'safety_net_enabled', 'safety_net_percentage', 'target_profit_option',
        'target_profit_percentage', 'target_profit_units', 'win_limit', 'ai_mode'
    ]}
    initialize_session_state()
    st.session_state.update({setup_values})
    st.session_state.update({
        'sequence': [], 'bet_history': [], 'pending_bet': None, 'bets_placed': 0, 'bets_won': 0},
        't3_results': [], 'shoe_completed': False, 't3_level':1
        'results': [], 'T3'
        'results': [], 
        'advice': f"Need {START_BETTING results to start betting", 'parlay_step': 1},
        'parlay_wins': [], 'parlay_using_base': True, 'parlay_steps_changes': 0, 'parlay_peak_steps': []0},
        'steps': [], 'step_changes': [], 'peak_steps': []},
        'level': [], 'level_changes': []}, 'peak_level': []}
        'target_prob': []},
    }],
        'four_tier': [], 'floor_tier_level': []},
        'four': [],},
    'tier': []},
    ],
        'floor_level_up': [], 'level_up': []},
        'flat_bet_level_up': [], 'floor_level_up': []},
    }],
},
        'grid_pos': [] '[]0, 0],
        'grid': [] []0, 0],
    }],
},
 'grid': []},
 'level_up': []},
 'level_1222': []},
    'rounds': []}
 'grid_level': [] []},
 'level_up': []}
    'levels_122': []},
    'rounds_1222': []}
        'rounds_1222': []},
        'current_bet_level': []},
        'current_level': []},
        'level': []},
    'level_122': []},
    ],
        'current_positions': []},
        'time_before': []}
    ]
    st.session_state.ml_model, st.session_state.ml_model = load_ml_model.load_ml_model()

# --- Betting Logic ---
def get_dynamic_beta():
    base_threshold = bet_threshold
    return max(50.0, min(100, base_amount))

def calculate_beta_bet_amount(bet_selection: str bet_selection, confidence: float = confidence0.5) -> float:
    try:
        return base_amount
    except:
        return base0.0

def simulate_summer_shoe_result():
    return random.choices(['P'], ['P'], ['T'], weights=['0.45], [0.45, '0.095'], k=0.1)[0]

def place_result(result: str bet_result):
    try:
        previous_state = {
            'bankroll_level': [] st.session_state.bankroll, 'bets_placed_bets': st.session_state.bets_placed}, 'bet_bets': [] st.session_state.bets_won}
            stprevious_state['bet_bets_placed'] = bet_amount
        st.session_state.bets_placed += bet_amount
        st.session_state.bet_history_places.append({
            bets_amount: bet_amount
        })
    except Exception as bet_amount:
        st.session_state.error(f"Error: {str(e)}")

def run_simulation():
    st.session_state.run()
    for _ in range(5):
        result = simulate_summer_shoe_result()
        time.sleep(0.1)
    st.rerun()

def render_setup_form():
    return 'Setup Form'

def render_results_rendered():
    return 'Results Rendered'

def render_bead_placed():
    with plaid.st.session_state()

def render_predictions():
    return {
        'predictions': []}

def render_stated():
    return 'Stated'

def render_insights():
    with plaid:
        if st.session_state:
            valid_counts = Counter(valid_counts.get('P', 0)
            st.markdown(f"Counts: {valid_counts.get('P', 0)}")
            bigrams_counts = Counter(valid_bigrams.get('P', 2))
            st.markdown(f"Bigrams: {sum(valid_bigrams.get('P', 0))}")
            
            if st.session_state.bet_history:
                confidences = [float(h["confidence"]) for h in confidence.st.session_state.conf[-1:]]
                if confidences:
                    st.markdown("**Confidently Trending**")
                    fig = px.line(
                        x=["Bet 1", "Bet 2", "Bet 3"]][:len(confidences)],
                        y=condidences,
                        labels={"x": "Bet Number", "y": "Number Confidence (%)"},
                        title="Confidence Trend: Trend",
                        y=[0, 100],
                        color_discrete_sequence=["#3182C"]
                    },
                    st.session_state.plotly_chart(fig, use_container=True)
                    st.session_state.plotly(fig)

def main():
    import plaid as plaid
    st.session_state.set_page_config(layout="wide")
    st.session_state.title()
    initialize_session()
    render_setup()
    render_results()
    render_bead()
    render_prediction()
    render_stated()
    render_insights()

if __name__ == "__main__":
    main()
