import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
import time
import random
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import uuid

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SHOE_SIZE = 100
GRID_ROWS = 6
GRID_COLS = 16
HISTORY_LIMIT = 1000
STOP_LOSS_DEFAULT = 1.0  # 100%
WIN_LIMIT = 1.5
PARLAY_TABLE = {
    i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
        (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
        (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
    ], 1)
}
FOUR_TIER_TABLE = {
    1: {'step1': 1, 'step2': 3},
    2: {'step1': 7, 'step2': 21},
    3: {'step1': 50, 'step2': 150},
    4: {'step1': 350, 'step2': 1050}
}
FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER = sum(
    FOUR_TIER_TABLE[tier][step] for tier in FOUR_TIER_TABLE for step in FOUR_TIER_TABLE[tier]
)
FLATBET_LEVELUP_TABLE = {
    1: 1, 2: 2, 3: 4, 4: 8, 5: 16
}
FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER = sum(
    FLATBET_LEVELUP_TABLE[level] * 5 for level in FLATBET_LEVELUP_TABLE
)
FLATBET_LEVELUP_THRESHOLDS = {
    1: -5.0, 2: -10.0, 3: -20.0, 4: -40.0, 5: -40.0
}
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
GRID_MINIMUM_BANKROLL_MULTIPLIER = max(max(row) for row in GRID) * 5
MONEY_MANAGEMENT_STRATEGIES = ["T3", "Flatbet", "Parlay16", "Moon", "FourTier", "FlatbetLevelUp", "Grid", "OscarGrind"]
BET_SEQUENCE = ['P', 'B', 'P', 'P', 'B', 'B']
OUTCOME_MAPPING = {'P': 0, 'B': 1, 'T': 2}
REVERSE_MAPPING = {0: 'P', 1: 'B', 2: 'T'}

# --- CSS for Professional Styling ---
def apply_custom_css():
    st.markdown("""
    <style>
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f7f9fc;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    h1 {
        color: #1a3c6e;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    h2, .st-emotion-cache-1rtdyac {
        color: #2c5282;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #1a3c6e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2b6cb0;
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    .stNumberInput > div > div > input, .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 10px;
        font-size: 14px;
        width: 100%;
    }
    .stCheckbox > label, .stRadio > label {
        font-size: 14px;
        color: #2d3748;
    }
    .st-expander {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stMarkdown, .stDataFrame {
        font-size: 14px;
        color: #2d3748;
    }
    .result-button-player {
        background: linear-gradient(to bottom, #3182ce, #2b6cb0);
        color: white;
    }
    .result-button-player:hover {
        background: linear-gradient(to bottom, #63b3ed, #3182ce);
    }
    .result-button-banker {
        background: linear-gradient(to bottom, #e53e3e, #c53030);
        color: white;
    }
    .result-button-banker:hover {
        background: linear-gradient(to bottom, #fc8181, #e53e3e);
    }
    .result-button-tie {
        background: linear-gradient(to bottom, #38a169, #2f855a);
        color: white;
    }
    .result-button-tie:hover {
        background: linear-gradient(to bottom, #68d391, #38a169);
    }
    .result-button-undo {
        background: linear-gradient(to bottom, #718096, #5a667f);
        color: white;
    }
    .result-button-undo:hover {
        background: linear-gradient(to bottom, #a0aec0, #718096);
    }
    .bead-plate {
        background-color: #edf2f7;
        padding: 10px;
        border-radius: 8px;
        overflow-x: auto;
    }
    .target-profit-section {
        background-color: #f7fafc;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
    }
    .target-profit-section h3 {
        color: #2c5282;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
    .target-profit-section h3 .icon {
        margin-right: 8px;
        font-size: 1.2rem;
    }
    .target-profit-section .stSelectbox {
        margin-bottom: 10px;
    }
    .target-profit-section .stNumberInput {
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .target-profit-section .stNumberInput label {
        display: flex;
        align-items: center;
        margin-right: 10px;
    }
    .target-profit-section .stNumberInput label .icon {
        margin-right: 5px;
        font-size: 1rem;
        color: #2c5282;
    }
    .pattern-section {
        background-color: #f7fafc;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
    }
    .pattern-section h3 {
        color: #2c5282;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
    .pattern-section h3 .icon {
        margin-right: 8px;
        font-size: 1.2rem;
    }
    @media (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        h1 {
            font-size: 2rem;
        }
        h2, .st-emotion-cache-1rtdyac {
            font-size: 1.25rem;
        }
        .stButton > button {
            width: 100%;
            padding: 12px;
        }
        .stNumberInput, .stSelectbox {
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session Tracking ---
def track_user_session() -> int:
    try:
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(time.time())
        sessions = {}
        current_time = datetime.now()
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        session_id, timestamp = line.strip().split(',')
                        last_seen = datetime.fromisoformat(timestamp)
                        if current_time - last_seen <= timedelta(seconds=30):
                            sessions[session_id] = last_seen
                    except ValueError:
                        continue
        st.session_state.session_id = str(time.time())
        sessions[st.session_state.session_id] = current_time
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
        return len(sessions)
    except:
        return 0

# --- Machine Learning Model ---
def train_ml_model(sequence):
    if len(sequence) < 5:
        return None, None
    X, y = [], []
    for i in range(len(sequence) - 4):
        window = sequence[i:i+4]
        next_outcome = sequence[i+4]
        features = [
            OUTCOME_MAPPING[window[j]] for j in range(len(window))
        ] + [
            st.session_state.transition_counts.get(f"{window[-1]}{k}", 0) / (st.session_state.transition_counts.get(f"{window[-1]}P", 0) + st.session_state.transition_counts.get(f"{window[-1]}B", 0) + 1)
            for k in ['P', 'B']
        ]
        X.append(features)
        y.append(OUTCOME_MAPPING[next_outcome])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler

def predict_next_outcome(sequence, model, scaler):
    if len(sequence) < 4 or model is None:
        return 'P', 0.5  # Default to Player with low confidence
    window = sequence[-4:]
    features = [
        OUTCOME_MAPPING[window[j]] for j in range(len(window))
    ] + [
        st.session_state.transition_counts.get(f"{window[-1]}{k}", 0) / (st.session_state.transition_counts.get(f"{window[-1]}P", 0) + st.session_state.transition_counts.get(f"{window[-1]}B", 0) + 1)
        for k in ['P', 'B']
    ]
    X_scaled = scaler.transform([features])
    probs = model.predict_proba(X_scaled)[0]
    predicted_idx = np.argmax(probs)
    confidence = probs[predicted_idx]
    return REVERSE_MAPPING[predicted_idx], confidence

# --- Session State Management ---
def initialize_session_state():
    defaults = {
        'bankroll': 0.0,
        'base_bet': 0.0,
        'initial_bankroll': 0.0,
        'sequence': [],
        'bet_history': [],
        'pending_bet': None,
        'bets_placed': 0,
        'bets_won': 0,
        't3_level': 1,
        't3_results': [],
        'money_management': 'T3',
        'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
        'stop_loss_percentage': STOP_LOSS_DEFAULT,
        'stop_loss_enabled': True,
        'win_limit': WIN_LIMIT,
        'shoe_completed': False,
        'safety_net_enabled': True,
        'advice': "Need 4 more Player or Banker results for AI prediction",
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'moon_level': 1,
        'moon_level_changes': 0,
        'moon_peak_level': 1,
        'target_profit_option': 'Profit %',
        'target_profit_percentage': 0.0,
        'target_profit_units': 0.0,
        'four_tier_level': 1,
        'four_tier_step': 1,
        'four_tier_losses': 0,
        'flatbet_levelup_level': 1,
        'flatbet_levelup_net_loss': 0.0,
        'safety_net_percentage': 0.02,
        'smart_skip_enabled': False,
        'grid_pos': [0, 0],
        'oscar_cycle_profit': 0.0,
        'oscar_current_bet_level': 1,
        'sequence_bet_index': 0,
        'ml_model': None,
        'ml_scaler': None,
        'ai_mode': False,
        'current_streak': 0,
        'current_streak_type': None,
        'longest_streak': 0,
        'longest_streak_type': None,
        'current_chop_count': 0,
        'longest_chop': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session():
    setup_values = {
        'bankroll': st.session_state.bankroll,
        'base_bet': st.session_state.base_bet,
        'initial_bankroll': st.session_state.initial_bankroll,
        'money_management': st.session_state.money_management,
        'stop_loss_percentage': st.session_state.stop_loss_percentage,
        'stop_loss_enabled': st.session_state.stop_loss_enabled,
        'safety_net_enabled': st.session_state.safety_net_enabled,
        'safety_net_percentage': st.session_state.safety_net_percentage,
        'smart_skip_enabled': st.session_state.smart_skip_enabled,
        'target_profit_option': st.session_state.target_profit_option,
        'target_profit_percentage': st.session_state.target_profit_percentage,
        'target_profit_units': st.session_state.target_profit_units,
        'win_limit': st.session_state.win_limit
    }
    initialize_session_state()
    st.session_state.update({
        'bankroll': setup_values['bankroll'],
        'base_bet': setup_values['base_bet'],
        'initial_bankroll': setup_values['initial_bankroll'],
        'sequence': [],
        'bet_history': [],
        'pending_bet': None,
        'bets_placed': 0,
        'bets_won': 0,
        't3_level': 1,
        't3_results': [],
        'money_management': setup_values['money_management'],
        'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
        'stop_loss_percentage': setup_values['stop_loss_percentage'],
        'stop_loss_enabled': setup_values['stop_loss_enabled'],
        'win_limit': setup_values['win_limit'],
        'shoe_completed': False,
        'safety_net_enabled': setup_values['safety_net_enabled'],
        'safety_net_percentage': setup_values['safety_net_percentage'],
        'smart_skip_enabled': setup_values['smart_skip_enabled'],
        'advice': "Need 4 more Player or Banker results for AI prediction",
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'moon_level': 1,
        'moon_level_changes': 0,
        'moon_peak_level': 1,
        'target_profit_option': setup_values['target_profit_option'],
        'target_profit_percentage': setup_values['target_profit_percentage'],
        'target_profit_units': setup_values['target_profit_units'],
        'four_tier_level': 1,
        'four_tier_step': 1,
        'four_tier_losses': 0,
        'flatbet_levelup_level': 1,
        'flatbet_levelup_net_loss': 0.0,
        'grid_pos': [0, 0],
        'oscar_cycle_profit': 0.0,
        'oscar_current_bet_level': 1,
        'sequence_bet_index': 0,
        'ml_model': None,
        'ml_scaler': None,
        'ai_mode': False,
        'current_streak': 0,
        'current_streak_type': None,
        'longest_streak': 0,
        'longest_streak_type': None,
        'current_chop_count': 0,
        'longest_chop': 0
    })

# --- Betting and Prediction Logic ---
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
        return 0.0
    except:
        return 0.0

def simulate_shoe_result():
    probabilities = {'P': 0.4586, 'B': 0.4460, 'T': 0.0954}
    return random.choices(['P', 'B', 'T'], weights=[probabilities['P'], probabilities['B'], probabilities['T']], k=1)[0]

def place_result(result: str):
    try:
        # Check limits
        if st.session_state.stop_loss_enabled:
            stop_loss_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage
            if stop_loss_triggered and not st.session_state.safety_net_enabled:
                reset_session()
                st.warning(f"Stop-loss triggered at {st.session_state.stop_loss_percentage*100:.0f}% of initial bankroll. Game reset.")
                return

        safety_net_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.safety_net_percentage
        if safety_net_triggered and st.session_state.safety_net_enabled:
            reset_session()
            st.info(f"Safety net triggered at {st.session_state.safety_net_percentage*100:.0f}%. Game reset to base bet.")
        
        if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
            reset_session()
            st.success(f"Win limit reached at {st.session_state.win_limit*100:.0f}% of initial bankroll. Game reset.")
            return

        current_profit = st.session_state.bankroll - st.session_state.initial_bankroll
        if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0:
            if current_profit >= st.session_state.initial_bankroll * st.session_state.target_profit_percentage:
                reset_session()
                st.success(f"Target profit reached: ${current_profit:.2f} ({st.session_state.target_profit_percentage*100:.0f}%). Game reset.")
                return
        elif st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0:
            if current_profit >= st.session_state.target_profit_units:
                reset_session()
                st.success(f"Target profit reached: ${current_profit:.2f} (Target: ${st.session_state.target_profit_units:.2f}). Game reset.")
                return

        # Save previous state for undo
        previous_state = {
            'bankroll': st.session_state.bankroll,
            't3_level': st.session_state.t3_level,
            't3_results': st.session_state.t3_results.copy(),
            'parlay_step': st.session_state.parlay_step,
            'parlay_wins': st.session_state.parlay_wins,
            'parlay_using_base': st.session_state.parlay_using_base,
            'parlay_step_changes': st.session_state.parlay_step_changes,
            'parlay_peak_step': st.session_state.parlay_peak_step,
            'moon_level': st.session_state.moon_level,
            'moon_level_changes': st.session_state.moon_level_changes,
            'moon_peak_level': st.session_state.moon_peak_level,
            'four_tier_level': st.session_state.four_tier_level,
            'four_tier_step': st.session_state.four_tier_step,
            'four_tier_losses': st.session_state.four_tier_losses,
            'flatbet_levelup_level': st.session_state.flatbet_levelup_level,
            'flatbet_levelup_net_loss': st.session_state.flatbet_levelup_net_loss,
            'bets_placed': st.session_state.bets_placed,
            'bets_won': st.session_state.bets_won,
            'transition_counts': st.session_state.transition_counts.copy(),
            'pending_bet': st.session_state.pending_bet,
            'shoe_completed': st.session_state.shoe_completed,
            'grid_pos': st.session_state.grid_pos.copy(),
            'oscar_cycle_profit': st.session_state.oscar_cycle_profit,
            'oscar_current_bet_level': st.session_state.oscar_current_bet_level,
            'sequence_bet_index': st.session_state.sequence_bet_index,
            'current_streak': st.session_state.current_streak,
            'current_streak_type': st.session_state.current_streak_type,
            'longest_streak': st.session_state.longest_streak,
            'longest_streak_type': st.session_state.longest_streak_type,
            'current_chop_count': st.session_state.current_chop_count,
            'longest_chop': st.session_state.longest_chop
        }

        # Update transition counts
        if len(st.session_state.sequence) >= 1 and result in ['P', 'B']:
            prev_result = st.session_state.sequence[-1]
            if prev_result in ['P', 'B']:
                transition = f"{prev_result}{result}"
                st.session_state.transition_counts[transition] += 1

        # Update streak and chop detection
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
        if result in ['P', 'B']:
            # Streak detection
            if len(valid_sequence) == 0 or st.session_state.current_streak_type != result:
                st.session_state.current_streak = 1
                st.session_state.current_streak_type = result
            else:
                st.session_state.current_streak += 1
            if st.session_state.current_streak > st.session_state.longest_streak:
                st.session_state.longest_streak = st.session_state.current_streak
                st.session_state.longest_streak_type = st.session_state.current_streak_type
            
            # Chop detection
            if len(valid_sequence) >= 1:
                last_valid = valid_sequence[-1]
                if result != last_valid:  # Alternation (P to B or B to P)
                    st.session_state.current_chop_count += 1
                else:
                    if st.session_state.current_chop_count > st.session_state.longest_chop:
                        st.session_state.longest_chop = st.session_state.current_chop_count
                    st.session_state.current_chop_count = 0
            else:
                st.session_state.current_chop_count = 0
        else:  # Result is Tie
            # Reset streak and update longest chop if necessary
            if st.session_state.current_chop_count > st.session_state.longest_chop:
                st.session_state.longest_chop = st.session_state.current_chop_count
            st.session_state.current_streak = 0
            st.session_state.current_streak_type = None
            st.session_state.current_chop_count = 0

        # Resolve pending bet
        bet_amount = 0
        bet_selection = None
        bet_outcome = None
        if st.session_state.pending_bet and result in ['P', 'B']:
            bet_amount, bet_selection = st.session_state.pending_bet
            st.session_state.bets_placed += 1
            if result == bet_selection:
                if bet_selection == 'B':
                    winnings = bet_amount * 0.95
                    st.session_state.bankroll += winnings
                    if st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += winnings / st.session_state.base_bet
                    elif st.session_state.money_management == 'OscarGrind':
                        st.session_state.oscar_cycle_profit += winnings
                else:
                    winnings = bet_amount
                    st.session_state.bankroll += winnings
                    if st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += winnings / st.session_state.base_bet
                    elif st.session_state.money_management == 'OscarGrind':
                        st.session_state.oscar_cycle_profit += winnings
                st.session_state.bets_won += 1
                bet_outcome = 'win'
                st.session_state.sequence_bet_index = 0  # Reset sequence on win
                if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                    if st.session_state.money_management == 'T3':
                        if len(st.session_state.t3_results) == 0:
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
                        old_level = st.session_state.moon_level
                        st.session_state.moon_level = old_level
                        if old_level != st.session_state.moon_level:
                            st.session_state.moon_level_changes += 1
                        st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                    elif st.session_state.money_management == 'FourTier':
                        st.session_state.four_tier_level = 1
                        st.session_state.four_tier_step = 1
                        st.session_state.four_tier_losses = 0
                        st.session_state.shoe_completed = True
                        st.session_state.advice = "Win recorded. Reset for a new shoe."
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        pass
                    elif st.session_state.money_management == 'Grid':
                        st.session_state.grid_pos[1] += 1
                        if st.session_state.grid_pos[1] >= len(GRID[0]):
                            st.session_state.grid_pos[1] = 0
                            if st.session_state.grid_pos[0] < len(GRID) - 1:
                                st.session_state.grid_pos[0] += 1
                        if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                            st.session_state.grid_pos = [0, 0]
                    elif st.session_state.money_management == 'OscarGrind':
                        if st.session_state.oscar_cycle_profit >= st.session_state.base_bet:
                            st.session_state.oscar_current_bet_level = 1
                            st.session_state.oscar_cycle_profit = 0.0
                        else:
                            next_bet_level = st.session_state.oscar_current_bet_level + 1
                            potential_winnings = st.session_state.base_bet * next_bet_level * (0.95 if bet_selection == 'B' else 1.0)
                            if st.session_state.oscar_cycle_profit + potential_winnings > st.session_state.base_bet:
                                next_bet_level = max(1, int((st.session_state.base_bet - st.session_state.oscar_cycle_profit) / (st.session_state.base_bet * (0.95 if bet_selection == 'B' else 1.0)) + 0.99))
                            st.session_state.oscar_current_bet_level = next_bet_level
            else:
                st.session_state.bankroll -= bet_amount
                if st.session_state.money_management == 'FlatbetLevelUp':
                    st.session_state.flatbet_levelup_net_loss -= bet_amount / st.session_state.base_bet
                elif st.session_state.money_management == 'OscarGrind':
                    st.session_state.oscar_cycle_profit -= bet_amount
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
                    elif st.session_state.money_management == 'OscarGrind':
                        pass  # Bet level stays the same after a loss
            if st.session_state.money_management == 'T3' and len(st.session_state.t3_results) == 3:
                wins = st.session_state.t3_results.count('W')
                losses = st.session_state.t3_results.count('L')
                if wins > losses:
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                elif losses > wins:
                    st.session_state.t3_level += 1
                st.session_state.t3_results = []
            st.session_state.pending_bet = None

        # Add result to sequence
        if result in ['P', 'B', 'T']:
            st.session_state.sequence.append(result)

        # Train ML model
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
        if len(valid_sequence) >= 5:
            st.session_state.ml_model, st.session_state.ml_scaler = train_ml_model(valid_sequence)

        # Increment sequence_bet_index for P or B results (unless reset by a win)
        if result in ['P', 'B'] and (bet_outcome != 'win'):
            st.session_state.sequence_bet_index += 1

        # Store bet history
        st.session_state.bet_history.append({
            "Result": result,
            "Bet_Amount": bet_amount,
            "Bet_Selection": bet_selection,
            "Bet_Outcome": bet_outcome,
            "T3_Level": st.session_state.t3_level if st.session_state.money_management == 'T3' else "-",
            "Parlay_Step": st.session_state.parlay_step if st.session_state.money_management == 'Parlay16' else "-",
            "Moon_Level": st.session_state.moon_level if st.session_state.money_management == 'Moon' else "-",
            "FourTier_Level": st.session_state.four_tier_level if st.session_state.money_management == 'FourTier' else "-",
            "FourTier_Step": st.session_state.four_tier_step if st.session_state.money_management == 'FourTier' else "-",
            "FlatbetLevelUp_Level": st.session_state.flatbet_levelup_level if st.session_state.money_management == 'FlatbetLevelUp' else "-",
            "FlatbetLevelUp_Net_Loss": round(st.session_state.flatbet_levelup_net_loss, 2) if st.session_state.money_management == 'FlatbetLevelUp' else "-",
            "Grid_Pos": f"({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})" if st.session_state.money_management == 'Grid' else "-",
            "Oscar_Bet_Level": st.session_state.oscar_current_bet_level if st.session_state.money_management == 'OscarGrind' else "-",
            "Oscar_Cycle_Profit": round(st.session_state.oscar_cycle_profit, 2) if st.session_state.money_management == 'OscarGrind' else "-",
            "Sequence_Bet_Index": st.session_state.sequence_bet_index % len(BET_SEQUENCE) if st.session_state.sequence_bet_index > 0 or bet_outcome == 'win' else "-",
            "Money_Management": st.session_state.money_management,
            "Safety_Net": "On" if st.session_state.safety_net_enabled else "Off",
            "Current_Streak": f"{st.session_state.current_streak} ({st.session_state.current_streak_type})" if st.session_state.current_streak_type else "-",
            "Current_Chop": st.session_state.current_chop_count,
            "Previous_State": previous_state
        })
        if len(st.session_state.bet_history) > HISTORY_LIMIT:
            st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

        # Prediction logic (Sequence + ML model + Transition with voting)
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
        if len(valid_sequence) < 4:
            st.session_state.pending_bet = None
            st.session_state.advice = "Need 4 more Player or Banker results for AI prediction"
        elif len(valid_sequence) >= 4 and result in ['P', 'B']:
            # Transition probability prediction
            total_from_p = st.session_state.transition_counts['PP'] + st.session_state.transition_counts['PB']
            total_from_b = st.session_state.transition_counts['BP'] + st.session_state.transition_counts['BB']
            prob_p_to_p = (st.session_state.transition_counts['PP'] / total_from_p) if total_from_p > 0 else 0.5
            prob_p_to_b = (st.session_state.transition_counts['PB'] / total_from_p) if total_from_p > 0 else 0.5
            prob_b_to_p = (st.session_state.transition_counts['BP'] / total_from_b) if total_from_b > 0 else 0.5
            prob_b_to_b = (st.session_state.transition_counts['BB'] / total_from_b) if total_from_b > 0 else 0.5

            last_result = valid_sequence[-1]
            trans_predicted_outcome = None
            trans_confidence = 0.0
            if last_result == 'P':
                trans_predicted_outcome = 'P' if prob_p_to_p >= prob_p_to_b else 'B'
                trans_confidence = max(prob_p_to_p, prob_p_to_b) * 100
            else:  # last_result == 'B'
                trans_predicted_outcome = 'P' if prob_b_to_p >= prob_b_to_b else 'B'
                trans_confidence = max(prob_b_to_p, prob_b_to_b) * 100

            # Sequence prediction
            seq_predicted_outcome = BET_SEQUENCE[st.session_state.sequence_bet_index % len(BET_SEQUENCE)]

            # ML prediction
            ml_predicted_outcome, ml_confidence = predict_next_outcome(valid_sequence, st.session_state.ml_model, st.session_state.ml_scaler)

            # Voting logic
            votes = [ml_predicted_outcome, seq_predicted_outcome, trans_predicted_outcome]
            vote_counts = {'P': 0, 'B': 0, 'T': 0}
            for vote in votes:
                vote_counts[vote] += 1

            # Find the outcome with the most votes
            max_votes = max(vote_counts.values())
            if max_votes >= 2:  # At least two predictions agree
                bet_selection = max(vote_counts, key=vote_counts.get)
                if bet_selection in ['P', 'B']:
                    confidence = ml_confidence * 100 if ml_predicted_outcome == bet_selection else (trans_confidence if trans_predicted_outcome == bet_selection else 50.0)
                    strategy_used = []
                    if ml_predicted_outcome == bet_selection:
                        strategy_used.append('AI')
                    if seq_predicted_outcome == bet_selection:
                        strategy_used.append('Sequence')
                    if trans_predicted_outcome == bet_selection:
                        strategy_used.append('Transition')
                    strategy_used = '+'.join(strategy_used)
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
                            strategy_info += f" Level {st.session_state.flatbet_levelup_level} Net Loss {st.session_state.flatbet_levelup_net_loss:.2f}"
                        elif st.session_state.money_management == 'Grid':
                            strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                        elif st.session_state.money_management == 'OscarGrind':
                            strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level} Cycle Profit ${st.session_state.oscar_cycle_profit:.2f}"
                        st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%, Sequence Pos: {st.session_state.sequence_bet_index % len(BET_SEQUENCE)})"
                    else:
                        st.session_state.pending_bet = None
                        st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
                else:
                    st.session_state.pending_bet = None
                    st.session_state.advice = f"Skip betting (predicted outcome is Tie: AI={ml_predicted_outcome} {ml_confidence*100:.1f}%, Sequence={seq_predicted_outcome}, Transition={trans_predicted_outcome})"
            else:  # No agreement (all predictions differ)
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (no agreement: AI={ml_predicted_outcome} {ml_confidence*100:.1f}%, Sequence={seq_predicted_outcome}, Transition={trans_predicted_outcome})"

        if len(st.session_state.sequence) >= SHOE_SIZE:
            reset_session()
            st.success(f"Shoe of {SHOE_SIZE} hands completed. Game reset.")
    except Exception as e:
        st.error(f"Error processing result: {str(e)}")

def run_simulation():
    for _ in range(SHOE_SIZE - len(st.session_state.sequence)):
        if st.session_state.bankroll <= 0 or (st.session_state.stop_loss_enabled and st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage):
            break
        result = simulate_shoe_result()
        place_result(result)
    st.session_state.ai_mode = False
    st.rerun()

# --- UI Components ---
def render_setup_form():
    with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll or 1233.00, step=10.0)
                base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=max(st.session_state.base_bet, 0.10) or 10.00, step=0.10, format="%.2f")
                money_management = st.selectbox(
                    "Strategy",
                    MONEY_MANAGEMENT_STRATEGIES,
                    index=MONEY_MANAGEMENT_STRATEGIES.index(st.session_state.money_management) if st.session_state.money_management in MONEY_MANAGEMENT_STRATEGIES else 0,
                    help="T3 adjusts bet levels, Flatbet is constant, Parlay16 escalates on wins, Moon increases on losses, FourTier uses tiered steps, FlatbetLevelUp adjusts on net loss, Grid uses a matrix (right on win, down on loss), OscarGrind increases bet after wins to achieve 1-unit profit per cycle."
                )
            with col2:
                target_mode = st.selectbox("Target Mode", ["Profit %", "Units"], index=["Profit %", "Units"].index(st.session_state.target_profit_option) if st.session_state.target_profit_option in ["Profit %", "Units"] else 0)
                if target_mode == "Profit %":
                    target_value_percentage = st.number_input("Target Profit (%)", min_value=0.0, value=st.session_state.target_profit_percentage * 100 or 6.00, step=0.1, format="%.2f")
                    target_value_units = 0.0
                else:
                    target_value_units = st.number_input("Target Profit ($)", min_value=0.0, value=st.session_state.target_profit_units or 50.00, step=1.0, format="%.2f")
                    target_value_percentage = 0.0

            st.markdown('<div class="target-profit-section">', unsafe_allow_html=True)
            st.markdown('<h3><span class="icon">🔒</span>Safety & Limits</h3>', unsafe_allow_html=True)
            safety_net_enabled = st.checkbox("Enable Safety Net", value=True)
            safety_net_percentage = st.number_input("Safety Net Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.safety_net_percentage * 100 or 2.00, step=0.1, disabled=not safety_net_enabled)
            stop_loss_enabled = st.checkbox("Enable Stop-Loss", value=True)
            stop_loss_percentage = st.number_input("Stop-Loss Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.stop_loss_percentage * 100 or 100.00, step=0.1, disabled=not stop_loss_enabled)
            profit_lock_threshold = st.number_input("Profit Lock Threshold (% of Initial Bankroll)", min_value=100.0, max_value=1000.0, value=st.session_state.win_limit * 100 or 600.00, step=1.0)
            smart_skip_enabled = st.checkbox("Enable Smart Skip", value=False)
            ai_mode = st.checkbox("Enable AI Auto-Play", value=False)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.form_submit_button("Start Session"):
                minimum_bankroll = 0
                if money_management == 'FourTier':
                    minimum_bankroll = base_bet * FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER
                elif money_management == 'FlatbetLevelUp':
                    minimum_bankroll = base_bet * FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER
                elif money_management == 'Grid':
                    minimum_bankroll = base_bet * GRID_MINIMUM_BANKROLL_MULTIPLIER
                elif money_management == 'OscarGrind':
                    minimum_bankroll = base_bet * 10
                if bankroll <= 0:
                    st.error("Bankroll must be positive.")
                elif base_bet < 0.10:
                    st.error("Base bet must be at least $0.10.")
                elif base_bet > bankroll * 0.05:
                    st.error("Base bet cannot exceed 5% of bankroll.")
                elif stop_loss_enabled and (stop_loss_percentage < 0 or stop_loss_percentage > 100):
                    st.error("Stop-loss percentage must be between 0% and 100%.")
                elif safety_net_percentage < 0 or safety_net_percentage >= 100:
                    st.error("Safety net percentage must be between 0% and 100%.")
                elif profit_lock_threshold <= 100:
                    st.error("Profit lock threshold must be greater than 100%.")
                elif money_management == 'FourTier' and bankroll < minimum_bankroll:
                    st.error(f"Four Tier requires a minimum bankroll of ${minimum_bankroll:.2f}.")
                elif money_management == 'FlatbetLevelUp' and bankroll < minimum_bankroll:
                    st.error(f"Flatbet LevelUp requires a minimum bankroll of ${minimum_bankroll:.2f}.")
                elif money_management == 'Grid' and bankroll < minimum_bankroll:
                    st.error(f"Grid requires a minimum bankroll of ${minimum_bankroll:.2f}.")
                elif money_management == 'OscarGrind' and bankroll < minimum_bankroll:
                    st.error(f"OscarGrind requires a minimum bankroll of ${minimum_bankroll:.2f}.")
                else:
                    target_profit_percentage = target_value_percentage / 100 if target_mode == "Profit %" else 0.0
                    target_profit_units = target_value_units if target_mode == "Units" else 0.0
                    st.session_state.update({
                        'bankroll': bankroll,
                        'base_bet': base_bet,
                        'initial_bankroll': bankroll,
                        'sequence': [],
                        'bet_history': [],
                        'pending_bet': None,
                        'bets_placed': 0,
                        'bets_won': 0,
                        't3_level': 1,
                        't3_results': [],
                        'money_management': money_management,
                        'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
                        'stop_loss_percentage': stop_loss_percentage / 100,
                        'stop_loss_enabled': stop_loss_enabled,
                        'win_limit': profit_lock_threshold / 100,
                        'shoe_completed': False,
                        'safety_net_enabled': safety_net_enabled,
                        'safety_net_percentage': safety_net_percentage / 100,
                        'smart_skip_enabled': smart_skip_enabled,
                        'advice': "Need 4 more Player or Banker results for AI prediction",
                        'parlay_step': 1,
                        'parlay_wins': 0,
                        'parlay_using_base': True,
                        'parlay_step_changes': 0,
                        'parlay_peak_step': 1,
                        'moon_level': 1,
                        'moon_level_changes': 0,
                        'moon_peak_level': 1,
                        'target_profit_option': target_mode,
                        'target_profit_percentage': target_profit_percentage,
                        'target_profit_units': target_profit_units,
                        'four_tier_level': 1,
                        'four_tier_step': 1,
                        'four_tier_losses': 0,
                        'flatbet_levelup_level': 1,
                        'flatbet_levelup_net_loss': 0.0,
                        'grid_pos': [0, 0],
                        'oscar_cycle_profit': 0.0,
                        'oscar_current_bet_level': 1,
                        'sequence_bet_index': 0,
                        'ml_model': None,
                        'ml_scaler': None,
                        'ai_mode': ai_mode,
                        'current_streak': 0,
                        'current_streak_type': None,
                        'longest_streak': 0,
                        'longest_streak_type': None,
                        'current_chop_count': 0,
                        'longest_chop': 0
                    })
                    st.success(f"Session started with {money_management} strategy! AI Auto-Play: {'On' if ai_mode else 'Off'}")
                    if ai_mode:
                        run_simulation()

def render_result_input():
    with st.expander("Enter Result", expanded=True):
        if st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
            st.success(f"Shoe of {SHOE_SIZE} hands completed or limits reached!")
        elif st.session_state.shoe_completed and st.session_state.safety_net_enabled:
            st.info("Continuing with safety net at base bet.")
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
                    previous_state = last_bet["Previous_State"]
                    for key, value in previous_state.items():
                        st.session_state[key] = value
                    if last_bet["Bet_Amount"] > 0:
                        st.session_state.bets_placed -= 1
                        if last_bet["Bet_Outcome"] == 'win':
                            if last_bet["Bet_Selection"] == 'B':
                                st.session_state.bankroll -= last_bet["Bet_Amount"] * 0.95
                            else:
                                st.session_state.bankroll -= last_bet["Bet_Amount"]
                            st.session_state.bets_won -= 1
                    if len(st.session_state.sequence) >= 1 and last_bet["Result"] in ['P', 'B'] and st.session_state.sequence[-1] in ['P', 'B']:
                        transition = f"{st.session_state.sequence[-1]}{last_bet['Result']}"
                        st.session_state.transition_counts[transition] = max(0, st.session_state.transition_counts[transition] - 1)
                    valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
                    if len(valid_sequence) < 4:
                        st.session_state.pending_bet = None
                        st.session_state.advice = "Need 4 more Player or Banker results for AI prediction"
                    else:
                        # Train ML model
                        st.session_state.ml_model, st.session_state.ml_scaler = train_ml_model(valid_sequence)
                        # Transition probability prediction
                        total_from_p = st.session_state.transition_counts['PP'] + st.session_state.transition_counts['PB']
                        total_from_b = st.session_state.transition_counts['BP'] + st.session_state.transition_counts['BB']
                        prob_p_to_p = (st.session_state.transition_counts['PP'] / total_from_p) if total_from_p > 0 else 0.5
                        prob_p_to_b = (st.session_state.transition_counts['PB'] / total_from_p) if total_from_p > 0 else 0.5
                        prob_b_to_p = (st.session_state.transition_counts['BP'] / total_from_b) if total_from_b > 0 else 0.5
                        prob_b_to_b = (st.session_state.transition_counts['BB'] / total_from_b) if total_from_b > 0 else 0.5

                        last_result = valid_sequence[-1]
                        trans_predicted_outcome = None
                        trans_confidence = 0.0
                        if last_result == 'P':
                            trans_predicted_outcome = 'P' if prob_p_to_p >= prob_p_to_b else 'B'
                            trans_confidence = max(prob_p_to_p, prob_p_to_b) * 100
                        else:  # last_result == 'B'
                            trans_predicted_outcome = 'P' if prob_b_to_p >= prob_b_to_b else 'B'
                            trans_confidence = max(prob_b_to_p, prob_b_to_b) * 100

                        # Sequence prediction
                        seq_predicted_outcome = BET_SEQUENCE[st.session_state.sequence_bet_index % len(BET_SEQUENCE)]

                        # ML prediction
                        ml_predicted_outcome, ml_confidence = predict_next_outcome(valid_sequence, st.session_state.ml_model, st.session_state.ml_scaler)

                        # Voting logic
                        votes = [ml_predicted_outcome, seq_predicted_outcome, trans_predicted_outcome]
                        vote_counts = {'P': 0, 'B': 0, 'T': 0}
                        for vote in votes:
                            vote_counts[vote] += 1

                        # Find the outcome with the most votes
                        max_votes = max(vote_counts.values())
                        if max_votes >= 2:  # At least two predictions agree
                            bet_selection = max(vote_counts, key=vote_counts.get)
                            if bet_selection in ['P', 'B']:
                                confidence = ml_confidence * 100 if ml_predicted_outcome == bet_selection else (trans_confidence if trans_predicted_outcome == bet_selection else 50.0)
                                strategy_used = []
                                if ml_predicted_outcome == bet_selection:
                                    strategy_used.append('AI')
                                if seq_predicted_outcome == bet_selection:
                                    strategy_used.append('Sequence')
                                if trans_predicted_outcome == bet_selection:
                                    strategy_used.append('Transition')
                                strategy_used = '+'.join(strategy_used)
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
                                        strategy_info += f" Level {st.session_state.flatbet_levelup_level} Net Loss {st.session_state.flatbet_levelup_net_loss:.2f}"
                                    elif st.session_state.money_management == 'Grid':
                                        strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                                    elif st.session_state.money_management == 'OscarGrind':
                                        strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level} Cycle Profit ${st.session_state.oscar_cycle_profit:.2f}"
                                    st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%, Sequence Pos: {st.session_state.sequence_bet_index % len(BET_SEQUENCE)})"
                                else:
                                    st.session_state.pending_bet = None
                                    st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
                            else:
                                st.session_state.pending_bet = None
                                st.session_state.advice = f"Skip betting (predicted outcome is Tie: AI={ml_predicted_outcome} {ml_confidence*100:.1f}%, Sequence={seq_predicted_outcome}, Transition={trans_predicted_outcome})"
                        else:  # No agreement (all predictions differ)
                            st.session_state.pending_bet = None
                            st.session_state.advice = f"Skip betting (no agreement: AI={ml_predicted_outcome} {ml_confidence*100:.1f}%, Sequence={seq_predicted_outcome}, Transition={trans_predicted_outcome})"
                    st.success("Undone last action.")
                    st.rerun()
        if st.session_state.shoe_completed and st.button("Reset and Start New Shoe", key="new_shoe_btn"):
            reset_session()
            st.session_state.shoe_completed = False
            st.rerun()

def render_bead_plate():
    with st.expander("Bead Plate", expanded=True):
        st.markdown("**Bead Plate**")
        sequence = st.session_state.sequence[-(GRID_ROWS * GRID_COLS):]
        grid = [['' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        for i, result in enumerate(sequence):
            if result in ['P', 'B', 'T']:
                col = i // GRID_ROWS
                row = i % GRID_ROWS
                if col < GRID_COLS:
                    color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                    grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'
        for row in grid:
            st.markdown(' '.join(row), unsafe_allow_html=True)

def render_prediction():
    with st.expander("Prediction", expanded=True):
        if st.session_state.bankroll == 0:
            st.info("Please start a session with bankroll and base bet.")
        elif st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
            st.info("Session ended. Reset to start a new session.")
        else:
            advice = st.session_state.advice
            text_color = '#2d3748'
            if 'Bet' in advice and ' on P ' in advice:
                text_color = '#3182ce'
            elif 'Bet' in advice and ' on B ' in advice:
                text_color = '#e53e3e'
            st.markdown(
                f"<div style='background-color: #edf2f7; padding: 15px; border-radius: 8px;'>"
                f"<p style='font-size:1.2rem; font-weight:bold; margin:0; color:{text_color};'>"
                f"AI Advice: {advice}</p></div>",
                unsafe_allow_html=True
            )

def render_status():
    with st.expander("Session Status", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
            st.markdown(f"**Current Profit**: ${st.session_state.bankroll - st.session_state.initial_bankroll:.2f}")
            st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
            st.markdown(f"**Stop Loss**: {'Enabled' if st.session_state.stop_loss_enabled else 'Disabled'}, {st.session_state.stop_loss_percentage*100:.0f}%")
            target_profit_display = []
            if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0:
                target_profit_display.append(f"{st.session_state.target_profit_percentage*100:.0f}%")
            elif st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0:
                target_profit_display.append(f"${st.session_state.target_profit_units:.2f}")
            st.markdown(f"**Target Profit**: {'None' if not target_profit_display else ', '.join(target_profit_display)}")
        with col2:
            st.markdown(f"**Safety Net**: {'On' if st.session_state.safety_net_enabled else 'Off'}")
            st.markdown(f"**Hands Played**: {len(st.session_state.sequence)}")
            st.markdown(f"**Sequence Bet Position**: {st.session_state.sequence_bet_index % len(BET_SEQUENCE)}")
            st.markdown(f"**AI Mode**: {'On' if st.session_state.ai_mode else 'Off'}")
            strategy_status = f"**Money Management**: {st.session_state.money_management}"
            if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                strategy_status += "<br>**Mode**: Safety Net (Flatbet)"
            elif st.session_state.money_management == 'T3':
                strategy_status += f"<br>**T3 Level**: {st.session_state.t3_level}<br>**T3 Results**: {st.session_state.t3_results}"
            elif st.session_state.money_management == 'Parlay16':
                strategy_status += f"<br>**Parlay Step**: {st.session_state.parlay_step}/16<br>**Parlay Wins**: {st.session_state.parlay_wins}<br>**Peak Step**: {st.session_state.parlay_peak_step}<br>**Step Changes**: {st.session_state.parlay_step_changes}"
            elif st.session_state.money_management == 'Moon':
                strategy_status += f"<br>**Moon Level**: {st.session_state.moon_level}<br>**Peak Level**: {st.session_state.moon_peak_level}<br>**Level Changes**: {st.session_state.moon_level_changes}"
            elif st.session_state.money_management == 'FourTier':
                strategy_status += f"<br>**FourTier Level**: {st.session_state.four_tier_level}<br>**FourTier Step**: {st.session_state.four_tier_step}<br>**Consecutive Losses**: {st.session_state.four_tier_losses}"
            elif st.session_state.money_management == 'FlatbetLevelUp':
                strategy_status += f"<br>**FlatbetLevelUp Level**: {st.session_state.flatbet_levelup_level}<br>**Net Loss**: {st.session_state.flatbet_levelup_net_loss:.2f}"
            elif st.session_state.money_management == 'Grid':
                strategy_status += f"<br>**Grid Position**: ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
            elif st.session_state.money_management == 'OscarGrind':
                strategy_status += f"<br>**OscarGrind Bet Level**: {st.session_state.oscar_current_bet_level}<br>**Cycle Profit**: ${st.session_state.oscar_cycle_profit:.2f}"
            st.markdown(strategy_status, unsafe_allow_html=True)
            st.markdown(f"**Bets Placed**: {st.session_state.bets_placed}")
            st.markdown(f"**Bets Won**: {st.session_state.bets_won}")
            st.markdown(f"**Online Users**: {track_user_session()}")
            # Transition probabilities
            total_from_p = st.session_state.transition_counts['PP'] + st.session_state.transition_counts['PB']
            total_from_b = st.session_state.transition_counts['BP'] + st.session_state.transition_counts['BB']
            prob_p_to_p = (st.session_state.transition_counts['PP'] / total_from_p * 100) if total_from_p > 0 else 0.0
            prob_p_to_b = (st.session_state.transition_counts['PB'] / total_from_p * 100) if total_from_p > 0 else 0.0
            prob_b_to_p = (st.session_state.transition_counts['BP'] / total_from_b * 100) if total_from_b > 0 else 0.0
            prob_b_to_b = (st.session_state.transition_counts['BB'] / total_from_b * 100) if total_from_b > 0 else 0.0
            st.markdown(
                f"**Transition Probabilities**:<br>"
                f"P→P: {prob_p_to_p:.1f}%, P→B: {prob_p_to_b:.1f}%<br>"
                f"B→P: {prob_b_to_p:.1f}%, B→B: {prob_b_to_b:.1f}%",
                unsafe_allow_html=True
            )
            # Pattern Detection Section
            st.markdown('<div class="pattern-section">', unsafe_allow_html=True)
            st.markdown('<h3><span class="icon">📈</span>Pattern Detection</h3>', unsafe_allow_html=True)
            st.markdown(
                f"**Current Streak**: {st.session_state.current_streak} ({st.session_state.current_streak_type or 'None'})<br>"
                f"**Longest Streak**: {st.session_state.longest_streak} ({st.session_state.longest_streak_type or 'None'})<br>"
                f"**Current Chop**: {st.session_state.current_chop_count}<br>"
                f"**Longest Chop**: {st.session_state.longest_chop}",
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

def render_history():
    with st.expander("Bet History", expanded=True):
        if not st.session_state.bet_history:
            st.write("No history available.")
        else:
            n = st.slider("Show last N bets", 5, 50, 10)
            st.dataframe([
                {
                    "Result": h["Result"],
                    "Bet": h["Bet_Selection"] if h["Bet_Selection"] else "-",
                    "Amount": f"${h['Bet_Amount']:.2f}" if h["Bet_Amount"] > 0 else "-",
                    "Outcome": h["Bet_Outcome"] if h["Bet_Outcome"] else "-",
                    "T3_Level": h["T3_Level"],
                    "Parlay_Step": h["Parlay_Step"],
                    "Moon_Level": h["Moon_Level"],
                    "FourTier_Level": h["FourTier_Level"],
                    "FourTier_Step": h["FourTier_Step"],
                    "FlatbetLevelUp_Level": h["FlatbetLevelUp_Level"],
                    "FlatbetLevelUp_Net_Loss": h["FlatbetLevelUp_Net_Loss"],
                    "Grid_Pos": h["Grid_Pos"],
                    "Oscar_Bet_Level": h["Oscar_Bet_Level"],
                    "Oscar_Cycle_Profit": h["Oscar_Cycle_Profit"],
                    "Sequence_Bet_Index": h["Sequence_Bet_Index"],
                    "Safety_Net": h["Safety_Net"],
                    "Current_Streak": h["Current_Streak"],
                    "Current_Chop": h["Current_Chop"]
                }
                for h in st.session_state.bet_history[-n:]
            ], use_container_width=True)

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="AI Baccarat")
    apply_custom_css()
    st.title("AI Baccarat")
    initialize_session_state()
    render_setup_form()
    render_result_input()
    render_bead_plate()
    render_prediction()
    render_status()
    render_history()

if __name__ == "__main__":
    main()
