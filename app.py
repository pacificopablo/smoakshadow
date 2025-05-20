import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random
from collections import defaultdict

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SHOE_SIZE = 100
GRID_ROWS = 6
GRID_COLS = 16
HISTORY_LIMIT = 1000
SEQUENCE_LENGTH = 6
STOP_LOSS_DEFAULT = 0.8  # Default stop loss at 80% of initial bankroll
WIN_LIMIT = 1.5   # Default stop at 150% of initial bankroll (overridden by profit lock threshold)
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
)  # 1 + 3 + 7 + 21 + 50 + 150 + 350 = 537
FLATBET_LEVELUP_TABLE = {
    1: 1,  # Level 1: Bet base_bet * 1
    2: 2,  # Level 2: Bet base_bet * 2
    3: 4,  # Level 3: Bet base_bet * 4
    4: 8,  # Level 4: Bet base_bet * 8
    5: 16  # Level 5: Bet base_bet * 16
}
FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER = sum(
    FLATBET_LEVELUP_TABLE[level] * 5 for level in FLATBET_LEVELUP_TABLE
)  # (1*5 + 2*5 + 4*5 + 8*5 + 16*5) = 5 + 10 + 20 + 40 + 80 = 155
FLATBET_LEVELUP_THRESHOLDS = {
    1: -5.0,   # Move to Level 2 after -5 units net loss at Level 1 (base_bet * 1)
    2: -10.0,  # Move to Level 3 after -10 units net loss at Level 2 (base_bet * 2)
    3: -20.0,  # Move to Level 4 after -20 units net loss at Level 3 (base_bet * 4)
    4: -40.0,  # Move to Level 5 after -40 units net loss at Level 4 (base_bet * 8)
    5: -40.0   # Stay at Level 5 after -40 units net loss (base_bet * 16)
}
MONEY_MANAGEMENT_STRATEGIES = ["T3", "Flatbet", "Parlay16", "Moon", "FourTier", "FlatbetLevelUp"]

# --- CSS for Professional Styling ---
def apply_custom_css():
    try:
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
        .st-expander > div > div {
            background-color: #f7fafc;
            border-radius: 8px;
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
    except Exception as e:
        st.warning(f"Failed to apply custom CSS: {str(e)}")

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
    except PermissionError:
        st.error("Unable to access session file.")
        return 0
    except Exception as e:
        st.error(f"Error in session tracking: {str(e)}")
        return 0

# --- Prediction Logic ---
def generate_baccarat_data(num_games=10000):
    outcomes = ['P', 'B']
    return [random.choice(outcomes) for _ in range(num_games)]

def prepare_data(outcomes, sequence_length=6):
    try:
        le = LabelEncoder()
        encoded_outcomes = le.fit_transform(outcomes)
        X, y = [], []
        for i in range(len(encoded_outcomes) - sequence_length):
            X.append(encoded_outcomes[i:i + sequence_length])
            y.append(encoded_outcomes[i + sequence_length])
        return np.array(X), np.array(y), le
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return np.array([]), np.array([]), None

def train_model():
    try:
        outcomes = generate_baccarat_data()
        X, y, le = prepare_data(outcomes, SEQUENCE_LENGTH)
        if X.size == 0 or y.size == 0 or le is None:
            st.error("Failed to generate training data.")
            return None, None
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, le
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

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
        'model': None,
        'le': None,
        't3_level': 1,
        't3_results': [],
        'money_management': 'T3',
        'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
        'stop_loss_percentage': STOP_LOSS_DEFAULT,
        'win_limit': WIN_LIMIT,
        'shoe_completed': False,
        'safety_net_enabled': True,  # Enabled by default
        'advice': f"Need {SEQUENCE_LENGTH} results",
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
        'safety_net_percentage': 0.02,  # Default 2%
        'smart_skip_enabled': False,  # Disabled by default
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session():
    try:
        setup_values = {
            'initial_bankroll': st.session_state.initial_bankroll,
            'base_bet': st.session_state.base_bet,
            'money_management': st.session_state.money_management,
            'stop_loss_percentage': st.session_state.stop_loss_percentage,
            'safety_net_enabled': st.session_state.safety_net_enabled,
            'safety_net_percentage': st.session_state.safety_net_percentage,
            'smart_skip_enabled': st.session_state.smart_skip_enabled,
            'target_profit_option': st.session_state.target_profit_option,
            'target_profit_percentage': st.session_state.target_profit_percentage,
            'target_profit_units': st.session_state.target_profit_units
        }
        initialize_session_state()
        st.session_state.update({
            'bankroll': setup_values['initial_bankroll'],
            'base_bet': setup_values['base_bet'],
            'initial_bankroll': setup_values['initial_bankroll'],
            'sequence': [],
            'bet_history': [],
            'pending_bet': None,
            'bets_placed': 0,
            'bets_won': 0,
            'model': None,
            'le': None,
            't3_level': 1,
            't3_results': [],
            'money_management': setup_values['money_management'],
            'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
            'stop_loss_percentage': setup_values['stop_loss_percentage'],
            'win_limit': WIN_LIMIT,
            'shoe_completed': False,
            'safety_net_enabled': setup_values['safety_net_enabled'],
            'safety_net_percentage': setup_values['safety_net_percentage'],
            'smart_skip_enabled': setup_values['smart_skip_enabled'],
            'advice': f"Need {SEQUENCE_LENGTH} results",
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
        })
    except Exception as e:
        st.error(f"Error resetting session: {str(e)}")

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
        return 0.0
    except Exception as e:
        st.error(f"Error calculating bet amount: {str(e)}")
        return 0.0

def place_result(result: str):
    try:
        # Check stop loss
        stop_loss_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage
        if stop_loss_triggered and not st.session_state.safety_net_enabled:
            st.session_state.shoe_completed = True
            st.warning(f"Bankroll below {st.session_state.stop_loss_percentage*100:.0f}% of initial bankroll. Session ended. Reset or exit.")
            return

        # Check safety net trigger
        safety_net_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.safety_net_percentage
        if safety_net_triggered and st.session_state.safety_net_enabled:
            st.session_state.shoe_completed = True
            st.info(f"Safety net triggered at {st.session_state.safety_net_percentage*100:.0f}%. Betting at base bet.")

        # Check win limit
        if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
            st.session_state.shoe_completed = True
            st.success(f"Bankroll above {st.session_state.win_limit*100:.0f}% of initial bankroll. Session ended. Reset or exit.")
            return

        # Check target profit
        current_profit = st.session_state.bankroll - st.session_state.initial_bankroll
        if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0:
            if current_profit >= st.session_state.initial_bankroll * st.session_state.target_profit_percentage:
                st.session_state.shoe_completed = True
                st.success(f"Target profit reached: ${current_profit:.2f} ({st.session_state.target_profit_percentage*100:.0f}% of bankroll). Session ended. Reset or exit.")
                return
        elif st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0:
            if current_profit >= st.session_state.target_profit_units:
                st.session_state.shoe_completed = True
                st.success(f"Target profit reached: ${current_profit:.2f} (Target: ${st.session_state.target_profit_units:.2f}). Session ended. Reset or exit.")
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
            'shoe_completed': st.session_state.shoe_completed
        }

        # Update Markov transition counts
        if len(st.session_state.sequence) >= 1:
            prev_result = st.session_state.sequence[-1]
            if result in ['P', 'B'] and prev_result in ['P', 'B']:
                transition = f"{prev_result}{result}"
                st.session_state.transition_counts[transition] += 1

        # Resolve pending bet
        bet_amount = 0
        bet_selection = None
        bet_outcome = None
        if st.session_state.pending_bet and result in ['P', 'B']:
            bet_amount, bet_selection = st.session_state.pending_bet
            st.session_state.bets_placed += 1
            if result == bet_selection:
                if bet_selection == 'B':
                    st.session_state.bankroll += bet_amount * 0.95
                    if st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += (bet_amount * 0.95) / st.session_state.base_bet
                else:
                    st.session_state.bankroll += bet_amount
                    if st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += bet_amount / st.session_state.base_bet
                st.session_state.bets_won += 1
                bet_outcome = 'win'
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
                        pass  # No change on win, net loss updated above
            else:
                st.session_state.bankroll -= bet_amount
                if st.session_state.money_management == 'FlatbetLevelUp':
                    st.session_state.flatbet_levelup_net_loss -= bet_amount / st.session_state.base_bet
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
            if st.session_state.money_management == 'T3' and len(st.session_state.t3_results) == 3 and not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
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
            "Money_Management": st.session_state.money_management,
            "Safety_Net": "On" if st.session_state.safety_net_enabled else "Off",
            "Previous_State": previous_state
        })
        if len(st.session_state.bet_history) > HISTORY_LIMIT:
            st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

        # Prediction logic
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
        if len(valid_sequence) < SEQUENCE_LENGTH:
            st.session_state.pending_bet = None
            st.session_state.advice = f"Need {SEQUENCE_LENGTH - len(valid_sequence)} more Player or Banker results"
        elif len(valid_sequence) >= SEQUENCE_LENGTH and result in ['P', 'B']:
            prediction_sequence = valid_sequence[-SEQUENCE_LENGTH:]
            encoded_input = st.session_state.le.transform(prediction_sequence)
            input_array = np.array([encoded_input])
            prediction_probs = st.session_state.model.predict_proba(input_array)[0]
            predicted_class = np.argmax(prediction_probs)
            predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
            model_confidence = np.max(prediction_probs) * 100

            lb6_selection = None
            lb6_confidence = 0
            sixth_prior = 'N/A'
            if len(valid_sequence) >= 6:
                sixth_prior = valid_sequence[-6]
                outcome_index = st.session_state.le.transform([sixth_prior])[0]
                lb6_confidence = prediction_probs[outcome_index] * 100
                if lb6_confidence > 40 and sixth_prior == predicted_outcome:
                    lb6_selection = sixth_prior

            markov_selection = None
            markov_confidence = 0
            last_outcome = valid_sequence[-1]
            total_from_p = st.session_state.transition_counts['PP'] + st.session_state.transition_counts['PB']
            total_from_b = st.session_state.transition_counts['BP'] + st.session_state.transition_counts['BB']
            if last_outcome == 'P' and total_from_p > 0:
                prob_p_to_p = st.session_state.transition_counts['PP'] / total_from_p
                prob_p_to_b = st.session_state.transition_counts['PB'] / total_from_p
                if prob_p_to_p > prob_p_to_b and prob_p_to_p > 0.5 and 'P' == predicted_outcome:
                    markov_selection = 'P'
                    markov_confidence = prob_p_to_p * 100
                elif prob_p_to_b > prob_p_to_p and prob_p_to_b > 0.5 and 'B' == predicted_outcome:
                    markov_selection = 'B'
                    markov_confidence = prob_p_to_b * 100
            elif last_outcome == 'B' and total_from_b > 0:
                prob_b_to_p = st.session_state.transition_counts['BP'] / total_from_b
                prob_b_to_b = st.session_state.transition_counts['BB'] / total_from_b
                if prob_b_to_p > prob_b_to_b and prob_b_to_p > 0.5 and 'P' == predicted_outcome:
                    markov_selection = 'P'
                    markov_confidence = prob_b_to_p * 100
                elif prob_b_to_b > prob_b_to_b and prob_b_to_b > 0.5 and 'B' == predicted_outcome:
                    markov_selection = 'B'
                    markov_confidence = prob_b_to_b * 100

            strategy_used = None
            bet_selection = None
            confidence = 0.0
            if model_confidence > 50 and (lb6_selection or markov_selection) and not (st.session_state.smart_skip_enabled and confidence < 60):
                bet_selection = predicted_outcome
                confidence = model_confidence
                if lb6_selection and markov_selection:
                    strategy_used = 'Model+LB6+Markov'
                    confidence = max(model_confidence, lb6_confidence, markov_confidence)
                elif lb6_selection:
                    strategy_used = 'Model+LB6'
                    confidence = max(model_confidence, lb6_confidence)
                elif markov_selection:
                    strategy_used = 'Model+Markov'
                    confidence = max(model_confidence, markov_confidence)

            if bet_selection:
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
                    st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%)"
                else:
                    st.session_state.pending_bet = None
                    st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (low confidence or no matching strategy: Model {model_confidence:.1f}% ({predicted_outcome}), LB6 {lb6_confidence:.1f}% ({sixth_prior}), Markov {markov_confidence:.1f}% ({markov_selection if markov_selection else 'N/A'})"

        if len(st.session_state.sequence) >= SHOE_SIZE:
            st.session_state.shoe_completed = True
            if st.session_state.safety_net_enabled and safety_net_triggered:
                st.info(f"Shoe completed, but continuing with safety net at base bet.")
            else:
                st.success(f"Shoe of {SHOE_SIZE} hands completed!")
    except Exception as e:
        st.error(f"Error in place_result: {str(e)}")

# --- UI Components ---
def render_setup_form():
    with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll or 1233.00, step=10.0, help="Your initial total funds. Must be positive and sufficient for the chosen strategy (e.g., FourTier requires at least $537 for a $1 base bet).")
                base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=max(st.session_state.base_bet, 0.10) or 10.00, step=0.10, format="%.2f", help="The smallest bet unit. Should not exceed 5% of your bankroll.")
                money_management = st.selectbox("Strategy", MONEY_MANAGEMENT_STRATEGIES, index=MONEY_MANAGEMENT_STRATEGIES.index(st.session_state.money_management) if st.session_state.money_management in MONEY_MANAGEMENT_STRATEGIES else MONEY_MANAGEMENT_STRATEGIES.index("T3"), help="Choose a betting strategy: T3 adjusts bet levels, Flatbet keeps bets constant, Parlay16 escalates with wins, Moon increases with losses, FourTier uses tiered steps, FlatbetLevelUp adjusts based on net loss.")
            with col2:
                target_mode = st.selectbox("Target Mode", ["Profit %", "Units"], index=["Profit %", "Units"].index(st.session_state.target_profit_option) if st.session_state.target_profit_option in ["Profit %", "Units"] else 0, help="Select how to measure your profit goal: as a percentage of bankroll or in fixed units.")
                if target_mode == "Profit %":
                    target_value_percentage = st.number_input("Target Profit (%)", min_value=0.0, value=st.session_state.target_profit_percentage * 100 if st.session_state.target_profit_percentage > 0 else 6.00, step=0.1, format="%.2f", help="Profit goal as a percentage of your initial bankroll (e.g., 6% of $1000 = $60). Set to 0 to disable.")
                    target_value_units = 0.0  # Reset units if percentage is selected
                else:  # Units
                    target_value_units = st.number_input("Target Profit ($)", min_value=0.0, value=st.session_state.target_profit_units if st.session_state.target_profit_units > 0 else 50.00, step=1.0, format="%.2f", help="Profit goal in fixed dollars (e.g., $50). Set to 0 to disable.")
                    target_value_percentage = 0.0  # Reset percentage if units is selected

            st.markdown('<div class="target-profit-section">', unsafe_allow_html=True)
            st.markdown('<h3><span class="icon">🎯</span>Safety & Limits</h3>', unsafe_allow_html=True)
            safety_net_enabled = st.checkbox("Enable Safety Net", value=True, help="Activates a fallback mode at base bet when bankroll drops to the safety net percentage, allowing continued play.")
            safety_net_percentage = st.number_input("Safety Net Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.safety_net_percentage * 100 or 2.00, step=0.1, disabled=not safety_net_enabled, help="Bankroll threshold (e.g., 2%) to trigger safety net. Must be between 0% and 100%.")

            stop_loss_enabled = st.checkbox("Enable Stop-Loss", value=True, help="Stops the session when bankroll falls below the stop-loss percentage.")
            stop_loss_percentage = st.number_input("Stop-Loss Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.stop_loss_percentage * 100 or 10.00, step=0.1, disabled=not stop_loss_enabled, help="Bankroll percentage (e.g., 10%) to trigger stop-loss. Must be between 0% and 100%.")

            profit_lock_threshold = st.number_input("Profit Lock Threshold (% of Initial Bankroll)", min_value=100.0, max_value=1000.0, value=st.session_state.win_limit * 100 or 600.00, step=1.0, help="Stops the session when bankroll exceeds this percentage of the initial bankroll (e.g., 600% of $1000 = $6000).")

            smart_skip_enabled = st.checkbox("Enable Smart Skip", value=False, help="Skips bets when confidence is low (below 60%) to reduce risk.")

            st.markdown('</div>', unsafe_allow_html=True)

            if st.form_submit_button("Start Session"):
                try:
                    minimum_bankroll = 0
                    if money_management == 'FourTier':
                        minimum_bankroll = base_bet * FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER
                    elif money_management == 'FlatbetLevelUp':
                        minimum_bankroll = base_bet * FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER
                    if bankroll <= 0:
                        st.error("Bankroll must be positive.")
                    elif base_bet < 0.10:
                        st.error("Base bet must be at least $0.10.")
                    elif base_bet > bankroll * 0.05:
                        st.error("Base bet cannot exceed 5% of bankroll.")
                    elif stop_loss_percentage <= 0 or stop_loss_percentage >= 100:
                        st.error("Stop-loss percentage must be between 0% and 100%.")
                    elif safety_net_percentage < 0 or safety_net_percentage >= 100:
                        st.error("Safety net percentage must be between 0% and 100%.")
                    elif profit_lock_threshold <= 100:
                        st.error("Profit lock threshold must be greater than 100% of initial bankroll.")
                    elif money_management == 'FourTier' and bankroll < minimum_bankroll:
                        st.error(f"Four Tier strategy requires a minimum bankroll of ${minimum_bankroll:.2f} for a base bet of ${base_bet:.2f}.")
                    elif money_management == 'FlatbetLevelUp' and bankroll < minimum_bankroll:
                        st.error(f"Flatbet LevelUp strategy requires a minimum bankroll of ${minimum_bankroll:.2f} for a base bet of ${base_bet:.2f}.")
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
                            'model': None,
                            'le': None,
                            't3_level': 1,
                            't3_results': [],
                            'money_management': money_management,
                            'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
                            'stop_loss_percentage': stop_loss_percentage / 100,
                            'win_limit': profit_lock_threshold / 100,
                            'shoe_completed': False,
                            'safety_net_enabled': safety_net_enabled,
                            'safety_net_percentage': safety_net_percentage / 100,
                            'smart_skip_enabled': smart_skip_enabled,
                            'advice': f"Need {SEQUENCE_LENGTH} results",
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
                        })
                        st.session_state.model, st.session_state.le = train_model()
                        if st.session_state.model is None or st.session_state.le is None:
                            st.error("Failed to initialize prediction model.")
                        else:
                            st.success(f"Session started with {money_management} strategy and safety net {'enabled' if safety_net_enabled else 'disabled'}!")
                except Exception as e:
                    st.error(f"Error in form submission: {str(e)}")

def render_result_input():
    with st.expander("Enter Result", expanded=True):
        try:
            if st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
                st.success(f"Shoe of {SHOE_SIZE} hands completed or limits reached!")
            elif st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                st.info("Continuing with safety net at base bet.")
            cols = st.columns(4)
            with cols[0]:
                if st.button("Player", key="player_btn", disabled=(st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0):
                    place_result("P")
                    st.rerun()
            with cols[1]:
                if st.button("Banker", key="banker_btn", disabled=(st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0):
                    place_result("B")
                    st.rerun()
            with cols[2]:
                if st.button("Tie", key="tie_btn", disabled=(st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0):
                    place_result("T")
                    st.rerun()
            with cols[3]:
                if st.button("Undo Last", key="undo_btn", disabled=not st.session_state.bet_history or (st.session_state.shoe_completed and not st.session_state.safety_net_enabled) or st.session_state.bankroll == 0):
                    try:
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
                            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
                            if len(valid_sequence) < SEQUENCE_LENGTH:
                                st.session_state.pending_bet = None
                                st.session_state.advice = f"Need {SEQUENCE_LENGTH - len(valid_sequence)} more Player or Banker results"
                            else:
                                prediction_sequence = valid_sequence[-SEQUENCE_LENGTH:]
                                encoded_input = st.session_state.le.transform(prediction_sequence)
                                input_array = np.array([encoded_input])
                                prediction_probs = st.session_state.model.predict_proba(input_array)[0]
                                predicted_class = np.argmax(prediction_probs)
                                predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
                                model_confidence = np.max(prediction_probs) * 100
                                lb6_selection = None
                                lb6_confidence = 0
                                sixth_prior = 'N/A'
                                if len(valid_sequence) >= 6:
                                    sixth_prior = valid_sequence[-6]
                                    outcome_index = st.session_state.le.transform([sixth_prior])[0]
                                    lb6_confidence = prediction_probs[outcome_index] * 100
                                    if lb6_confidence > 40 and sixth_prior == predicted_outcome:
                                        lb6_selection = sixth_prior
                                markov_selection = None
                                markov_confidence = 0
                                last_outcome = valid_sequence[-1]
                                total_from_p = st.session_state.transition_counts['PP'] + st.session_state.transition_counts['PB']
                                total_from_b = st.session_state.transition_counts['BP'] + st.session_state.transition_counts['BB']
                                if last_outcome == 'P' and total_from_p > 0:
                                    prob_p_to_p = st.session_state.transition_counts['PP'] / total_from_p
                                    prob_p_to_b = st.session_state.transition_counts['PB'] / total_from_p
                                    if prob_p_to_p > prob_p_to_b and prob_p_to_p > 0.5 and 'P' == predicted_outcome:
                                        markov_selection = 'P'
                                        markov_confidence = prob_p_to_p * 100
                                    elif prob_p_to_b > prob_p_to_p and prob_p_to_b > 0.5 and 'B' == predicted_outcome:
                                        markov_selection = 'B'
                                        markov_confidence = prob_p_to_b * 100
                                elif last_outcome == 'B' and total_from_b > 0:
                                    prob_b_to_p = st.session_state.transition_counts['BP'] / total_from_b
                                    prob_b_to_b = st.session_state.transition_counts['BB'] / total_from_b
                                    if prob_b_to_p > prob_b_to_b and prob_b_to_p > 0.5 and 'P' == predicted_outcome:
                                        markov_selection = 'P'
                                        markov_confidence = prob_b_to_p * 100
                                    elif prob_b_to_b > prob_b_to_b and prob_b_to_b > 0.5 and 'B' == predicted_outcome:
                                        markov_selection = 'B'
                                        markov_confidence = prob_b_to_b * 100
                                if model_confidence > 50 and (lb6_selection or markov_selection) and not (st.session_state.smart_skip_enabled and confidence < 60):
                                    bet_selection = predicted_outcome
                                    confidence = model_confidence
                                    if lb6_selection and markov_selection:
                                        strategy_used = 'Model+LB6+Markov'
                                        confidence = max(model_confidence, lb6_confidence, markov_confidence)
                                    elif lb6_selection:
                                        strategy_used = 'Model+LB6'
                                        confidence = max(model_confidence, lb6_confidence)
                                    elif markov_selection:
                                        strategy_used = 'Model+Markov'
                                        confidence = max(model_confidence, markov_confidence)
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
                                        st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%)"
                                    else:
                                        st.session_state.pending_bet = None
                                        st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
                                else:
                                    st.session_state.pending_bet = None
                                    st.session_state.advice = f"Skip betting (low confidence or no matching strategy: Model {model_confidence:.1f}% ({predicted_outcome}), LB6 {lb6_confidence:.1f}% ({sixth_prior}), Markov {markov_confidence:.1f}% ({markov_selection if markov_selection else 'N/A'})"
                            st.success("Undone last action.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error undoing last action: {str(e)}")
            if st.session_state.shoe_completed and st.button("Reset and Start New Shoe", key="new_shoe_btn"):
                reset_session()
                st.session_state.shoe_completed = False
                st.rerun()
        except Exception as e:
            st.error(f"Error rendering result input: {str(e)}")

def render_bead_plate():
    with st.expander("Bead Plate", expanded=True):
        try:
            st.markdown("**Bead Plate**")
            sequence = st.session_state.sequence[-(GRID_ROWS * GRID_COLS):]
            grid = [['' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
            for i, result in enumerate(sequence):
                if result in ['P', 'B', 'T']:
                    col = i // GRID_ROWS
                    row = i % GRID_ROWS
                    if col < GRID_COLS:
                        if result == 'P':
                            color = '#3182ce'
                        elif result == 'B':
                            color = '#e53e3e'
                        else:
                            color = '#38a169'
                        grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'
            for row in grid:
                st.markdown(' '.join(row), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering bead plate: {str(e)}")

def render_prediction():
    with st.expander("Prediction", expanded=True):
        try:
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
                    f"Advice: {advice}</p></div>",
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error rendering prediction: {str(e)}")

def render_status():
    with st.expander("Session Status", expanded=True):
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
                st.markdown(f"**Current Profit**: ${st.session_state.bankroll - st.session_state.initial_bankroll:.2f}")
                st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
                st.markdown(f"**Stop Loss**: {st.session_state.stop_loss_percentage*100:.0f}%")
                target_profit_display = []
                if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0:
                    target_profit_display.append(f"{st.session_state.target_profit_percentage*100:.0f}%")
                elif st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0:
                    target_profit_display.append(f"${st.session_state.target_profit_units:.2f}")
                st.markdown(f"**Target Profit**: {'None' if not target_profit_display else ', '.join(target_profit_display)}")
            with col2:
                st.markdown(f"**Safety Net**: {'On' if st.session_state.safety_net_enabled else 'Off'}")
                st.markdown(f"**Hands Played**: {len(st.session_state.sequence)}")
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
                st.markdown(strategy_status, unsafe_allow_html=True)
                st.markdown(f"**Bets Placed**: {st.session_state.bets_placed}")
                st.markdown(f"**Bets Won**: {st.session_state.bets_won}")
                st.markdown(f"**Online Users**: {track_user_session()}")
        except Exception as e:
            st.error(f"Error rendering status: {str(e)}")

def render_history():
    with st.expander("Bet History", expanded=True):
        try:
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
                        "T3_Level": h["T3_Level"] if h["Money_Management"] == 'T3' else "-",
                        "Parlay_Step": h["Parlay_Step"] if h["Money_Management"] == 'Parlay16' else "-",
                        "Moon_Level": h["Moon_Level"] if h["Money_Management"] == 'Moon' else "-",
                        "FourTier_Level": h["FourTier_Level"] if h["Money_Management"] == 'FourTier' else "-",
                        "FourTier_Step": h["FourTier_Step"] if h["Money_Management"] == 'FourTier' else "-",
                        "FlatbetLevelUp_Level": h["FlatbetLevelUp_Level"] if h["Money_Management"] == 'FlatbetLevelUp' else "-",
                        "FlatbetLevelUp_Net_Loss": h["FlatbetLevelUp_Net_Loss"] if h["Money_Management"] == 'FlatbetLevelUp' else "-",
                        "Safety_Net": h.get("Safety_Net", "N/A")
                    }
                    for h in st.session_state.bet_history[-n:]
                ], use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering history: {str(e)}")

# --- Main Application ---
def main():
    try:
        st.set_page_config(layout="wide", page_title="Mang Baccarat")
        apply_custom_css()
        st.title("Mang Baccarat")
        initialize_session_state()
        st.write(f"Streamlit version: {st.__version__}")
        if st.button("Clear Session State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        col1, col2 = st.columns([2, 1])
        with col1:
            render_setup_form()
            render_result_input()
            render_bead_plate()
            render_prediction()
            render_status()
        with col2:
            render_history()
    except Exception as e:
        st.error(f"Error in main application: {str(e)}")

if __name__ == "__main__":
    main()
