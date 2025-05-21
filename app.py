import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
import time
import random
from collections import defaultdict, Counter
import uuid

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SHOE_SIZE = 100
GRID_ROWS = 6
GRID_COLS = 16
HISTORY_LIMIT = 1000
SEQUENCE_LENGTH = 6
STOP_LOSS_DEFAULT = 0.8  # Default stop loss at 80% of initial bankroll
WIN_LIMIT = 1.5   # Default stop at 150% of initial bankroll
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
)  # (1*5 + 2*5 + 4*5 + 8*5 + 16*5) = 155
FLATBET_LEVELUP_THRESHOLDS = {
    1: -5.0,   # Move to Level 2 after -5 units net loss at Level 1
    2: -10.0,  # Move to Level 3 after -10 units net loss at Level 2
    3: -20.0,  # Move to Level 4 after -20 units net loss at Level 3
    4: -40.0,  # Move to Level 5 after -40 units net loss at Level 4
    5: -40.0   # Stay at Level 5 after -40 units net loss
}
MONEY_MANAGEMENT_STRATEGIES = ["T3", "Flatbet", "Parlay16", "Moon", "FourTier", "FlatbetLevelUp", "D'Alembert", "-1 +2", "Suchi Masterline"]

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
            st.session_state.session_id = str(uuid.uuid4())
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
def predict_next():
    if not st.session_state.sequence:
        return "?", "No history yet."
    if st.session_state.break_countdown > 0:
        return "?", f"In break ({st.session_state.break_countdown} left)."
    last = st.session_state.sequence[-1]
    counts = st.session_state.transitions[last]
    total = sum(counts.values())
    if not counts:
        return "?", f"No data available after '{last}' to predict next."
    probabilities = {k: v / total for k, v in counts.items()}
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    prediction = sorted_probs[0][0]
    explanation = f"Last result: {last}\nTransition probabilities:\n"
    for outcome, prob in sorted_probs:
        explanation += f"  {last} → {outcome}: {prob:.2f}\n"
    return prediction, explanation

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
        'transition_counts': {'PP': 0, 'PB': 0, 'PT': 0, 'BP': 0, 'BB': 0, 'BT': 0, 'TP': 0, 'TB': 0, 'TT': 0},
        'stop_loss_percentage': STOP_LOSS_DEFAULT,
        'win_limit': WIN_LIMIT,
        'shoe_completed': False,
        'safety_net_enabled': True,
        'safety_net_percentage': 0.02,
        'smart_skip_enabled': False,
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
        'session_id': str(uuid.uuid4()),
        'last_prediction': None,
        'explanation': "No explanation available.",
        'masterline_step': 0,
        'in_force2': False,
        'force2_failed': False,
        'break_countdown': 0,
        'transitions': defaultdict(Counter),
        'target_profit_reached': False,
        'session_started': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session():
    try:
        setup_values = {
            'bankroll': st.session_state.bankroll,
            'base_bet': st.session_state.base_bet,
            'initial_bankroll': st.session_state.initial_bankroll,
            'money_management': st.session_state.money_management,
            'stop_loss_percentage': st.session_state.stop_loss_percentage,
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
            'transition_counts': {'PP': 0, 'PB': 0, 'PT': 0, 'BP': 0, 'BB': 0, 'BT': 0, 'TP': 0, 'TB': 0, 'TT': 0},
            'stop_loss_percentage': setup_values['stop_loss_percentage'],
            'win_limit': setup_values['win_limit'],
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
            'session_id': str(uuid.uuid4()),
            'last_prediction': None,
            'explanation': "No explanation available.",
            'masterline_step': 0,
            'in_force2': False,
            'force2_failed': False,
            'break_countdown': 0,
            'transitions': defaultdict(Counter),
            'target_profit_reached': False
        })
    except Exception as e:
        st.error(f"Error resetting session: {str(e)}")

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
        elif st.session_state.money_management == "D'Alembert":
            return st.session_state.base_bet * st.session_state.bet
        elif st.session_state.money_management == '-1 +2':
            return st.session_state.base_bet * st.session_state.bet
        elif st.session_state.money_management == 'Suchi Masterline':
            return st.session_state.base_bet * st.session_state.bet
        return 0.0
    except Exception as e:
        st.error(f"Error calculating bet amount: {str(e)}")
        return 0.0

def handle_masterline(win, result):
    actual_bet = st.session_state.bet * st.session_state.base_bet
    commission = 0.95 if result == "B" else 1.0
    if st.session_state.bankroll < actual_bet:
        st.session_state.pending_bet = None
        st.session_state.advice = f"Skip betting (insufficient bankroll for ${actual_bet:.2f})"
        return
    if st.session_state.in_force2:
        if win:
            st.session_state.bankroll += (2 * st.session_state.base_bet) * commission
            st.session_state.in_force2 = False
            st.session_state.bet = 1
            st.session_state.masterline_step = 0
        else:
            st.session_state.bankroll -= 2 * st.session_state.base_bet
            st.session_state.force2_failed = True
            st.session_state.in_force2 = False
            st.session_state.break_countdown = 3
    elif st.session_state.force2_failed:
        st.session_state.force2_failed = False
        st.session_state.bet = 1
        st.session_state.masterline_step = 0
    elif win:
        ladder = [1, 3, 2, 5]
        st.session_state.bankroll += (ladder[st.session_state.masterline_step] * st.session_state.base_bet) * commission
        st.session_state.masterline_step += 1
        if st.session_state.masterline_step > 3:
            st.session_state.masterline_step = 0
            st.session_state.bet = 1
        else:
            st.session_state.bet = ladder[st.session_state.masterline_step]
    else:
        if st.session_state.masterline_step == 0:
            st.session_state.in_force2 = True
            st.session_state.bet = 2
        else:
            st.session_state.bankroll -= actual_bet
            st.session_state.break_countdown = 3
            st.session_state.masterline_step = 0
            st.session_state.bet = 1

def place_result(result: str):
    try:
        # Check stop loss
        stop_loss_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage
        if stop_loss_triggered and not st.session_state.safety_net_enabled:
            reset_session()
            st.warning(f"Stop-loss triggered at {st.session_state.stop_loss_percentage*100:.0f}% of initial bankroll. Game reset, continue playing.")
            return

        # Check safety net trigger
        safety_net_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.safety_net_percentage
        if safety_net_triggered and st.session_state.safety_net_enabled:
            reset_session()
            st.info(f"Safety net triggered at {st.session_state.safety_net_percentage*100:.0f}%. Game reset, continue playing at base bet.")
            return

        # Check win limit
        if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
            reset_session()
            st.success(f"Win limit reached at {st.session_state.win_limit*100:.0f}% of initial bankroll. Game reset, continue playing.")
            return

        # Check target profit
        current_profit = st.session_state.bankroll - st.session_state.initial_bankroll
        if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0:
            if current_profit >= st.session_state.initial_bankroll * st.session_state.target_profit_percentage:
                reset_session()
                st.session_state.target_profit_reached = True
                st.success(f"Target profit reached: ${current_profit:.2f} ({st.session_state.target_profit_percentage*100:.0f}% of bankroll). Game reset, continue playing.")
                return
        elif st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0:
            if current_profit >= st.session_state.target_profit_units:
                reset_session()
                st.session_state.target_profit_reached = True
                st.success(f"Target profit reached: ${current_profit:.2f} (Target: ${st.session_state.target_profit_units:.2f}). Game reset, continue playing.")
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
            'bet': st.session_state.bet,
            'masterline_step': st.session_state.masterline_step,
            'in_force2': st.session_state.in_force2,
            'force2_failed': st.session_state.force2_failed,
            'break_countdown': st.session_state.break_countdown,
            'last_prediction': st.session_state.last_prediction,
            'explanation': st.session_state.explanation,
            'transitions': st.session_state.transitions.copy(),
            'target_profit_reached': st.session_state.target_profit_reached
        }

        # Update transition counts
        if len(st.session_state.sequence) >= 1 and result in ['P', 'B', 'T']:
            prev_result = st.session_state.sequence[-1]
            if prev_result in ['P', 'B', 'T']:
                transition = f"{prev_result}{result}"
                st.session_state.transition_counts[transition] += 1
                st.session_state.transitions[prev_result][result] += 1

        # Resolve pending bet
        bet_amount = 0
        bet_selection = None
        bet_outcome = None
        if st.session_state.pending_bet and result in ['P', 'B']:
            bet_amount, bet_selection = st.session_state.pending_bet
            actual_bet = bet_amount
            if st.session_state.bankroll < actual_bet:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (insufficient bankroll for ${actual_bet:.2f})"
                return
            st.session_state.bets_placed += 1
            win = (bet_selection == result)
            commission = 0.95 if result == 'B' else 1.0
            if win:
                st.session_state.bankroll += actual_bet * commission
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
                        st.session_state.flatbet_levelup_net_loss += (actual_bet * commission) / st.session_state.base_bet
                    elif st.session_state.money_management == "D'Alembert":
                        st.session_state.bet = max(1, st.session_state.bet - 2)
                    elif st.session_state.money_management == '-1 +2':
                        st.session_state.bet += 2
                    elif st.session_state.money_management == 'Suchi Masterline':
                        handle_masterline(True, result)
            else:
                st.session_state.bankroll -= actual_bet
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
                        st.session_state.flatbet_levelup_net_loss -= actual_bet / st.session_state.base_bet
                        current_level = st.session_state.flatbet_levelup_level
                        if current_level < 5 and st.session_state.flatbet_levelup_net_loss <= FLATBET_LEVELUP_THRESHOLDS[current_level]:
                            st.session_state.flatbet_levelup_level = min(st.session_state.flatbet_levelup_level + 1, 5)
                            st.session_state.flatbet_levelup_net_loss = 0.0
                    elif st.session_state.money_management == "D'Alembert":
                        st.session_state.bet += 1
                    elif st.session_state.money_management == '-1 +2':
                        st.session_state.bet = max(1, st.session_state.bet - 1)
                    elif st.session_state.money_management == 'Suchi Masterline':
                        handle_masterline(False, result)
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
            "D_Alembert_Bet": st.session_state.bet if st.session_state.money_management == "D'Alembert" else "-",
            "Minus1Plus2_Bet": st.session_state.bet if st.session_state.money_management == '-1 +2' else "-",
            "Masterline_Step": st.session_state.masterline_step if st.session_state.money_management == 'Suchi Masterline' else "-",
            "Safety_Net": "On" if st.session_state.safety_net_enabled else "Off",
            "Previous_State": previous_state
        })
        if len(st.session_state.bet_history) > HISTORY_LIMIT:
            st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

        # Prediction logic
        prediction, explanation = predict_next()
        st.session_state.last_prediction = prediction if prediction in ['P', 'B'] else None
        st.session_state.explanation = explanation
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
        if len(valid_sequence) < SEQUENCE_LENGTH or prediction == "?" or st.session_state.break_countdown > 0:
            st.session_state.pending_bet = None
            st.session_state.advice = explanation
        elif prediction in ['P', 'B']:
            counts = st.session_state.transitions[st.session_state.sequence[-1]]
            total = sum(counts.values())
            confidence = (counts[prediction] / total * 100) if total > 0 else 0.0
            if st.session_state.smart_skip_enabled and confidence < 60:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (low confidence: {confidence:.1f}% ({prediction}))"
            else:
                bet_amount = calculate_bet_amount(prediction)
                if bet_amount <= st.session_state.bankroll:
                    st.session_state.pending_bet = (bet_amount, prediction)
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
                    elif st.session_state.money_management == "D'Alembert":
                        strategy_info += f" Bet {st.session_state.bet}x"
                    elif st.session_state.money_management == '-1 +2':
                        strategy_info += f" Bet {st.session_state.bet}x"
                    elif st.session_state.money_management == 'Suchi Masterline':
                        status = "Base" if st.session_state.masterline_step == 0 else f"Step {st.session_state.masterline_step + 1}"
                        if st.session_state.in_force2:
                            status = "Force 2"
                        strategy_info += f" {status}"
                    st.session_state.advice = f"Bet ${bet_amount:.2f} on {prediction} ({strategy_info}, Confidence: {confidence:.1f}%)"
                else:
                    st.session_state.pending_bet = None
                    st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"

        if len(st.session_state.sequence) >= SHOE_SIZE:
            reset_session()
            st.success(f"Shoe of {SHOE_SIZE} hands completed. Game reset, continue playing.")
    except Exception as e:
        st.error(f"Error in place_result: {str(e)}")

# --- UI Components ---
def render_setup_form():
    with st.expander("Session Setup", expanded=not st.session_state.session_started):
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll or 1233.00, step=10.0)
                base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=max(st.session_state.base_bet, 0.10) or 10.00, step=0.10, format="%.2f")
                money_management = st.selectbox("Strategy", MONEY_MANAGEMENT_STRATEGIES, index=MONEY_MANAGEMENT_STRATEGIES.index(st.session_state.money_management) if st.session_state.money_management in MONEY_MANAGEMENT_STRATEGIES else 0)
            with col2:
                target_mode = st.selectbox("Target Mode", ["None", "Profit %", "Units"], index=["None", "Profit %", "Units"].index(st.session_state.target_profit_option) if st.session_state.target_profit_option in ["None", "Profit %", "Units"] else 1)
                if target_mode == "Profit %":
                    target_value_percentage = st.number_input("Target Profit (%)", min_value=0.0, value=st.session_state.target_profit_percentage * 100 if st.session_state.target_profit_percentage > 0 else 6.00, step=0.1, format="%.2f")
                    target_value_units = 0.0
                elif target_mode == "Units":
                    target_value_units = st.number_input("Target Profit ($)", min_value=0.0, value=st.session_state.target_profit_units if st.session_state.target_profit_units > 0 else 50.00, step=1.0, format="%.2f")
                    target_value_percentage = 0.0
                else:
                    target_value_percentage = 0.0
                    target_value_units = 0.0

            st.markdown('<div class="target-profit-section">', unsafe_allow_html=True)
            st.markdown('<h3><span class="icon">🔒</span>Safety & Limits</h3>', unsafe_allow_html=True)
            safety_net_enabled = st.checkbox("Enable Safety Net", value=st.session_state.safety_net_enabled)
            safety_net_percentage = st.number_input("Safety Net Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.safety_net_percentage * 100 or 2.00, step=0.1, disabled=not safety_net_enabled)
            stop_loss_enabled = st.checkbox("Enable Stop-Loss", value=True)
            stop_loss_percentage = st.number_input("Stop-Loss Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.stop_loss_percentage * 100 or 10.00, step=0.1, disabled=not stop_loss_enabled)
            profit_lock_threshold = st.number_input("Profit Lock Threshold (% of Initial Bankroll)", min_value=100.0, max_value=1000.0, value=st.session_state.win_limit * 100 or 600.00, step=1.0)
            smart_skip_enabled = st.checkbox("Enable Smart Skip", value=st.session_state.smart_skip_enabled)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.form_submit_button("Start Session"):
                try:
                    minimum_bankroll = 0
                    if money_management == 'FourTier':
                        minimum_bankroll = base_bet * FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER
                    elif money_management == 'FlatbetLevelUp':
                        minimum_bankroll = base_bet * FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER
                    elif money_management == 'Suchi Masterline':
                        minimum_bankroll = base_bet * 11  # Based on ladder [1, 3, 2, 5] + Force 2
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
                        st.error(f"Four Tier strategy requires a minimum bankroll of ${minimum_bankroll:.2f}.")
                    elif money_management == 'FlatbetLevelUp' and bankroll < minimum_bankroll:
                        st.error(f"Flatbet LevelUp strategy requires a minimum bankroll of ${minimum_bankroll:.2f}.")
                    elif money_management == 'Suchi Masterline' and bankroll < minimum_bankroll:
                        st.error(f"Suchi Masterline strategy requires a minimum bankroll of ${minimum_bankroll:.2f}.")
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
                            'transition_counts': {'PP': 0, 'PB': 0, 'PT': 0, 'BP': 0, 'BB': 0, 'BT': 0, 'TP': 0, 'TB': 0, 'TT': 0},
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
                            'session_id': str(uuid.uuid4()),
                            'last_prediction': None,
                            'explanation': "No explanation available.",
                            'masterline_step': 0,
                            'in_force2': False,
                            'force2_failed': False,
                            'break_countdown': 0,
                            'transitions': defaultdict(Counter),
                            'target_profit_reached': False,
                            'session_started': True,
                            'bet': 1
                        })
                        st.success(f"Session started with {money_management} strategy!")
                except Exception as e:
                    st.error(f"Error in form submission: {str(e)}")

def render_result_input():
    with st.expander("Enter Result", expanded=True):
        try:
            if st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
                st.success(f"Shoe of {SHOE_SIZE} hands completed or limits reached!")
            elif st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                st.info("Continuing with safety net at base bet.")
            elif not st.session_state.session_started:
                st.warning("Please start a session with a valid bankroll and base bet.")
                return
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
                            if len(st.session_state.sequence) >= 1 and last_bet["Result"] in ['P', 'B', 'T'] and st.session_state.sequence[-1] in ['P', 'B', 'T']:
                                transition = f"{st.session_state.sequence[-1]}{last_bet['Result']}"
                                st.session_state.transition_counts[transition] = max(0, st.session_state.transition_counts[transition] - 1)
                                st.session_state.transitions[st.session_state.sequence[-1]][last_bet["Result"]] -= 1
                            prediction, explanation = predict_next()
                            st.session_state.last_prediction = prediction if prediction in ['P', 'B'] else None
                            st.session_state.explanation = explanation
                            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
                            if len(valid_sequence) < SEQUENCE_LENGTH or prediction == "?":
                                st.session_state.pending_bet = None
                                st.session_state.advice = explanation
                            else:
                                counts = st.session_state.transitions[st.session_state.sequence[-1]]
                                total = sum(counts.values())
                                confidence = (counts[prediction] / total * 100) if total > 0 else 0.0
                                if st.session_state.smart_skip_enabled and confidence < 60:
                                    st.session_state.pending_bet = None
                                    st.session_state.advice = f"Skip betting (low confidence: {confidence:.1f}% ({prediction}))"
                                else:
                                    bet_amount = calculate_bet_amount(prediction)
                                    if bet_amount <= st.session_state.bankroll:
                                        st.session_state.pending_bet = (bet_amount, prediction)
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
                                        elif st.session_state.money_management == "D'Alembert":
                                            strategy_info += f" Bet {st.session_state.bet}x"
                                        elif st.session_state.money_management == '-1 +2':
                                            strategy_info += f" Bet {st.session_state.bet}x"
                                        elif st.session_state.money_management == 'Suchi Masterline':
                                            status = "Base" if st.session_state.masterline_step == 0 else f"Step {st.session_state.masterline_step + 1}"
                                            if st.session_state.in_force2:
                                                status = "Force 2"
                                            strategy_info += f" {status}"
                                        st.session_state.advice = f"Bet ${bet_amount:.2f} on {prediction} ({strategy_info}, Confidence: {confidence:.1f}%)"
                                    else:
                                        st.session_state.pending_bet = None
                                        st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
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
            html = """
            <style>
                .bead-plate {
                    background-color: #edf2f7;
                    padding: 10px;
                    border-radius: 8px;
                    overflow-x: auto;
                }
            </style>
            <div class='bead-plate'>
            """
            for row in grid:
                html += '<div>' + ' '.join(row) + '</div>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering bead plate: {str(e)}")

def render_prediction():
    with st.expander("Prediction", expanded=True):
        try:
            if st.session_state.bankroll == 0 or not st.session_state.session_started:
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
                    f"Advice: {advice}</p>"
                    f"<p style='font-size:1rem; margin:5px 0 0 0; color:#2d3748;'>"
                    f"Explanation: {st.session_state.explanation}</p></div>",
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
                valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
                streak_info = "No streak detected"
                if len(valid_sequence) >= 4 and len(set(valid_sequence[-4:])) == 1:
                    streak_length = len([x for x in valid_sequence[::-1] if x == valid_sequence[-1]])
                    streak_info = f"Streak detected: {valid_sequence[-1]} x {streak_length}"
                st.markdown(f"**Streak Status**: {streak_info}")
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
                elif st.session_state.money_management == "D'Alembert":
                    strategy_status += f"<br>**Bet Multiplier**: {st.session_state.bet}x"
                elif st.session_state.money_management == '-1 +2':
                    strategy_status += f"<br>**Bet Multiplier**: {st.session_state.bet}x"
                elif st.session_state.money_management == 'Suchi Masterline':
                    status = "Base" if st.session_state.masterline_step == 0 else f"Ladder Step {st.session_state.masterline_step + 1}"
                    if st.session_state.in_force2:
                        status = "Force 2"
                    elif st.session_state.break_countdown > 0:
                        status = f"Break ({st.session_state.break_countdown} left)"
                    strategy_status += f"<br>**Status**: {status}"
                st.markdown(strategy_status, unsafe_allow_html=True)
                st.markdown(f"**Bets Placed**: {st.session_state.bets_placed}")
                st.markdown(f"**Bets Won**: {st.session_state.bets_won}")
                st.markdown(f"**Online Users**: {track_user_session()}")
                st.markdown(f"**Transition Counts**: PP: {st.session_state.transition_counts['PP']}, "
                            f"PB: {st.session_state.transition_counts['PB']}, PT: {st.session_state.transition_counts['PT']}, "
                            f"BP: {st.session_state.transition_counts['BP']}, BB: {st.session_state.transition_counts['BB']}, "
                            f"BT: {st.session_state.transition_counts['BT']}, TP: {st.session_state.transition_counts['TP']}, "
                            f"TB: {st.session_state.transition_counts['TB']}, TT: {st.session_state.transition_counts['TT']}")
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
                        "Amount": f"${h['Bet_Amount']:.2f}" if h['Bet_Amount'] > 0 else "-",
                        "Outcome": h["Bet_Outcome"] if h["Bet_Outcome"] else "-",
                        "T3_Level": h["T3_Level"],
                        "Parlay_Step": h["Parlay_Step"],
                        "Moon_Level": h["Moon_Level"],
                        "FourTier_Level": h["FourTier_Level"],
                        "FourTier_Step": h["FourTier_Step"],
                        "FlatbetLevelUp_Level": h["FlatbetLevelUp_Level"],
                        "FlatbetLevelUp_Net_Loss": h["FlatbetLevelUp_Net_Loss"],
                        "D_Alembert_Bet": h["D_Alembert_Bet"],
                        "Minus1Plus2_Bet": h["Minus1Plus2_Bet"],
                        "Masterline_Step": h["Masterline_Step"],
                        "Safety_Net": h["Safety_Net"]
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
