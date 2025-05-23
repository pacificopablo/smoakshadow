import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
import time
import random

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SHOE_SIZE = 80
GRID_ROWS = 6
GRID_COLS = 16
HISTORY_LIMIT = 50
SEQUENCE_LENGTH = 6
STOP_LOSS_DEFAULT = 0.0
WIN_LIMIT = 6.0
PARLAY_SEQUENCE = [1, 1, 1, 2, 3, 4, 6, 8, 12, 16, 22, 30, 40, 52, 70, 95]
FOUR_TIER_BETTING = {1: [1, 1], 2: [2, 2], 3: [5, 5], 4: [10, 10]}
FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER = 50
FLATBET_LEVELUP_BETTING = {1: 1, 2: 2, 3: 3, 4: 5, 5: 10}
FLATBET_LEVELUP_THRESHOLDS = {1: -10, 2: -20, 3: -30, 4: -50, 5: -100}
FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER = 50
GRID_MINIMUM_BANKROLL_MULTIPLIER = 50
GRID = [
    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 0],
    [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0],
    [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 0],
    [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
MONEY_MANAGEMENT_STRATEGIES = ['T3', 'Flatbet', 'Parlay16', 'Moon', 'FourTier', 'FlatbetLevelUp', 'Grid', 'OscarGrind']

# --- CSS for Professional Styling ---
def apply_custom_css():
    st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f7fafc;
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
        background-color: #3182ce;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border: none;
        transition: background-color 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2b6cb0;
    }
    .stButton > button:disabled {
        background-color: #e2e8f0;
        color: #a0aec0;
    }
    .stButton > button[key="player_btn"] {
        background-color: #3182ce;
    }
    .stButton > button[key="banker_btn"] {
        background-color: #e53e3e;
    }
    .stButton > button[key="banker_btn"]:hover {
        background-color: #c53030;
    }
    .stButton > button[key="tie_btn"] {
        background-color: #38a169;
    }
    .stButton > button[key="tie_btn"]:hover {
        background-color: #2f855a;
    }
    .stButton > button[key="undo_btn"] {
        background-color: #a0aec0;
    }
    .stButton > button[key="undo_btn"]:hover {
        background-color: #718096;
    }
    .stButton > button[key="new_shoe_btn"] {
        background-color: #ed8936;
    }
    .stButton > button[key="new_shoe_btn"]:hover {
        background-color: #dd6b20;
    }
    .stNumberInput input, .stSelectbox, .stCheckbox {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.5rem;
    }
    .stNumberInput label, .stSelectbox label, .stCheckbox label {
        color: #4a5568;
        font-weight: 500;
    }
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background-color: #ffffff;
        margin-bottom: 1rem;
    }
    .stExpander summary {
        background-color: #edf2f7;
        padding: 0.75rem;
        border-radius: 8px 8px 0 0;
    }
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    .stSlider > div > div > div > div {
        background-color: #3182ce;
    }
    .stSlider > div > div > div > div > div {
        border-color: #3182ce;
    }
    .target-profit-section {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f7fafc;
        margin-top: 1rem;
    }
    .target-profit-section h3 {
        margin: 0 0 0.5rem 0;
        color: #2d3748;
        display: flex;
        align-items: center;
    }
    .target-profit-section .icon {
        margin-right: 0.5rem;
        font-size: 1.2rem;
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
    except PermissionError:
        st.error("Unable to access session file.")
        return 0
    except Exception as e:
        st.error(f"Error in session tracking: {str(e)}")
        return 0

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
        'money_management': 'Flatbet',
        'stop_loss_percentage': STOP_LOSS_DEFAULT,
        'stop_loss_enabled': True,
        'win_limit': WIN_LIMIT,
        'shoe_completed': False,
        'safety_net_enabled': True,
        'safety_net_percentage': 0.02,
        'smart_skip_enabled': False,
        'advice': "Enter a result to start AI betting",
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
        'grid_pos': [0, 0],
        'oscar_cycle_profit': 0.0,
        'oscar_current_bet_level': 1
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if 'model' not in st.session_state or 'le' not in st.session_state:
        st.session_state.model, st.session_state.le = None, None

def reset_session():
    setup_values = {
        'bankroll': st.session_state.bankroll,
        'base_bet': st.session_state.base_bet,
        'initial_bankroll': st.session_state.initial_bankroll,
        'stop_loss_percentage': st.session_state.stop_loss_percentage,
        'stop_loss_enabled': st.session_state.stop_loss_enabled,
        'safety_net_enabled': st.session_state.safety_net_enabled,
        'safety_net_percentage': st.session_state.safety_net_percentage,
        'smart_skip_enabled': st.session_state.smart_skip_enabled,
        'target_profit_option': st.session_state.target_profit_option,
        'target_profit_percentage': st.session_state.target_profit_percentage,
        'target_profit_units': st.session_state.target_profit_units,
        'win_limit': st.session_state.win_limit,
        'money_management': st.session_state.money_management
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
        'stop_loss_percentage': setup_values['stop_loss_percentage'],
        'stop_loss_enabled': setup_values['stop_loss_enabled'],
        'win_limit': setup_values['win_limit'],
        'shoe_completed': False,
        'safety_net_enabled': setup_values['safety_net_enabled'],
        'safety_net_percentage': setup_values['safety_net_percentage'],
        'smart_skip_enabled': setup_values['smart_skip_enabled'],
        'advice': "Enter a result to start AI betting",
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
        'oscar_current_bet_level': 1
    })

# --- Betting and Prediction Logic ---
def calculate_bet_amount(bet_selection: str) -> float:
    try:
        base_bet = st.session_state.base_bet
        if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
            return base_bet
        if st.session_state.money_management == 'T3':
            return base_bet * st.session_state.t3_level
        elif st.session_state.money_management == 'Parlay16':
            if st.session_state.parlay_using_base:
                return base_bet * PARLAY_SEQUENCE[st.session_state.parlay_step - 1]
            return base_bet
        elif st.session_state.money_management == 'Moon':
            return base_bet * st.session_state.moon_level
        elif st.session_state.money_management == 'FourTier':
            return base_bet * FOUR_TIER_BETTING[st.session_state.four_tier_level][st.session_state.four_tier_step - 1]
        elif st.session_state.money_management == 'FlatbetLevelUp':
            return base_bet * FLATBET_LEVELUP_BETTING[st.session_state.flatbet_levelup_level]
        elif st.session_state.money_management == 'Grid':
            return base_bet * GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]]
        elif st.session_state.money_management == 'OscarGrind':
            return base_bet * st.session_state.oscar_current_bet_level
        return base_bet
    except Exception as e:
        st.error(f"Error calculating bet amount: {str(e)}")
        return 0.0

def place_result(result: str):
    try:
        # Check stop loss
        if st.session_state.stop_loss_enabled:
            stop_loss_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage
            if stop_loss_triggered and not st.session_state.safety_net_enabled:
                reset_session()
                st.warning(f"Stop-loss triggered at {st.session_state.stop_loss_percentage*100:.0f}% of initial bankroll. Game reset.")
                return

        # Check safety net trigger
        safety_net_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.safety_net_percentage
        if safety_net_triggered and st.session_state.safety_net_enabled:
            reset_session()
            st.info(f"Safety net triggered at {st.session_state.safety_net_percentage*100:.0f}%. Game reset to base bet.")
        
        # Check win limit
        if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
            reset_session()
            st.success(f"Win limit reached at {st.session_state.win_limit*100:.0f}% of initial bankroll. Game reset.")
            return

        # Check target profit
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

        # Check bankroll requirements for strategies
        min_bankroll_requirements = {
            'FourTier': FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER,
            'FlatbetLevelUp': FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER,
            'Grid': GRID_MINIMUM_BANKROLL_MULTIPLIER,
            'OscarGrind': 10,
            'T3': 5,
            'Parlay16': 95,
            'Moon': 10,
            'Flatbet': 1
        }
        for strategy, multiplier in min_bankroll_requirements.items():
            if st.session_state.bankroll < st.session_state.base_bet * multiplier:
                if st.session_state.money_management == strategy:
                    st.session_state.money_management = 'Flatbet'

        # AI-driven strategy selection
        profit_ratio = current_profit / st.session_state.initial_bankroll if st.session_state.initial_bankroll > 0 else 0
        shoe_progress = len(st.session_state.sequence) / SHOE_SIZE
        if shoe_progress > 0.75 and st.session_state.safety_net_enabled:
            st.session_state.money_management = 'Flatbet'
        elif abs(profit_ratio) < 0.1:
            st.session_state.money_management = 'Flatbet' if random.random() < 0.5 else 'OscarGrind'
        else:
            aggressive_strategies = ['T3', 'Parlay16', 'Moon']
            st.session_state.money_management = random.choice(aggressive_strategies)

        # Save previous state for undo
        previous_state = {
            'bankroll': st.session_state.bankroll,
            'base_bet': st.session_state.base_bet,
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
            'pending_bet': st.session_state.pending_bet,
            'shoe_completed': st.session_state.shoe_completed,
            'grid_pos': st.session_state.grid_pos.copy(),
            'oscar_cycle_profit': st.session_state.oscar_cycle_profit,
            'oscar_current_bet_level': st.session_state.oscar_current_bet_level,
            'money_management': st.session_state.money_management
        }

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
                        pass
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
            "Safety_Net": "On" if st.session_state.safety_net_enabled else "Off",
            "Previous_State": previous_state,
            "Money_Management": st.session_state.money_management
        })
        if len(st.session_state.bet_history) > HISTORY_LIMIT:
            st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

        # AI-driven bet selection
        if result in ['P', 'B']:
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']][-6:]
            p_count = valid_sequence.count('P')
            total = len(valid_sequence)
            p_prob = p_count / total if total > 0 else 0.5
            b_prob = 1 - p_prob
            streak = False
            if len(valid_sequence) >= 3 and len(set(valid_sequence[-3:])) == 1:
                streak = True
                if valid_sequence[-1] == 'P':
                    p_prob += 0.2
                    b_prob -= 0.2
                else:
                    b_prob += 0.2
                    p_prob -= 0.2
            p_prob = max(0, min(1, p_prob + random.uniform(-0.1, 0.1)))
            b_prob = 1 - p_prob
            bet_selection = 'P' if random.random() < p_prob else 'B'
            rationale = f"AI Probability: P {p_prob*100:.0f}%, B {b_prob*100:.0f}%"
            if streak:
                rationale += f", Streak of {valid_sequence[-1]}"
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
                st.session_state.advice = f"AI Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {rationale})"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"AI Skipped betting (bet ${bet_amount:.2f} exceeds bankroll)"

        if len(st.session_state.sequence) >= SHOE_SIZE:
            reset_session()
            st.success(f"Shoe of {SHOE_SIZE} hands completed. Game reset.")
    except Exception as e:
        st.error(f"Error in place_result: {str(e)}")

# --- UI Components ---
def render_setup_form():
    with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll or 1233.00, step=10.0)
                base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=max(st.session_state.base_bet, 0.10) or 10.00, step=0.10, format="%.2f")
                money_management = st.selectbox("Strategy", MONEY_MANAGEMENT_STRATEGIES, index=MONEY_MANAGEMENT_STRATEGIES.index(st.session_state.money_management))
            with col2:
                target_mode = st.selectbox("Target Mode", ["Profit %", "Units"], index=["Profit %", "Units"].index(st.session_state.target_profit_option))
                if target_mode == "Profit %":
                    target_value_percentage = st.number_input("Target Profit (%)", min_value=0.0, value=st.session_state.target_profit_percentage * 100 or 6.00, step=0.1, format="%.2f")
                    target_value_units = 0.0
                else:
                    target_value_units = st.number_input("Target Profit ($)", min_value=0.0, value=st.session_state.target_profit_units or 50.00, step=1.0, format="%.2f")
                    target_value_percentage = 0.0

            st.markdown('<div class="target-profit-section">', unsafe_allow_html=True)
            st.markdown('<h3><span class="icon">🛡</span>Safety & Limits</h3>', unsafe_allow_html=True)
            safety_net_enabled = st.checkbox("Enable Safety Net", value=True)
            safety_net_percentage = st.number_input("Safety Net Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.safety_net_percentage * 100 or 2.00, step=0.1, disabled=not safety_net_enabled)
            stop_loss_enabled = st.checkbox("Enable Stop-Loss", value=True)
            stop_loss_percentage = st.number_input("Stop-Loss Percentage (%)", min_value=0.0, max_value=100.0, value=st.session_state.stop_loss_percentage * 100 or 100.00, step=0.1, disabled=not stop_loss_enabled)
            profit_lock_threshold = st.number_input("Profit Lock Threshold (% of Initial Bankroll)", min_value=100.0, max_value=1000.0, value=st.session_state.win_limit * 100 or 600.00, step=1.0)
            smart_skip_enabled = st.checkbox("Enable Smart Skip", value=False)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.form_submit_button("Start Session"):
                minimum_bankroll = max([
                    FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER,
                    FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER,
                    GRID_MINIMUM_BANKROLL_MULTIPLIER,
                    10
                ]) * base_bet
                if bankroll <= 0:
                    st.error("Bankroll must be positive.")
                elif base_bet < 0.10:
                    st.error("Base bet must be at least $0.10.")
                elif base_bet > bankroll * 0.05:
                    st.error("Base bet cannot exceed 5% of bankroll.")
                elif bankroll < minimum_bankroll:
                    st.error(f"Bankroll must be at least ${minimum_bankroll:.2f} for the selected strategy.")
                elif stop_loss_percentage <= 0 or stop_loss_percentage >= 100:
                    st.error("Stop-loss percentage must be between 0% and 100%.")
                elif safety_net_percentage < 0 or safety_net_percentage >= 100:
                    st.error("Safety net percentage must be between 0% and 100%.")
                elif profit_lock_threshold <= 100:
                    st.error("Profit lock threshold must be greater than 100%.")
                else:
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
                        'stop_loss_percentage': stop_loss_percentage / 100,
                        'stop_loss_enabled': stop_loss_enabled,
                        'win_limit': profit_lock_threshold / 100,
                        'shoe_completed': False,
                        'safety_net_enabled': safety_net_enabled,
                        'safety_net_percentage': safety_net_percentage / 100,
                        'smart_skip_enabled': smart_skip_enabled,
                        'advice': "Enter a result to start AI betting",
                        'parlay_step': 1,
                        'parlay_wins': 0,
                        'parlay_using_base': True,
                        'parlay_step_changes': 0,
                        'parlay_peak_step': 1,
                        'moon_level': 1,
                        'moon_level_changes': 0,
                        'moon_peak_level': 1,
                        'target_profit_option': target_mode,
                        'target_profit_percentage': target_value_percentage / 100,
                        'target_profit_units': target_value_units,
                        'four_tier_level': 1,
                        'four_tier_step': 1,
                        'four_tier_losses': 0,
                        'flatbet_levelup_level': 1,
                        'flatbet_levelup_net_loss': 0.0,
                        'grid_pos': [0, 0],
                        'oscar_cycle_profit': 0.0,
                        'oscar_current_bet_level': 1
                    })
                    st.success(f"Session started with {money_management} strategy!")

def render_result_input():
    with st.expander("Enter Result", expanded=True):
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
                    valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']][-6:]
                    p_count = valid_sequence.count('P')
                    total = len(valid_sequence)
                    p_prob = p_count / total if total > 0 else 0.5
                    b_prob = 1 - p_prob
                    streak = False
                    if len(valid_sequence) >= 3 and len(set(valid_sequence[-3:])) == 1:
                        streak = True
                        if valid_sequence[-1] == 'P':
                            p_prob += 0.2
                            b_prob -= 0.2
                        else:
                            b_prob += 0.2
                            p_prob -= 0.2
                    p_prob = max(0, min(1, p_prob + random.uniform(-0.1, 0.1)))
                    b_prob = 1 - p_prob
                    bet_selection = 'P' if random.random() < p_prob else 'B'
                    rationale = f"AI Probability: P {p_prob*100:.0f}%, B {b_prob*100:.0f}%"
                    if streak:
                        rationale += f", Streak of {valid_sequence[-1]}"
                    min_bankroll_requirements = {
                        'FourTier': FOUR_TIER_MINIMUM_BANKROLL_MULTIPLIER,
                        'FlatbetLevelUp': FLATBET_LEVELUP_MINIMUM_BANKROLL_MULTIPLIER,
                        'Grid': GRID_MINIMUM_BANKROLL_MULTIPLIER,
                        'OscarGrind': 10,
                        'T3': 5,
                        'Parlay16': 95,
                        'Moon': 10,
                        'Flatbet': 1
                    }
                    for strategy, multiplier in min_bankroll_requirements.items():
                        if st.session_state.bankroll < st.session_state.base_bet * multiplier:
                            if st.session_state.money_management == strategy:
                                st.session_state.money_management = 'Flatbet'
                    profit_ratio = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_bankroll if st.session_state.initial_bankroll > 0 else 0
                    shoe_progress = len(st.session_state.sequence) / SHOE_SIZE
                    if shoe_progress > 0.75 and st.session_state.safety_net_enabled:
                        st.session_state.money_management = 'Flatbet'
                    elif abs(profit_ratio) < 0.1:
                        st.session_state.money_management = 'Flatbet' if random.random() < 0.5 else 'OscarGrind'
                    else:
                        aggressive_strategies = ['T3', 'Parlay16', 'Moon']
                        st.session_state.money_management = random.choice(aggressive_strategies)
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
                        st.session_state.advice = f"AI Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {rationale})"
                    else:
                        st.session_state.pending_bet = None
                        st.session_state.advice = f"AI Skipped betting (bet ${bet_amount:.2f} exceeds bankroll)"
                    st.success("Undone last action.")
                    st.rerun()

def render_bead_plate():
    with st.expander("Bead Plate", expanded=True):
        if not st.session_state.sequence:
            st.write("No results yet.")
        else:
            bead_plate = []
            row = []
            for i, result in enumerate(st.session_state.sequence):
                if i % 6 == 0 and i != 0:
                    bead_plate.append(row)
                    row = []
                color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                row.append(f'<div style="width:30px; height:30px; background-color:{color}; border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">{result}</div>')
            if row:
                bead_plate.append(row + [''] * (6 - len(row)))
            html = '<table style="border-collapse:collapse;">'
            for row in bead_plate:
                html += '<tr>'
                for cell in row:
                    html += f'<td style="padding:5px;">{cell}</td>'
                html += '</tr>'
            html += '</table>'
            st.markdown(html, unsafe_allow_html=True)

def render_prediction():
    with st.expander("Prediction", expanded=True):
        if st.session_state.bankroll == 0:
            st.info("Please start a session with a bankroll.")
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
                f"AI Decision: {advice}</p></div>",
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
                    "Safety_Net": h["Safety_Net"],
                    "Money_Management": h["Money_Management"]
                }
                for h in st.session_state.bet_history[-n:]
            ], use_container_width=True)

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="Mang Baccarat")
    apply_custom_css()
    st.title("Mang Baccarat")
    initialize_session_state()
    col1, col2 = st.columns([2, 1])
    with col1:
        render_setup_form()
        render_result_input()
        render_bead_plate()
        render_prediction()
        render_status()
    with col2:
        render_history()

if __name__ == "__main__":
    main()
