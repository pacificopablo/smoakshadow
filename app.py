import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import plotly.express as px
import plotly.graph_objects as go
import time
from typing import Tuple, Optional
import uuid

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SIMULATION_LOG = os.path.join(tempfile.gettempdir(), "simulation_log.txt")
STRATEGIES = ["Flat Bet", "D'Alembert", "-1 +2", "Suchi Masterline"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
SHOE_SIZE = 100
GRID_ROWS = 6
GRID_COLS = 16

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
    .stRadio > div > label, .stCheckbox > div > label {
        font-size: 14px;
        color: #4a5568;
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
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time())
    sessions = {}
    current_time = datetime.now()
    try:
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
    except PermissionError:
        st.error("Unable to access session file.")
        return 0
    sessions[st.session_state.session_id] = current_time
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
    except PermissionError:
        st.error("Unable to write to session file.")
        return 0
    return len(sessions)

# --- Session State Management ---
def initialize_session_state():
    defaults = {
        'bankroll': 519.0,
        'base_bet': 5.0,
        'initial_base_bet': 5.0,
        'bet': 1.0,
        'profit': 0.0,
        'strategy': 'Flat Bet',
        'sequence': [],
        'transitions': defaultdict(Counter),
        'last_prediction': None,
        'masterline_step': 0,
        'in_force2': False,
        'force2_failed': False,
        'break_countdown': 0,
        'history': [],
        'wins': 0,
        'losses': 0,
        'target_mode': 'Profit %',
        'target_value': 5.0,
        'initial_bankroll': 519.0,
        'target_hit': False,
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
        'consecutive_losses': 0,
        'last_was_tie': False,
        'safety_net_percentage': 5.0,
        'safety_net_enabled': True,
        'profit_lock': 519.0,
        'stop_loss_enabled': True,
        'stop_loss_percentage': 15.0,
        'profit_lock_notification': None,
        'profit_lock_threshold': 5.0,
        'is_paused': False,
        'shoe_completed': False,
        'advice': "Session initialized: Ready for bets.",
        'status': "Base"
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    if st.session_state.strategy not in STRATEGIES:
        st.session_state.strategy = 'Flat Bet'

def reset_session():
    setup_values = {
        'initial_bankroll': st.session_state.get('initial_bankroll', 519.0),
        'base_bet': st.session_state.get('base_bet', 5.0),
        'initial_base_bet': st.session_state.get('initial_base_bet', 5.0),
        'strategy': st.session_state.get('strategy', 'Flat Bet'),
        'target_mode': st.session_state.get('target_mode', 'Profit %'),
        'target_value': st.session_state.get('target_value', 5.0),
        'safety_net_enabled': st.session_state.get('safety_net_enabled', True),
        'safety_net_percentage': st.session_state.get('safety_net_percentage', 5.0),
        'stop_loss_enabled': st.session_state.get('stop_loss_enabled', True),
        'stop_loss_percentage': st.session_state.get('stop_loss_percentage', 15.0),
        'profit_lock_threshold': st.session_state.get('profit_lock_threshold', 5.0)
    }
    initialize_session_state()
    st.session_state.update({
        'bankroll': setup_values['initial_bankroll'],
        'base_bet': setup_values['base_bet'],
        'initial_base_bet': setup_values['initial_base_bet'],
        'bet': 1.0,
        'profit': 0.0,
        'strategy': setup_values['strategy'],
        'sequence': [],
        'transitions': defaultdict(Counter),
        'last_prediction': None,
        'masterline_step': 0,
        'in_force2': False,
        'force2_failed': False,
        'break_countdown': 0,
        'history': [],
        'wins': 0,
        'losses': 0,
        'target_mode': setup_values['target_mode'],
        'target_value': setup_values['target_value'],
        'initial_bankroll': setup_values['initial_bankroll'],
        'target_hit': False,
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
        'consecutive_losses': 0,
        'last_was_tie': False,
        'safety_net_percentage': setup_values['safety_net_percentage'],
        'safety_net_enabled': setup_values['safety_net_enabled'],
        'profit_lock': setup_values['initial_bankroll'],
        'stop_loss_enabled': setup_values['stop_loss_enabled'],
        'stop_loss_percentage': setup_values['stop_loss_percentage'],
        'profit_lock_notification': None,
        'profit_lock_threshold': setup_values['profit_lock_threshold'],
        'is_paused': False,
        'shoe_completed': False,
        'advice': "Session reset: Ready for new bets.",
        'status': "Base"
    })

# --- Prediction Logic ---
def predict_next() -> Tuple[Optional[str], str]:
    if st.session_state.break_countdown > 0:
        return "?", f"In break. Prediction paused ({st.session_state.break_countdown} left)."
    if len(st.session_state.sequence) < 2:
        return "?", "Insufficient history (need at least two results)."
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

# --- Betting Logic ---
def check_target_hit() -> bool:
    if st.session_state.target_mode == 'Profit %':
        profit = st.session_state.bankroll - st.session_state.initial_bankroll
        target = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        if profit >= target:
            st.session_state.advice = "Target hit: Profit goal reached."
            return True
    elif st.session_state.target_mode == 'Wins':
        if st.session_state.wins >= st.session_state.target_value:
            st.session_state.advice = "Target hit: Win goal reached."
            return True
    return False

def smart_stop() -> bool:
    if not st.session_state.stop_loss_enabled:
        return False
    stop_loss_threshold = st.session_state.initial_bankroll * (st.session_state.stop_loss_percentage / 100)
    if st.session_state.bankroll <= stop_loss_threshold:
        st.session_state.is_paused = True
        st.session_state.advice = f"Paused: Stop-loss triggered (Bankroll: ${st.session_state.bankroll:.2f}, Needs: >${stop_loss_threshold:.2f})"
        return True
    return False

def handle_masterline(win: bool):
    if st.session_state.in_force2:
        if win:
            st.session_state.profit += 2
            st.session_state.in_force2 = False
            st.session_state.bet = 1
            st.session_state.masterline_step = 0
            st.session_state.status = "Base"
        else:
            st.session_state.profit -= 2
            st.session_state.force2_failed = True
            st.session_state.in_force2 = False
            st.session_state.break_countdown = 3
            st.session_state.status = "Break (3 left)"
    elif st.session_state.force2_failed:
        st.session_state.force2_failed = False
        st.session_state.bet = 1
        st.session_state.masterline_step = 0
        st.session_state.status = "Base"
    elif win:
        ladder = [1, 3, 2, 5]
        st.session_state.profit += ladder[st.session_state.masterline_step]
        st.session_state.masterline_step += 1
        if st.session_state.masterline_step > 3:
            st.session_state.masterline_step = 0
            st.session_state.bet = 1
        else:
            st.session_state.bet = ladder[st.session_state.masterline_step]
        st.session_state.status = f"Ladder Step {st.session_state.masterline_step + 1}"
    else:
        if st.session_state.masterline_step == 0:
            st.session_state.in_force2 = True
            st.session_state.bet = 2
            st.session_state.status = "Force 2"
        else:
            st.session_state.profit -= st.session_state.bet
            st.session_state.break_countdown = 3
            st.session_state.masterline_step = 0
            st.session_state.bet = 1
            st.session_state.status = "Break (3 left)"

def calculate_bet_amount(pred: str, conf: str) -> Tuple[Optional[float], Optional[str]]:
    if st.session_state.break_countdown > 0:
        return None, f"No bet: In break ({st.session_state.break_countdown} left)"
    if len(st.session_state.sequence) < 2:
        return None, "No bet: Waiting for at least two results"
    if st.session_state.is_paused:
        return None, "No bet: Paused due to stop-loss"
    if st.session_state.shoe_completed:
        return None, "No bet: Shoe completed"
    if pred not in ['P', 'B']:
        return None, "No bet: Invalid prediction"

    bet_amount = st.session_state.bet * st.session_state.base_bet

    if st.session_state.safety_net_enabled:
        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if (bet_amount > st.session_state.bankroll or
            st.session_state.bankroll - bet_amount < safe_bankroll * 0.5 or
            bet_amount > st.session_state.bankroll * 0.15):
            return None, "No bet: Risk too high for current bankroll."

    return bet_amount, f"Next Bet: ${bet_amount:.2f} on {pred}"

def place_result(result: str):
    if st.session_state.target_hit:
        reset_session()
        st.session_state.advice = "Session reset: Target hit."
        return
    if st.session_state.shoe_completed:
        st.session_state.advice = "No action: Shoe completed."
        return

    previous_state = {
        "bankroll": st.session_state.bankroll,
        "bet": st.session_state.bet,
        "profit": st.session_state.profit,
        "masterline_step": st.session_state.masterline_step,
        "in_force2": st.session_state.in_force2,
        "force2_failed": st.session_state.force2_failed,
        "break_countdown": st.session_state.break_countdown,
        "last_prediction": st.session_state.last_prediction,
        "wins": st.session_state.wins,
        "losses": st.session_state.losses,
        "prediction_accuracy": st.session_state.prediction_accuracy.copy(),
        "consecutive_losses": st.session_state.consecutive_losses,
        "is_paused": st.session_state.is_paused,
        "status": st.session_state.get('status', '')
    }

    if st.session_state.sequence:
        prev = st.session_state.sequence[-1]
        st.session_state.transitions[prev][result] += 1
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > SEQUENCE_LIMIT:
        st.session_state.sequence = st.session_state.sequence[-SEQUENCE_LIMIT:]

    if st.session_state.break_countdown > 0:
        st.session_state.break_countdown -= 1
        st.session_state.status = f"Break ({st.session_state.break_countdown} left)"
        st.session_state.history.append({
            "Bet": None,
            "Result": result,
            "Amount": 0.0,
            "Win": False,
            "Status": st.session_state.get('status', ''),
            "Previous_State": previous_state,
            "Bet_Placed": False
        })
        if len(st.session_state.history) > HISTORY_LIMIT:
            st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]
        prediction, explanation = predict_next()
        st.session_state.last_prediction = prediction if prediction in ["P", "B"] else None
        st.session_state.insights = {"Explanation": explanation}
        st.session_state.advice = f"No bet: In break ({st.session_state.break_countdown} left)"
        return

    bet_amount = None
    bet_placed = False
    selection = None
    win = False

    if result in ["P", "B"] and len(st.session_state.sequence) >= 2:
        prediction, explanation = predict_next()
        selection = prediction if prediction in ["P", "B"] else None
        if selection:
            bet_amount, advice = calculate_bet_amount(selection, explanation)
            if bet_amount:
                bet_placed = True
                win = result == selection
                if win:
                    winnings = bet_amount * (0.95 if selection == 'B' else 1.0)
                    st.session_state.bankroll += winnings
                    st.session_state.profit += st.session_state.bet
                    st.session_state.wins += 1
                    st.session_state.consecutive_losses = 0
                    st.session_state.prediction_accuracy[selection] += 1
                    st.session_state.prediction_accuracy['total'] += 1
                else:
                    st.session_state.bankroll -= bet_amount
                    st.session_state.profit -= st.session_state.bet
                    st.session_state.losses += 1
                    st.session_state.consecutive_losses += 1
                    st.session_state.prediction_accuracy['total'] += 1

                if st.session_state.strategy == "Flat Bet":
                    st.session_state.bet = 1
                elif st.session_state.strategy == "D'Alembert":
                    if win:
                        st.session_state.bet = max(1, st.session_state.bet - 2)
                    else:
                        st.session_state.bet += 1
                elif st.session_state.strategy == "-1 +2":
                    if win:
                        st.session_state.bet += 2
                    else:
                        st.session_state.bet = max(1, st.session_state.bet - 1)
                elif st.session_state.strategy == "Suchi Masterline":
                    handle_masterline(win)
                st.session_state.advice = advice
            else:
                st.session_state.advice = advice or "No bet placed: Insufficient conditions."
        else:
            st.session_state.advice = explanation
    else:
        st.session_state.advice = "No bet placed: Insufficient history or Tie result."

    st.session_state.history.append({
        "Bet": selection,
        "Result": result,
        "Amount": bet_amount if bet_amount else 0.0,
        "Win": win,
        "Status": st.session_state.get('status', ''),
        "Previous_State": previous_state,
        "Bet_Placed": bet_placed
    })
    if len(st.session_state.history) > HISTORY_LIMIT:
        st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]

    if st.session_state.safety_net_enabled:
        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if st.session_state.bankroll <= safe_bankroll:
            st.session_state.advice = "Session reset: Safety net triggered."
            reset_session()
            return

    if smart_stop():
        return

    if check_target_hit():
        st.session_state.target_hit = True
        st.session_state.advice = "Target hit: Session reset."
        reset_session()
        return

    if len(st.session_state.sequence) >= SHOE_SIZE:
        st.session_state.shoe_completed = True
        st.session_state.advice = f"Shoe of {SHOE_SIZE} hands completed."

    st.session_state.profit_lock = max(st.session_state.profit_lock, st.session_state.bankroll)

    prediction, explanation = predict_next()
    st.session_state.last_prediction = prediction if prediction in ["P", "B"] else None
    st.session_state.insights = {"Explanation": explanation}

# --- Simulation Logic ---
def simulate_shoe(num_hands: int = SHOE_SIZE, strategy: str = 'Flat Bet') -> dict:
    outcomes = np.random.choice(['P', 'B', 'T'], size=num_hands, p=[0.4462, 0.4586, 0.0952])
    sequence = []
    correct = total = 0
    for outcome in outcomes:
        sequence.append(outcome)
        place_result(outcome)
        if st.session_state.last_prediction and outcome in ['P', 'B']:
            total += 1
            if st.session_state.last_prediction == outcome:
                correct += 1
        st.session_state.sequence = sequence.copy()
    accuracy = (correct / total * 100) if total > 0 else 0
    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'sequence': sequence,
        'final_bankroll': st.session_state.bankroll,
        'wins': st.session_state.wins,
        'losses': st.session_state.losses,
        'profit': st.session_state.profit,
        'strategy': strategy
    }
    try:
        with open(SIMULATION_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: Strategy={strategy}, Accuracy={accuracy:.1f}%, Correct={correct}/{total}, "
                    f"Final Bankroll=${result['final_bankroll']:.2f}, Profit={result['profit']:.2f} units, Wins={result['wins']}, Losses={result['losses']}\n")
    except PermissionError:
        st.warning("Unable to write to simulation log. Results not saved to file.")
    return result

def simulate_to_target(strategy: str, num_shoes: int) -> dict:
    results = []
    for _ in range(num_shoes):
        reset_session()
        st.session_state.strategy = strategy
        result = simulate_shoe(num_hands=SHOE_SIZE, strategy=strategy)
        results.append(result)
    accuracies = [r['accuracy'] for r in results]
    final_bankrolls = [r['final_bankroll'] for r in results]
    profits = [r['profit'] for r in results]
    wins = sum(r['wins'] for r in results)
    losses = sum(r['losses'] for r in results)
    return {
        'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
        'std_accuracy': np.std(accuracies) if accuracies else 0.0,
        'avg_bankroll': np.mean(final_bankrolls) if final_bankrolls else 0.0,
        'std_bankroll': np.std(final_bankrolls) if final_bankrolls else 0.0,
        'avg_profit': np.mean(profits) if profits else 0.0,
        'std_profit': np.std(profits) if profits else 0.0,
        'wins': wins,
        'losses': losses,
        'results': results
    }

# --- UI Components ---
def render_setup_form():
    with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
                base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=max(st.session_state.base_bet, 0.10), step=0.10, format="%.2f")
            with col2:
                betting_strategy = st.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index(st.session_state.strategy))
                target_mode = st.selectbox("Target Mode", ['Profit %', 'Wins'], index=0 if st.session_state.target_mode == 'Profit %' else 1)
                target_value = st.number_input("Target Value", min_value=0.0, value=float(st.session_state.target_value), step=1.0)
            safety_net_enabled = st.checkbox("Enable Safety Net", value=st.session_state.safety_net_enabled)
            safety_net_percentage = st.session_state.safety_net_percentage
            if safety_net_enabled:
                safety_net_percentage = st.number_input("Safety Net Percentage (%)", min_value=0.0, max_value=50.0, value=5.0, step=5.0)
            stop_loss_enabled = st.checkbox("Enable Stop-Loss", value=st.session_state.stop_loss_enabled)
            stop_loss_percentage = st.session_state.stop_loss_percentage
            if stop_loss_enabled:
                stop_loss_percentage = st.number_input("Stop-Loss Percentage (%)", min_value=10.0, max_value=90.0, value=st.session_state.stop_loss_percentage, step=1.0)
            profit_lock_threshold = st.number_input("Profit Lock Threshold (% of Initial Bankroll)", min_value=0.0, value=st.session_state.profit_lock_threshold, step=1.0)
            if st.form_submit_button("Start Session"):
                if bankroll <= 0:
                    st.error("Bankroll must be positive.")
                elif base_bet < 0.10:
                    st.error("Base bet must be at least $0.10.")
                elif base_bet > bankroll:
                    st.error("Base bet cannot exceed bankroll.")
                else:
                    st.session_state.update({
                        'bankroll': bankroll,
                        'base_bet': base_bet,
                        'initial_base_bet': base_bet,
                        'bet': 1.0,
                        'profit': 0.0,
                        'strategy': betting_strategy,
                        'sequence': [],
                        'transitions': defaultdict(Counter),
                        'last_prediction': None,
                        'masterline_step': 0,
                        'in_force2': False,
                        'force2_failed': False,
                        'break_countdown': 0,
                        'history': [],
                        'wins': 0,
                        'losses': 0,
                        'target_mode': target_mode,
                        'target_value': target_value,
                        'initial_bankroll': bankroll,
                        'target_hit': False,
                        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
                        'consecutive_losses': 0,
                        'last_was_tie': False,
                        'safety_net_percentage': safety_net_percentage,
                        'safety_net_enabled': safety_net_enabled,
                        'profit_lock': bankroll,
                        'stop_loss_enabled': stop_loss_enabled,
                        'stop_loss_percentage': stop_loss_percentage,
                        'profit_lock_notification': None,
                        'profit_lock_threshold': profit_lock_threshold,
                        'is_paused': False,
                        'shoe_completed': False,
                        'advice': "Session started: Ready for bets.",
                        'status': "Base"
                    })
                    st.success(f"Session started with {betting_strategy} strategy!")

def render_result_input():
    with st.expander("Enter Result", expanded=True):
        if st.session_state.shoe_completed:
            st.success(f"Shoe of {SHOE_SIZE} hands completed!")
        cols = st.columns(4)
        with cols[0]:
            if st.button("Player", key="player_btn", disabled=st.session_state.shoe_completed):
                place_result("P")
        with cols[1]:
            if st.button("Banker", key="banker_btn", disabled=st.session_state.shoe_completed):
                place_result("B")
        with cols[2]:
            if st.button("Tie", key="tie_btn", disabled=st.session_state.shoe_completed):
                place_result("T")
        with cols[3]:
            if st.button("Undo Last", key="undo_btn", disabled=not st.session_state.history or st.session_state.shoe_completed):
                if not st.session_state.sequence:
                    st.warning("No results to undo.")
                else:
                    try:
                        if st.session_state.history:
                            last = st.session_state.history.pop()
                            previous_state = last['Previous_State']
                            for key, value in previous_state.items():
                                st.session_state[key] = value
                            st.session_state.sequence.pop()
                            if last['Bet_Placed'] and not last['Win']:
                                if len(st.session_state.sequence) >= 1:
                                    prev = st.session_state.sequence[-1] if st.session_state.sequence else None
                                    if prev and last['Result'] in st.session_state.transitions[prev]:
                                        st.session_state.transitions[prev][last['Result']] -= 1
                                        if st.session_state.transitions[prev][last['Result']] == 0:
                                            del st.session_state.transitions[prev][last['Result']]
                            prediction, explanation = predict_next()
                            st.session_state.last_prediction = prediction if prediction in ["P", "B"] else None
                            bet_amount, advice = calculate_bet_amount(prediction, explanation)
                            st.session_state.advice = advice
                            st.session_state.insights = {"Explanation": explanation}
                            st.session_state.shoe_completed = False
                            st.success("Undone last action.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error undoing last action: {str(e)}")
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
                    if result == 'P':
                        color = '#3182ce'
                    elif result == 'B':
                        color = '#e53e3e'
                    else:
                        color = '#38a169'
                    grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'
        for row in grid:
            st.markdown(' '.join(row), unsafe_allow_html=True)

def render_prediction():
    with st.expander("Prediction", expanded=True):
        if st.session_state.profit_lock_notification:
            st.success(st.session_state.profit_lock_notification)
            st.session_state.profit_lock_notification = None
        pred, explanation = predict_next()
        if pred in ['P', 'B']:
            bet_amount, advice = calculate_bet_amount(pred, explanation)
            st.session_state.advice = advice
            color = '#3182ce' if pred == 'P' else '#e53e3e'
            st.markdown(f"<div style='background-color: #edf2f7; padding: 15px; border-radius: 8px;'><p style='color:{color}; font-size:1.5rem; font-weight:bold; margin:0;'>{advice or 'No bet recommended.'}</p></div>", unsafe_allow_html=True)
        else:
            st.session_state.advice = explanation
            st.info(st.session_state.advice or "No prediction available.")

def render_insights():
    with st.expander("Prediction Insights", expanded=True):
        if not st.session_state.insights:
            st.write("No insights available.")
        else:
            for factor, contribution in st.session_state.insights.items():
                st.markdown(f"**{factor}**: {contribution}")

def render_status():
    with st.expander("Session Status", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
            st.markdown(f"**Profit Lock**: ${st.session_state.profit_lock:.2f}")
            st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
            st.markdown(f"**Safety Net**: {'Enabled' if st.session_state.safety_net_enabled else 'Disabled'}"
                        f"{' | ' + str(st.session_state.safety_net_percentage) + '%' if st.session_state.safety_net_enabled else ''}")
            st.markdown(f"**Stop-Loss**: {'Enabled' if st.session_state.stop_loss_enabled else 'Disabled'}"
                        f"{' | ' + str(st.session_state.stop_loss_percentage) + '%' if st.session_state.stop_loss_enabled else ''}")
        with col2:
            strategy_status = f"**Strategy**: {st.session_state.strategy}"
            if st.session_state.strategy == "Suchi Masterline":
                strategy_status += f"<br>Status: {st.session_state.get('status', '')}<br>Step: {st.session_state.masterline_step}"
            st.markdown(strategy_status, unsafe_allow_html=True)
        st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
        st.markdown(f"**Profit (Units)**: {st.session_state.profit:.2f}")
        st.markdown(f"**Online Users**: {track_user_session()}")
        if st.session_state.initial_base_bet > 0 and st.session_state.initial_bankroll > 0:
            profit = st.session_state.bankroll - st.session_state.initial_bankroll
            units_profit = profit / st.session_state.initial_base_bet
            st.markdown(f"**Profit**: {units_profit:.2f} units (${profit:.2f})")
        else:
            st.markdown("**Profit**: 0.00 units ($0.00)")

def render_accuracy():
    with st.expander("Prediction Accuracy", expanded=True):
        acc = st.session_state.prediction_accuracy
        total = acc['total']
        if total > 0:
            overall = ((acc['P'] + acc['B']) / total * 100)
            p_acc = (acc['P'] / total * 100)
            b_acc = (acc['B'] / total * 100)
            st.markdown(f"**Overall Accuracy**: {overall:.1f}%")
            st.markdown(f"**Player Accuracy**: {p_acc:.1f}% ({acc['P']}/{total})")
            st.markdown(f"**Banker Accuracy**: {b_acc:.1f}% ({acc['B']}/{total})")
        else:
            st.write("No predictions made yet.")
        if st.session_state.history:
            accuracy_data = []
            correct = total = 0
            for h in st.session_state.history[-50:]:
                if h['Bet_Placed'] and h['Bet'] in ['P', 'B']:
                    total += 1
                    if h['Win']:
                        correct += 1
                    accuracy_data.append(correct / max(total, 1) * 100)
            if accuracy_data:
                st.line_chart(accuracy_data, use_container_width=True)

def render_history():
    with st.expander("Bet History", expanded=True):
        if not st.session_state.history:
            st.write("No history available.")
        else:
            n = st.slider("Show last N bets", 5, 50, 10)
            st.dataframe([
                {
                    "Bet": h["Bet"] if h["Bet"] else "-",
                    "Result": h["Result"],
                    "Amount": f"${h['Amount']:.2f}" if h["Bet_Placed"] else "-",
                    "Outcome": "Win" if h["Win"] else "Loss" if h["Bet_Placed"] else "-",
                    "Status": h["Status"] if st.session_state.strategy == "Suchi Masterline" else "-"
                }
                for h in st.session_state.history[-n:]
            ], use_container_width=True)

def render_export():
    with st.expander("Export Session", expanded=True):
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False)
            st.download_button(label="Download History as CSV", data=csv, file_name="baccarat_history.csv", mime="text/csv")
        if os.path.exists(SIMULATION_LOG):
            with open(SIMULATION_LOG, 'r', encoding='utf-8') as f:
                log_content = f.read()
            st.download_button(label="Download Simulation Log", data=log_content, file_name="simulation_log.txt", mime="text/plain")

def render_simulation():
    with st.expander("Run Simulation", expanded=True):
        num_shoes = st.number_input("Number of Shoes to Simulate", min_value=1, max_value=100, value=10, step=1)
        strategy = st.selectbox("Simulation Strategy", STRATEGIES, index=STRATEGIES.index(st.session_state.strategy))
        if st.button("Run Simulation", key="run_sim_btn"):
            with st.spinner("Running simulation..."):
                result = simulate_to_target(strategy, num_shoes)
                st.write(f"**Simulation Results**")
                st.write(f"Average Accuracy: {result['avg_accuracy']:.1f}% (Â±{result['std_accuracy']:.1f}%)")
                st.write(f"Average Final Bankroll: ${result['avg_bankroll']:.2f} (Â±${result['std_bankroll']:.2f})")
                st.write(f"Average Profit: {result['avg_profit']:.2f} units (Â±{result['std_profit']:.2f})")
                st.write(f"Total Wins: {result['wins']}")
                st.write(f"Total Losses: {result['losses']}")
                fig = go.Figure()
                for i, res in enumerate(result['results']):
                    fig.add_trace(go.Scatter(x=list(range(1, len(res['sequence']) + 1)), y=[r['final_bankroll'] for r in result['results'][:i+1]], mode='lines', name=f"Shoe {i+1}"))
                fig.update_layout(title="Bankroll Over Shoes", xaxis_title="Hand", yaxis_title="Bankroll ($)", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

def render_profit_dashboard():
    with st.expander("Profit Dashboard", expanded=True):
        profit = st.session_state.bankroll - st.session_state.initial_bankroll
        roi = (profit / st.session_state.initial_bankroll * 100) if st.session_state.initial_bankroll > 0 else 0.0
        st.markdown(f"**Profit**: ${profit:.2f}")
        st.markdown(f"**ROI**: {roi:.2f}%")

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
        render_profit_dashboard()
        render_status()
        render_insights()
    with col2:
        render_accuracy()
        render_history()
        render_export()
        render_simulation()

if __name__ == "__main__":
    main()
