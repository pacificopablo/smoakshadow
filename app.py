import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime
from collections import defaultdict
from itertools import product
import plotly.express as px
import plotly.graph_objects as go

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SIMULATION_LOG = os.path.join(tempfile.gettempdir(), "simulation_log.txt")
PARLAY_TABLE = {
    i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
        (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
        (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
    ], 1)
}
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Z1003.1", "Genius"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
LOSS_LOG_LIMIT = 50
WINDOW_SIZE = 50
T3_MAX_LEVEL = 10
SHOE_SIZE = 100
GRID_ROWS = 6
GRID_COLS = 16  # Adjust based on your desired bead plate size

# Placeholder for profit_enhancements module
def adjust_safety_net():
    return st.session_state.safety_net_percentage

def recommend_strategy(sequence):
    return "T3"

def enhanced_z1003_bet(loss_count, base_bet):
    factors = [1.0, 2.0, 3.0]
    return base_bet * factors[min(loss_count, len(factors) - 1)]

def calculate_roi():
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    return (profit / st.session_state.initial_bankroll * 100) if st.session_state.initial_bankroll > 0 else 0.0

def render_profit_dashboard():
    st.markdown("**Profit Dashboard**")
    st.markdown(f"ROI: {calculate_roi():.2f}%")

def apply_custom_css():
    st.markdown("""
        <style>
        .stButton>button { width: 100%; }
        .stTextInput>div>div>input { width: 100%; }
        .stNumberInput>div>div>input { width: 100%; }
        .stSelectbox>div>div>select { width: 100%; }
        .css-1d391kg { padding: 1rem; }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    defaults = {
        'bankroll': 519.0,
        'base_bet': 5.0,
        'initial_base_bet': 5.0,
        'sequence': [],
        'pending_bet': None,
        'strategy': 'Genius',
        't3_level': 1,
        't3_results': [],
        't3_level_changes': 0,
        't3_peak_level': 1,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'z1003_loss_count': 0,
        'z1003_bet_factor': 1.0,
        'z1003_continue': False,
        'z1003_level_changes': 0,
        'advice': "",
        'history': [],
        'wins': 0,
        'losses': 0,
        'target_mode': 'Profit %',
        'target_value': 5.0,
        'initial_bankroll': 519.0,
        'target_hit': False,
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
        'consecutive_losses': 0,
        'loss_log': [],
        'last_was_tie': False,
        'insights': {},
        'pattern_volatility': 0.0,
        'trend_score': {'streak': 0.0, 'chop': 0.0, 'double': 0.0},
        'pattern_success': defaultdict(int),
        'pattern_attempts': defaultdict(int),
        'safety_net_percentage': 10.0,
        'safety_net_enabled': True,
        'profit_lock': 519.0,
        'stop_loss_enabled': True,
        'stop_loss_percentage': 15.0,
        'profit_lock_notification': None,
        'profit_lock_threshold': 5.0,
        'recent_accuracy': 0.0,
        'smart_skip': True,
        'non_betting_deals': 0,
        'is_paused': False,
        'shoe_completed': False
    }
    defaults['pattern_success']['fourgram'] = 0
    defaults['pattern_attempts']['fourgram'] = 0
    defaults['pattern_success']['markov'] = 0
    defaults['pattern_attempts']['markov'] = 0
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.strategy not in STRATEGIES:
        st.session_state.strategy = 'Genius'

def reset_session():
    setup_values = {
        'initial_bankroll': st.session_state.initial_bankroll,
        'base_bet': st.session_state.base_bet,
        'initial_base_bet': st.session_state.initial_base_bet,
        'strategy': st.session_state.strategy,
        'target_mode': st.session_state.target_mode,
        'target_value': st.session_state.target_value,
        'safety_net_enabled': st.session_state.safety_net_enabled,
        'safety_net_percentage': st.session_state.safety_net_percentage,
        'stop_loss_enabled': st.session_state.stop_loss_enabled,
        'stop_loss_percentage': st.session_state.stop_loss_percentage,
        'profit_lock_threshold': st.session_state.profit_lock_threshold,
        'smart_skip': st.session_state.smart_skip
    }
    initialize_session_state()
    st.session_state.update({
        'bankroll': setup_values['initial_bankroll'],
        'base_bet': setup_values['base_bet'],
        'initial_base_bet': setup_values['initial_base_bet'],
        'strategy': setup_values['strategy'],
        'sequence': [],
        'pending_bet': None,
        't3_level': 1,
        't3_results': [],
        't3_level_changes': 0,
        't3_peak_level': 1,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1,
        'z1003_loss_count': 0,
        'z1003_bet_factor': 1.0,
        'z1003_continue': False,
        'z1003_level_changes': 0,
        'advice': "Session reset: Target or stop loss reached.",
        'history': [],
        'wins': 0,
        'losses': 0,
        'target_mode': setup_values['target_mode'],
        'target_value': setup_values['target_value'],
        'initial_bankroll': setup_values['initial_bankroll'],
        'target_hit': False,
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
        'consecutive_losses': 0,
        'loss_log': [],
        'last_was_tie': False,
        'insights': {},
        'pattern_volatility': 0.0,
        'trend_score': {'streak': 0.0, 'chop': 0.0, 'double': 0.0},
        'pattern_success': defaultdict(int),
        'pattern_attempts': defaultdict(int),
        'safety_net_percentage': setup_values['safety_net_percentage'],
        'safety_net_enabled': setup_values['safety_net_enabled'],
        'profit_lock': setup_values['initial_bankroll'],
        'stop_loss_enabled': setup_values['stop_loss_enabled'],
        'stop_loss_percentage': setup_values['stop_loss_percentage'],
        'profit_lock_notification': None,
        'profit_lock_threshold': setup_values['profit_lock_threshold'],
        'recent_accuracy': 0.0,
        'smart_skip': setup_values['smart_skip'],
        'non_betting_deals': 0,
        'is_paused': False,
        'shoe_completed': False
    })
    st.session_state.pattern_success['fourgram'] = 0
    st.session_state.pattern_attempts['fourgram'] = 0
    st.session_state.pattern_success['markov'] = 0
    st.session_state.pattern_attempts['markov'] = 0

def update_t3_level():
    if len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        losses = st.session_state.t3_results.count('L')
        old_level = st.session_state.t3_level
        if wins == 3:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
        elif wins == 2 and losses == 1:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses == 2 and wins == 1:
            st.session_state.t3_level = min(T3_MAX_LEVEL, st.session_state.t3_level + 1)
        elif losses == 3:
            st.session_state.t3_level = min(T3_MAX_LEVEL, st.session_state.t3_level + 2)
        if old_level != st.session_state.t3_level:
            st.session_state.t3_level_changes += 1
        st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, st.session_state.t3_level)
        st.session_state.t3_results = []

def analyze_patterns(sequence):
    if len(sequence) < 4:
        return 0.0, 0.0, {}
    recent = sequence[-WINDOW_SIZE:] if len(sequence) >= WINDOW_SIZE else sequence
    transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1] and recent[i] != 'T' and recent[i-1] != 'T')
    volatility = transitions / (len(recent) - 1) if len(recent) > 1 else 0.0
    streak = 0.0
    chop = 0.0
    double = 0.0
    for i in range(2, len(recent)):
        if recent[i] == recent[i-1] == recent[i-2] != 'T':
            streak += 1.0
        if i >= 3 and recent[i] == recent[i-2] != recent[i-1] != 'T' and recent[i-1] != recent[i-3] != 'T':
            chop += 1.0
        if i >= 3 and recent[i] == recent[i-1] != recent[i-2] != 'T' and recent[i-2] == recent[i-3] != 'T':
            double += 1.0
    total_patterns = max(1, streak + chop + double)
    streak_score = streak / total_patterns
    chop_score = chop / total_patterns
    double_score = double / total_patterns
    outcomes = [o for o in sequence if o != 'T']
    p_count = outcomes.count('P')
    b_count = outcomes.count('B')
    total = p_count + b_count
    shoe_bias = (b_count - p_count) / total if total > 0 else 0.0
    insights = {
        'volatility': volatility,
        'streak': streak_score,
        'chop': chop_score,
        'double': double_score
    }
    return volatility, shoe_bias, insights

def smart_predict():
    if len(st.session_state.sequence) < 4:
        return None, 0.0, {}
    sequence = st.session_state.sequence
    volatility, shoe_bias, insights = analyze_patterns(sequence)
    st.session_state.pattern_volatility = volatility
    st.session_state.trend_score = {'streak': insights['streak'], 'chop': insights['chop'], 'double': insights['double']}
    fourgram = sequence[-4:] if len(sequence) >= 4 else sequence
    fourgram_key = ''.join(fourgram)
    fourgram_pred = 'B' if fourgram_key in ['PPPP', 'BBBB', 'PPBB', 'BBPP'] else 'P'
    fourgram_conf = 60.0 if fourgram_key in ['PPPP', 'BBBB'] else 50.0
    markov_pred = None
    markov_conf = 0.0
    if len(sequence) >= 2:
        last_two = sequence[-2:]
        transitions = defaultdict(lambda: {'P': 0, 'B': 0, 'T': 0})
        for i in range(len(sequence) - 2):
            current = sequence[i:i+2]
            next_outcome = sequence[i+2]
            transitions[''.join(current)][next_outcome] += 1
        current = ''.join(last_two)
        if current in transitions:
            total = sum(transitions[current].values())
            if total > 0:
                p_prob = transitions[current]['P'] / total
                b_prob = transitions[current]['B'] / total
                markov_pred = 'P' if p_prob > b_prob else 'B'
                markov_conf = max(p_prob, b_prob) * 100
    final_pred = fourgram_pred
    final_conf = fourgram_conf
    if markov_conf > fourgram_conf:
        final_pred = markov_pred
        final_conf = markov_conf
    if st.session_state.trend_score['streak'] > 0.6 and sequence[-1] != 'T':
        final_pred = sequence[-1]
        final_conf += 10.0
    elif st.session_state.trend_score['chop'] > 0.6 and sequence[-1] != 'T':
        final_pred = 'P' if sequence[-1] == 'B' else 'B'
        final_conf += 5.0
    if shoe_bias > 0.2 and final_pred == 'P':
        final_pred = 'B'
        final_conf += 5.0
    elif shoe_bias < -0.2 and final_pred == 'B':
        final_pred = 'P'
        final_conf += 5.0
    final_conf = min(final_conf, 100.0)
    insights['fourgram'] = fourgram_pred
    insights['markov'] = markov_pred if markov_pred else 'None'
    return final_pred, final_conf, insights

def check_target_hit():
    if st.session_state.target_mode == 'Profit %':
        profit = st.session_state.bankroll - st.session_state.initial_bankroll
        target = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return profit >= target
    return st.session_state.wins >= st.session_state.target_value

def smart_stop():
    if not st.session_state.stop_loss_enabled:
        return False
    if st.session_state.consecutive_losses >= 3:
        if st.session_state.non_betting_deals < 3:
            st.session_state.is_paused = True
            return True
        st.session_state.is_paused = False
        st.session_state.non_betting_deals = 0
        return False
    stop_loss_threshold = st.session_state.initial_bankroll * (st.session_state.stop_loss_percentage / 100)
    if st.session_state.bankroll <= stop_loss_threshold:
        st.session_state.is_paused = True
        return True
    return False

def genius_bet(conf: float, shoe_bias: float) -> float:
    base = st.session_state.base_bet * (conf / 40.0)
    if st.session_state.consecutive_losses > 0 and conf > 50.0:
        base *= 2.0
    if shoe_bias > 0.2:
        base *= 1.1
    elif shoe_bias < -0.2:
        base *= 1.1
    base_bet = st.session_state.base_bet
    if base_bet > 0:
        rounded_bet = round(base / base_bet) * base_bet
        rounded_bet = max(rounded_bet, base_bet)
    else:
        rounded_bet = base
    capped_bet = min(rounded_bet, st.session_state.bankroll * 0.05)
    if base_bet > 0:
        capped_bet = (int(capped_bet / base_bet)) * base_bet
        capped_bet = max(capped_bet, base_bet) if capped_bet <= st.session_state.bankroll * 0.05 else 0
    return capped_bet

def calculate_bet_amount(pred: str, conf: float) -> tuple[any, any]:
    if len(st.session_state.sequence) < 8:
        return None, "No bet: Waiting for 9th hand"
    if st.session_state.smart_skip:
        if conf < 45.0:
            return None, f"No bet: Confidence too low ({conf:.1f}%)"
        if st.session_state.pattern_volatility > 0.6:
            return None, "No bet: High pattern volatility"
        if st.session_state.shoe_completed:
            return None, "No bet: Shoe completed"
        if smart_stop():
            if st.session_state.consecutive_losses >= 3:
                return None, f"No bet: Paused due to {st.session_state.consecutive_losses} consecutive losses ({st.session_state.non_betting_deals}/3 deals)"
            return None, f"No bet: Paused due to stop-loss (Bankroll: ${st.session_state.bankroll:.2f}, Needs: >${st.session_state.initial_bankroll * st.session_state.stop_loss_percentage / 100:.2f})"
    if pred is None or conf < 32.0:
        return None, f"No bet: Confidence too low"
    if st.session_state.strategy in ['T3', 'Genius'] and st.session_state.bankroll >= st.session_state.profit_lock:
        profit_gained = st.session_state.bankroll - st.session_state.profit_lock
        if profit_gained >= st.session_state.initial_bankroll * (st.session_state.profit_lock_threshold / 100):
            st.session_state.profit_lock_notification = f"Profit lock reached at ${st.session_state.bankroll:.2f} (+${profit_gained:.2f}). Resetting strategy."
            if st.session_state.strategy == 'T3':
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
            elif st.session_state.strategy == 'Genius':
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
            st.session_state.profit_lock = st.session_state.bankroll
    if st.session_state.strategy == 'Z1003.1':
        if st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
            return None, "No bet: Stopped after three losses (Z1003.1 rule)"
        bet_amount = enhanced_z1003_bet(st.session_state.z1003_loss_count, st.session_state.base_bet)
    elif st.session_state.strategy == 'Flatbet':
        bet_amount = st.session_state.base_bet
    elif st.session_state.strategy == 'T3':
        bet_amount = st.session_state.base_bet * st.session_state.t3_level
    elif st.session_state.strategy == 'Genius':
        shoe_bias = analyze_patterns(st.session_state.sequence)[-2]
        bet_amount = genius_bet(conf, shoe_bias)
    else:  # Parlay16
        key = 'base' if st.session_state.parlay_using_base else 'parlay'
        bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)
    if st.session_state.safety_net_enabled:
        safe_bankroll = st.session_state.initial_bankroll * (adjust_safety_net() / 100)
        if bet_amount > st.session_state.bankroll or st.session_state.bankroll - bet_amount < safe_bankroll * 0.5 or bet_amount > st.session_state.bankroll * 0.10:
            if st.session_state.strategy == 'T3':
                old_level = st.session_state.t3_level
                st.session_state.t3_level = 1
                if old_level != st.session_state.t3_level:
                    st.session_state.t3_level_changes += 1
                st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, old_level)
                bet_amount = st.session_state.base_bet
            elif st.session_state.strategy == 'Parlay16':
                old_step = st.session_state.parlay_step
                st.session_state.parlay_step = 1
                st.session_state.parlay_using_base = True
                if old_step != st.session_state.parlay_step:
                    st.session_state.parlay_step_changes += 1
                st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step]['base']
            elif st.session_state.strategy == 'Z1003.1':
                old_loss_count = st.session_state.z1003_loss_count
                st.session_state.z1003_loss_count = 0
                st.session_state.z1003_bet_factor = 1.0
                if old_loss_count != st.session_state.z1003_loss_count:
                    st.session_state.z1003_level_changes += 1
                bet_amount = st.session_state.base_bet
            elif st.session_state.strategy == 'Genius':
                bet_amount = st.session_state.base_bet
            return None, "No bet: Risk too high for current bankroll. Level/step reset to 1."
    return bet_amount, f"Next Bet: ${bet_amount:.2f} on {pred}"

def place_result(result: str):
    if st.session_state.target_hit:
        reset_session()
        return
    if st.session_state.safety_net_enabled:
        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if st.session_state.bankroll <= safe_bankroll:
            st.session_state.advice = "Session reset: Stop loss reached."
            reset_session()
            return
    st.session_state.last_was_tie = (result == 'T')
    bet_amount = 0
    bet_placed = False
    selection = None
    win = False
    previous_state = {
        "bankroll": st.session_state.bankroll,
        "t3_level": st.session_state.t3_level,
        "t3_results": st.session_state.t3_results.copy(),
        "parlay_step": st.session_state.parlay_step,
        "parlay_wins": st.session_state.parlay_wins,
        "parlay_using_base": st.session_state.parlay_using_base,
        "z1003_loss_count": st.session_state.z1003_loss_count,
        "z1003_bet_factor": st.session_state.z1003_bet_factor,
        "z1003_continue": st.session_state.z1003_continue,
        "pending_bet": st.session_state.pending_bet,
        "wins": st.session_state.wins,
        "losses": st.session_state.losses,
        "prediction_accuracy": st.session_state.prediction_accuracy.copy(),
        "consecutive_losses": st.session_state.consecutive_losses,
        "t3_level_changes": st.session_state.t3_level_changes,
        "parlay_step_changes": st.session_state.parlay_step_changes,
        "pattern_volatility": st.session_state.pattern_volatility,
        "trend_score": st.session_state.trend_score.copy(),
        "pattern_success": st.session_state.pattern_success.copy(),
        "pattern_attempts": st.session_state.pattern_attempts.copy(),
        "recent_accuracy": st.session_state.recent_accuracy,
        "non_betting_deals": st.session_state.non_betting_deals,
        "is_paused": st.session_state.is_paused
    }
    if st.session_state.is_paused:
        st.session_state.non_betting_deals += 1
    if result != 'T':
        pred, conf, _ = smart_predict()
        selection = pred
        if selection in ['P', 'B']:
            bet_amount, advice = calculate_bet_amount(selection, conf)
            if bet_amount is None:
                st.session_state.advice = advice or "No bet placed: Insufficient conditions."
            else:
                win = result == selection
                bet_placed = True
                if win:
                    st.session_state.bankroll += bet_amount * (0.95 if selection == 'B' else 1.0)
                    if st.session_state.strategy == 'T3':
                        st.session_state.t3_results.append('W')
                    elif st.session_state.strategy == 'Parlay16':
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
                    elif st.session_state.strategy == 'Z1003.1':
                        st.session_state.z1003_loss_count = 0
                        st.session_state.z1003_continue = False
                    elif st.session_state.strategy == 'Genius':
                        st.session_state.t3_results.append('W')
                    st.session_state.wins += 1
                    st.session_state.prediction_accuracy[selection] += 1
                    st.session_state.consecutive_losses = 0
                    for pattern in st.session_state.insights:
                        st.session_state.pattern_success[pattern] += 1
                        st.session_state.pattern_attempts[pattern] += 1
                else:
                    st.session_state.bankroll -= bet_amount
                    if st.session_state.strategy == 'T3':
                        st.session_state.t3_results.append('L')
                    elif st.session_state.strategy == 'Parlay16':
                        st.session_state.parlay_wins = 0
                        old_step = st.session_state.parlay_step
                        st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                        st.session_state.parlay_using_base = True
                        if old_step != st.session_state.parlay_step:
                            st.session_state.parlay_step_changes += 1
                        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                    elif st.session_state.strategy == 'Z1003.1':
                        st.session_state.z1003_loss_count += 1
                        if st.session_state.z1003_loss_count == 2 and st.session_state.history and st.session_state.history[-1]['Win']:
                            st.session_state.z1003_continue = True
                        elif st.session_state.z1003_loss_count >= 3:
                            st.session_state.z1003_continue = False
                    elif st.session_state.strategy == 'Genius':
                        st.session_state.t3_results.append('L')
                    st.session_state.losses += 1
                    st.session_state.consecutive_losses += 1
                    _, conf, _ = smart_predict()
                    st.session_state.loss_log.append({
                        'sequence': st.session_state.sequence[-10:],
                        'prediction': selection,
                        'result': result,
                        'confidence': f"{conf:.1f}",
                        'insights': st.session_state.insights.copy()
                    })
                    if len(st.session_state.loss_log) > LOSS_LOG_LIMIT:
                        st.session_state.loss_log = st.session_state.loss_log[-LOSS_LOG_LIMIT:]
                    for pattern in st.session_state.insights:
                        st.session_state.pattern_attempts[pattern] += 1
                st.session_state.prediction_accuracy['total'] += 1
    st.session_state.sequence.append(result)
    if len(st.session_state.sequence) > SEQUENCE_LIMIT:
        st.session_state.sequence = st.session_state.sequence[-SEQUENCE_LIMIT:]
    st.session_state.history.append({
        "Bet": selection,
        "Result": result,
        "Amount": bet_amount,
        "Win": win,
        "T3_Level": st.session_state.t3_level,
        "Parlay_Step": st.session_state.parlay_step,
        "Z1003_Loss_Count": st.session_state.z1003_loss_count,
        "Previous_State": previous_state,
        "Bet_Placed": bet_placed
    })
    if len(st.session_state.history) > HISTORY_LIMIT:
        st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]
    if check_target_hit():
        st.session_state.target_hit = True
        reset_session()
        return
    if st.session_state.safety_net_enabled:
        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if st.session_state.bankroll <= safe_bankroll:
            st.session_state.advice = "Session reset: Stop loss reached."
            reset_session()
            return
    pred, conf, insights = smart_predict()
    bet_amount, advice = calculate_bet_amount(pred, conf)
    if not smart_stop() and st.session_state.non_betting_deals >= 3 and st.session_state.is_paused:
        advice = "Betting resumed after 3 deals"
        st.session_state.is_paused = False
        st.session_state.non_betting_deals = 0
    st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
    st.session_state.advice = advice
    st.session_state.insights = insights
    if st.session_state.strategy in ['T3', 'Genius']:
        update_t3_level()
    if len(st.session_state.sequence) >= SHOE_SIZE:
        st.session_state.shoe_completed = True

def simulate_shoe(num_hands: int = SHOE_SIZE, strategy: str = 'Genius') -> dict:
    outcomes = np.random.choice(['P', 'B', 'T'], size=num_hands, p=[0.4462, 0.4586, 0.0952])
    sequence = []
    correct = total = 0
    pattern_success = defaultdict(int)
    pattern_attempts = defaultdict(int)
    for outcome in outcomes:
        sequence.append(outcome)
        pred, conf, insights = smart_predict()
        place_result(outcome)
        if pred and outcome in ['P', 'B']:
            total += 1
            if pred == outcome:
                correct += 1
                for pattern in insights:
                    pattern_success[pattern] += 1
                    pattern_attempts[pattern] += 1
            else:
                for pattern in insights:
                    pattern_attempts[pattern] += 1
            st.session_state.sequence = sequence.copy()
            st.session_state.prediction_accuracy['total'] += 1
            if outcome in ['P', 'B']:
                st.session_state.prediction_accuracy[outcome] += 1 if pred == outcome else 0
    accuracy = (correct / total * 100) if total > 0 else 0
    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'pattern_success': dict(pattern_success),
        'pattern_attempts': dict(pattern_attempts),
        'sequence': sequence,
        'final_bankroll': st.session_state.bankroll,
        'wins': st.session_state.wins,
        'losses': st.session_state.losses,
        'strategy': strategy
    }
    try:
        with open(SIMULATION_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: Strategy={strategy}, Accuracy={accuracy:.1f}%, Correct={correct}/{total}, "
                    f"Fourgram={result['pattern_success'].get('fourgram', 0)}/{result['pattern_attempts'].get('fourgram', 0)}, "
                    f"Markov={result['pattern_success'].get('markov', 0)}/{result['pattern_attempts'].get('markov', 0)}, "
                    f"Final Bankroll=${result['final_bankroll']:.2f}, Wins={result['wins']}, Losses={result['losses']}\n")
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
    wins = sum(r['wins'] for r in results)
    losses = sum(r['losses'] for r in results)
    return {
        'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
        'std_accuracy': np.std(accuracies) if accuracies else 0.0,
        'avg_bankroll': np.mean(final_bankrolls) if final_bankrolls else 0.0,
        'std_bankroll': np.std(final_bankrolls) if final_bankrolls else 0.0,
        'wins': wins,
        'losses': losses,
        'results': results
    }

def render_setup_form():
    with st.expander("Setup", expanded=True):
        st.session_state.initial_bankroll = st.number_input("Initial Bankroll ($)", min_value=0.0, value=st.session_state.initial_bankroll, step=10.0)
        st.session_state.bankroll = st.session_state.initial_bankroll
        st.session_state.profit_lock = st.session_state.initial_bankroll
        st.session_state.base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=st.session_state.base_bet, step=0.10, format="%.2f")
        st.session_state.initial_base_bet = st.session_state.base_bet
        st.session_state.strategy = st.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index(st.session_state.strategy))
        st.session_state.target_mode = st.selectbox("Target Mode", ['Profit %', 'Wins'], index=0 if st.session_state.target_mode == 'Profit %' else 1)
        if st.session_state.target_mode == 'Profit %':
            st.session_state.target_value = st.number_input("Target Profit (%)", min_value=0.0, value=st.session_state.target_value, step=1.0)
        else:
            st.session_state.target_value = st.number_input("Target Wins", min_value=0, value=int(st.session_state.target_value), step=1)
        st.session_state.safety_net_enabled = st.checkbox("Enable Safety Net", value=st.session_state.safety_net_enabled)
        if st.session_state.safety_net_enabled:
            st.session_state.safety_net_percentage = st.number_input("Safety Net (% of Initial Bankroll)", min_value=0.0, max_value=100.0, value=st.session_state.safety_net_percentage, step=1.0)
        st.session_state.stop_loss_enabled = st.checkbox("Enable Stop Loss", value=st.session_state.stop_loss_enabled)
        if st.session_state.stop_loss_enabled:
            st.session_state.stop_loss_percentage = st.number_input("Stop Loss (% of Initial Bankroll)", min_value=0.0, max_value=100.0, value=st.session_state.stop_loss_percentage, step=1.0)
        st.session_state.profit_lock_threshold = st.number_input("Profit Lock Threshold (% of Initial Bankroll)", min_value=0.0, value=st.session_state.profit_lock_threshold, step=1.0)
        st.session_state.smart_skip = st.checkbox("Enable Smart Skip", value=st.session_state.smart_skip)
        if st.button("Reset Session", key="reset_btn"):
            reset_session()
            st.rerun()

def render_result_input():
    with st.expander("Enter Result", expanded=True):
        if st.session_state.shoe_completed:
            st.success(f"Shoe of {SHOE_SIZE} hands completed!")
        cols = st.columns(4)
        with cols[0]:
            if st.button("Player", key="player_btn", disabled=st.session_state.shoe_completed):
                place_result("P")
                st.rerun()
        with cols[1]:
            if st.button("Banker", key="banker_btn", disabled=st.session_state.shoe_completed):
                place_result("B")
                st.rerun()
        with cols[2]:
            if st.button("Tie", key="tie_btn", disabled=st.session_state.shoe_completed):
                place_result("T")
                st.rerun()
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
                            if last['Bet_Placed'] and not last['Win'] and st.session_state.loss_log:
                                if st.session_state.loss_log[-1]['result'] == last['Result']:
                                    st.session_state.loss_log.pop()
                            if st.session_state.pending_bet:
                                amount, pred = st.session_state.pending_bet
                                conf = smart_predict()[1]
                                st.session_state.advice = f"Next Bet: ${amount:.2f} on {pred}"
                            else:
                                st.session_state.advice = "No bet pending."
                            st.session_state.last_was_tie = False
                            st.session_state.non_betting_deals = previous_state.get('non_betting_deals', 0)
                            st.session_state.is_paused = previous_state.get('is_paused', False)
                            st.session_state.shoe_completed = False
                            st.success("Undone last action.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error undoing last action: {str(e)}")
        if st.button("Run Automated Shoe Simulation", key="auto_shoe_btn", disabled=st.session_state.shoe_completed):
            result = simulate_shoe()
            st.session_state.shoe_completed = True
            st.rerun()
        if st.session_state.shoe_completed and st.button("Reset and Start New Shoe", key="new_shoe_btn"):
            reset_session()
            st.session_state.shoe_completed = False
            st.rerun()

def render_prediction():
    with st.expander("Prediction", expanded=True):
        pred, conf, _ = smart_predict()
        if pred:
            bet_amount, advice = calculate_bet_amount(pred, conf)
            st.write(f"Prediction: {pred} ({conf:.1f}% confidence)")
            st.write(advice or "No bet recommended.")
        else:
            st.write("No prediction available.")

def render_insights():
    with st.expander("Insights", expanded=True):
        if not st.session_state.insights:
            st.write("No insights available.")
        else:
            insights = st.session_state.insights
            st.write(f"Volatility: {insights.get('volatility', 0.0):.2f}")
            st.write(f"Streak Score: {insights.get('streak', 0.0):.2f}")
            st.write(f"Chop Score: {insights.get('chop', 0.0):.2f}")
            st.write(f"Double Score: {insights.get('double', 0.0):.2f}")
            st.write(f"Fourgram Prediction: {insights.get('fourgram', 'None')}")
            st.write(f"Markov Prediction: {insights.get('markov', 'None')}")

def render_status():
    with st.expander("Status", expanded=True):
        st.write(f"Bankroll: ${st.session_state.bankroll:.2f}")
        st.write(f"Strategy: {st.session_state.strategy}")
        st.write(f"Wins: {st.session_state.wins}")
        st.write(f"Losses: {st.session_state.losses}")
        if st.session_state.strategy in ['T3', 'Genius']:
            st.write(f"T3 Level: {st.session_state.t3_level} (Peak: {st.session_state.t3_peak_level}, Changes: {st.session_state.t3_level_changes})")
        if st.session_state.strategy == 'Parlay16':
            st.write(f"Parlay Step: {st.session_state.parlay_step} (Peak: {st.session_state.parlay_peak_step}, Changes: {st.session_state.parlay_step_changes})")
        if st.session_state.strategy == 'Z1003.1':
            st.write(f"Z1003 Loss Count: {st.session_state.z1003_loss_count} (Changes: {st.session_state.z1003_level_changes})")
        if st.session_state.profit_lock_notification:
            st.success(st.session_state.profit_lock_notification)
            st.session_state.profit_lock_notification = None

def render_genius_insights():
    with st.expander("Genius Insights", expanded=True):
        if not st.session_state.sequence:
            st.write("No data available.")
        else:
            data = []
            for i, outcome in enumerate(st.session_state.sequence):
                pred, conf, _ = smart_predict()
                data.append({'Hand': i+1, 'Outcome': outcome, 'Prediction': pred if pred else 'None', 'Confidence': conf})
            df = pd.DataFrame(data)
            fig = px.line(df, x='Hand', y='Confidence', title='Prediction Confidence Over Time', hover_data=['Outcome', 'Prediction'])
            st.plotly_chart(fig, use_container_width=True)

def render_accuracy():
    with st.expander("Accuracy", expanded=True):
        acc = st.session_state.prediction_accuracy
        total = acc['total']
        if total > 0:
            overall = ((acc['P'] + acc['B']) / total * 100) if total > 0 else 0.0
            p_acc = (acc['P'] / total * 100) if total > 0 else 0.0
            b_acc = (acc['B'] / total * 100) if total > 0 else 0.0
            st.write(f"Overall Accuracy: {overall:.1f}%")
            st.write(f"Player Accuracy: {p_acc:.1f}%")
            st.write(f"Banker Accuracy: {b_acc:.1f}%")
        else:
            st.write("No predictions made yet.")

def render_loss_log():
    with st.expander("Loss Log", expanded=True):
        if not st.session_state.loss_log:
            st.write("No losses recorded.")
        else:
            for log in st.session_state.loss_log:
                st.write(f"Sequence: {''.join(log['sequence'])}")
                st.write(f"Predicted: {log['prediction']}, Result: {log['result']}, Confidence: {log['confidence']}%")
                st.write("Insights:")
                for k, v in log['insights'].items():
                    st.write(f"  {k}: {v}")
                st.write("---")

def render_history():
    with st.expander("History", expanded=True):
        if not st.session_state.history:
            st.write("No history available.")
        else:
            history = st.session_state.history[::-1]
            for h in history[:10]:
                amount_str = f"${h['Amount']:.2f}" if isinstance(h['Amount'], (int, float)) else "No Bet"
                st.write(f"Bet: {h['Bet'] or 'None'}, Result: {h['Result']}, Amount: {amount_str}, Win: {h['Win']}")
                if h['T3_Level'] > 1:
                    st.write(f"T3 Level: {h['T3_Level']}")
                if h['Parlay_Step'] > 1:
                    st.write(f"Parlay Step: {h['Parlay_Step']}")
                if h['Z1003_Loss_Count'] > 0:
                    st.write(f"Z1003 Loss Count: {h['Z1003_Loss_Count']}")
                st.write("---")

def render_export():
    with st.expander("Export", expanded=True):
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False)
            st.download_button(label="Download History as CSV", data=csv, file_name="baccarat_history.csv", mime="text/csv")
        if os.path.exists(SIMULATION_LOG):
            with open(SIMULATION_LOG, 'r', encoding='utf-8') as f:
                log_content = f.read()
            st.download_button(label="Download Simulation Log", data=log_content, file_name="simulation_log.txt", mime="text/plain")

def render_simulation():
    with st.expander("Simulation", expanded=True):
        num_shoes = st.number_input("Number of Shoes to Simulate", min_value=1, max_value=100, value=10, step=1)
        strategy = st.selectbox("Simulation Strategy", STRATEGIES, index=STRATEGIES.index(st.session_state.strategy))
        if st.button("Run Simulation", key="run_sim_btn"):
            with st.spinner("Running simulation..."):
                result = simulate_to_target(strategy, num_shoes)
                st.write(f"Average Accuracy: {result['avg_accuracy']:.1f}% (Â±{result['std_accuracy']:.1f}%)")
                st.write(f"Average Final Bankroll: ${result['avg_bankroll']:.2f} (Â±${result['std_bankroll']:.2f})")
                st.write(f"Total Wins: {result['wins']}")
                st.write(f"Total Losses: {result['losses']}")
                fig = go.Figure()
                for i, res in enumerate(result['results']):
                    fig.add_trace(go.Scatter(x=list(range(1, len(res['sequence']) + 1)), y=[r['final_bankroll'] for r in result['results'][:i+1]], mode='lines', name=f"Shoe {i+1}"))
                fig.update_layout(title="Bankroll Over Shoes", xaxis_title="Hand", yaxis_title="Bankroll ($)", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

def render_bead_plate():
    with st.expander("Bead Plate", expanded=True):
        st.markdown("**Bead Plate**")
        sequence = st.session_state.sequence[- (GRID_ROWS * GRID_COLS):]  # Get the last GRID_ROWS * GRID_COLS results
        grid = [['' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        
        for i, result in enumerate(sequence[::-1]):  # Reverse to start from the latest
            if result in ['P', 'B']:
                row = i // GRID_COLS
                col = i % GRID_COLS
                if row < GRID_ROWS:
                    color = '#1E90FF' if result == 'P' else '#FF4040'  # Blue for Player, Red for Banker
                    grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'

        for row in grid:
            st.markdown(' '.join(row), unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Mang Baccarat")
    apply_custom_css()
    st.title("Mang Baccarat")
    initialize_session_state()
    col1, col2 = st.columns([2, 1])
    with col1:
        render_setup_form()
        render_result_input()
        render_prediction()
        render_insights()
        render_bead_plate()  # Add bead plate here
    with col2:
        render_status()
        render_genius_insights()
        render_accuracy()
        render_loss_log()
        render_history()
        render_export()
        render_simulation()
        render_profit_dashboard()

if __name__ == "__main__":
    main()
