import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import numpy as np
from typing import Tuple, Dict, Optional, List
import uuid
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Constants ---
SESSION_FILE = "online_users.txt"
SIMULATION_LOG = "simulation_log.txt"
PARLAY_TABLE = {
    i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
        (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
        (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
    ], 1)
}
STRATEGIES = ["Grok T3", "Flatbet", "Parlay16", "Z1003.1"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
LOSS_LOG_LIMIT = 50
WINDOW_SIZE = 50
GROK_T3_SEQUENCE_LENGTH = 10

# --- CSS for Professional Styling ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* General Styling */
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
    /* Result Buttons */
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
    /* Bead Plate */
    .bead-plate {
        background-color: #edf2f7;
        padding: 10px;
        border-radius: 8px;
        overflow-x: auto;
    }
    /* Responsive Design */
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

# --- Grok T3 Functions ---
def generate_baccarat_data(num_games=10000):
    outcomes = ['P', 'B']
    return [random.choice(outcomes) for _ in range(num_games)]

def prepare_data(outcomes, sequence_length=10):
    le = LabelEncoder()
    encoded_outcomes = le.fit_transform(outcomes)
    X, y = [], []
    for i in range(len(encoded_outcomes) - sequence_length):
        X.append(encoded_outcomes[i:i + sequence_length])
        y.append(encoded_outcomes[i + sequence_length])
    return np.array(X), np.array(y), le

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
        'bankroll': 0.0,
        'base_bet': 0.10,
        'initial_base_bet': 0.10,
        'sequence': [],
        'pending_bet': None,
        'strategy': 'Grok T3',
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
        'target_value': 10.0,
        'initial_bankroll': 0.0,
        'target_hit': False,
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
        'consecutive_losses': 0,
        'loss_log': [],
        'last_was_tie': False,
        'insights': {},
        'pattern_volatility': 0.0,
        'pattern_success': defaultdict(int),
        'pattern_attempts': defaultdict(int),
        'safety_net_percentage': 10.0,
        'safety_net_enabled': True,
        'grok_t3_model': None,
        'grok_t3_le': None
    }
    for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double', 'LB 6 Rep']:
        defaults['pattern_success'][pattern] = 0
        defaults['pattern_attempts'][pattern] = 0
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.strategy not in STRATEGIES:
        st.session_state.strategy = 'Grok T3'

def reset_session():
    initialize_session_state()
    st.session_state.update({
        'bankroll': st.session_state.initial_bankroll,
        'base_bet': 0.10,
        'initial_base_bet': 0.10,
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
        'advice': "Session reset: Target reached.",
        'history': [],
        'wins': 0,
        'losses': 0,
        'target_hit': False,
        'consecutive_losses': 0,
        'loss_log': [],
        'last_was_tie': False,
        'insights': {},
        'pattern_volatility': 0.0,
        'pattern_success': defaultdict(int),
        'pattern_attempts': defaultdict(int),
        'safety_net_percentage': 10.0,
        'safety_net_enabled': True,
        'grok_t3_model': None,
        'grok_t3_le': None
    })
    for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double', 'LB 6 Rep']:
        st.session_state.pattern_success[pattern] = 0
        st.session_state.pattern_attempts[pattern] = 0

# --- Prediction Logic for Grok T3 ---
def predict_grok_t3() -> Tuple[Optional[str], float, Dict]:
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B']]
    insights = {}
    if len(sequence) < GROK_T3_SEQUENCE_LENGTH:
        return None, 0.0, {'Status': f'Waiting for {GROK_T3_SEQUENCE_LENGTH - len(sequence)} more results'}
    if st.session_state.grok_t3_model is None or st.session_state.grok_t3_le is None:
        return None, 0.0, {'Status': 'Model not initialized'}
    
    # Model Prediction
    encoded_input = st.session_state.grok_t3_le.transform(sequence[-GROK_T3_SEQUENCE_LENGTH:])
    input_array = np.array([encoded_input])
    prediction_probs = st.session_state.grok_t3_model.predict_proba(input_array)[0]
    predicted_class = np.argmax(prediction_probs)
    predicted_outcome = st.session_state.grok_t3_le.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction_probs) * 100

    # Initialize bet selection and pattern confidences
    bet_selection = None
    pattern_confidences = []

    # LB 6 Rep Logic
    if len(sequence) >= GROK_T3_SEQUENCE_LENGTH + 5:
        sixth_prior = sequence[-6]
        outcome_index = st.session_state.grok_t3_le.transform([sixth_prior])[0]
        sixth_confidence = prediction_probs[outcome_index] * 100
        if sixth_confidence > 40:
            bet_selection = sixth_prior
            pattern_confidences.append((sixth_confidence, sixth_prior, 'LB 6 Rep'))
            insights['LB 6 Rep'] = f"Mirroring 6th prior: {sixth_prior} (Confidence: {sixth_confidence:.1f}%)"
        else:
            insights['LB 6 Rep'] = f"Low confidence for 6th prior: {sixth_confidence:.1f}%"
    else:
        insights['LB 6 Rep'] = 'Not enough results for 6th prior'

    # Helper function to get n-grams
    def get_ngrams(seq, n):
        return [''.join(seq[i:i+n]) for i in range(len(seq)-n+1)]

    # Bigram Analysis
    if len(sequence) >= 2:
        bigrams = get_ngrams(sequence, 2)
        last_bigram = bigrams[-1]
        bigram_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(bigrams)-1):
            bigram = bigrams[i]
            next_outcome = sequence[i+2] if i+2 < len(sequence) else None
            if next_outcome:
                bigram_counts[bigram][next_outcome] += 1
        if last_bigram in bigram_counts:
            total = sum(bigram_counts[last_bigram].values())
            if total > 0:
                p_prob = bigram_counts[last_bigram]['P'] / total * 100
                b_prob = bigram_counts[last_bigram]['B'] / total * 100
                bigram_pred = 'P' if p_prob > b_prob else 'B'
                bigram_conf = max(p_prob, b_prob)
                if bigram_conf > 50:
                    pattern_confidences.append((bigram_conf, bigram_pred, 'bigram'))
                insights['bigram'] = f"Bigram {last_bigram}: P={p_prob:.1f}%, B={b_prob:.1f}%"

    # Trigram Analysis
    if len(sequence) >= 3:
        trigrams = get_ngrams(sequence, 3)
        last_trigram = trigrams[-1]
        trigram_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(trigrams)-1):
            trigram = trigrams[i]
            next_outcome = sequence[i+3] if i+3 < len(sequence) else None
            if next_outcome:
                trigram_counts[trigram][next_outcome] += 1
        if last_trigram in trigram_counts:
            total = sum(trigram_counts[last_trigram].values())
            if total > 0:
                p_prob = trigram_counts[last_trigram]['P'] / total * 100
                b_prob = trigram_counts[last_trigram]['B'] / total * 100
                trigram_pred = 'P' if p_prob > b_prob else 'B'
                trigram_conf = max(p_prob, b_prob)
                if trigram_conf > 50:
                    pattern_confidences.append((trigram_conf, trigram_pred, 'trigram'))
                insights['trigram'] = f"Trigram {last_trigram}: P={p_prob:.1f}%, B={b_prob:.1f}%"

    # Fourgram Analysis
    if len(sequence) >= 4:
        fourgrams = get_ngrams(sequence, 4)
        last_fourgram = fourgrams[-1]
        fourgram_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(fourgrams)-1):
            fourgram = fourgrams[i]
            next_outcome = sequence[i+4] if i+4 < len(sequence) else None
            if next_outcome:
                fourgram_counts[fourgram][next_outcome] += 1
        if last_fourgram in fourgram_counts:
            total = sum(fourgram_counts[last_fourgram].values())
            if total > 0:
                p_prob = fourgram_counts[last_fourgram]['P'] / total * 100
                b_prob = fourgram_counts[last_fourgram]['B'] / total * 100
                fourgram_pred = 'P' if p_prob > b_prob else 'B'
                fourgram_conf = max(p_prob, b_prob)
                if fourgram_conf > 50:
                    pattern_confidences.append((fourgram_conf, fourgram_pred, 'fourgram'))
                insights['fourgram'] = f"Fourgram {last_fourgram}: P={p_prob:.1f}%, B={b_prob:.1f}%"

    # Streak Analysis
    if len(sequence) >= 3:
        last_three = sequence[-3:]
        if all(x == last_three[0] for x in last_three):  # Streak of 3
            streak_outcome = last_three[0]
            streak_counts = defaultdict(int)
            streak_length = 1
            for i in range(len(sequence)-2, -1, -1):
                if sequence[i] == streak_outcome:
                    streak_length += 1
                else:
                    break
            for i in range(len(sequence)-3):
                if sequence[i:i+3] == [streak_outcome]*3:
                    next_outcome = sequence[i+3] if i+3 < len(sequence) else None
                    if next_outcome:
                        streak_counts[next_outcome] += 1
            total = sum(streak_counts.values())
            if total > 0:
                p_prob = streak_counts['P'] / total * 100 if 'P' in streak_counts else 0
                b_prob = streak_counts['B'] / total * 100 if 'B' in streak_counts else 0
                streak_pred = streak_outcome  # Continue streak
                streak_conf = p_prob if streak_pred == 'P' else b_prob
                if streak_conf > 50:
                    pattern_confidences.append((streak_conf, streak_pred, 'streak'))
                insights['streak'] = f"Streak of {streak_outcome} (length {streak_length}): P={p_prob:.1f}%, B={b_prob:.1f}%"
        else:
            insights['streak'] = "No streak detected (need 3 identical outcomes)"

    # Chop Analysis
    if len(sequence) >= 4:
        last_four = sequence[-4:]
        if last_four == ['P', 'B', 'P', 'B'] or last_four == ['B', 'P', 'B', 'P']:  # Chop pattern
            chop_counts = defaultdict(int)
            for i in range(len(sequence)-4):
                if sequence[i:i+4] in [['P', 'B', 'P', 'B'], ['B', 'P', 'B', 'P']]:
                    next_outcome = sequence[i+4] if i+4 < len(sequence) else None
                    if next_outcome:
                        chop_counts[next_outcome] += 1
            total = sum(chop_counts.values())
            if total > 0:
                p_prob = chop_counts['P'] / total * 100 if 'P' in chop_counts else 0
                b_prob = chop_counts['B'] / total * 100 if 'B' in chop_counts else 0
                chop_pred = 'P' if last_four[-1] == 'B' else 'B'  # Continue alternation
                chop_conf = p_prob if chop_pred == 'P' else b_prob
                if chop_conf > 50:
                    pattern_confidences.append((chop_conf, chop_pred, 'chop'))
                insights['chop'] = f"Chop pattern: P={p_prob:.1f}%, B={b_prob:.1f}%"
        else:
            insights['chop'] = "No chop pattern detected (need PBPB or BPBP)"

    # Double Analysis
    if len(sequence) >= 4:
        last_four = sequence[-4:]
        if last_four == ['P', 'P', 'B', 'B'] or last_four == ['B', 'B', 'P', 'P']:  # Double pattern
            double_counts = defaultdict(int)
            for i in range(len(sequence)-4):
                if sequence[i:i+4] in [['P', 'P', 'B', 'B'], ['B', 'B', 'P', 'P']]:
                    next_outcome = sequence[i+4] if i+4 < len(sequence) else None
                    if next_outcome:
                        double_counts[next_outcome] += 1
            total = sum(double_counts.values())
            if total > 0:
                p_prob = double_counts['P'] / total * 100 if 'P' in double_counts else 0
                b_prob = double_counts['B'] / total * 100 if 'B' in double_counts else 0
                double_pred = last_four[-2]  # Continue double pattern
                double_conf = p_prob if double_pred == 'P' else b_prob
                if double_conf > 50:
                    pattern_confidences.append((double_conf, double_pred, 'double'))
                insights['double'] = f"Double pattern: P={p_prob:.1f}%, B={b_prob:.1f}%"
        else:
            insights['double'] = "No double pattern detected (need PPBB or BBPP)"

    # Select bet based on highest confidence pattern
    if pattern_confidences:
        max_conf, best_pred, best_pattern = max(pattern_confidences, key=lambda x: x[0])
        bet_selection = best_pred
        insights['Selected Pattern'] = f"Chose {best_pattern} with confidence {max_conf:.1f}%"

    insights['Model Confidence'] = f"P: {prediction_probs[st.session_state.grok_t3_le.transform(['P'])[0]]*100:.1f}%, B: {prediction_probs[st.session_state.grok_t3_le.transform(['B'])[0]]*100:.1f}%"
    
    # Update pattern volatility (standard deviation of pattern confidences)
    if pattern_confidences:
        confidences = [conf for conf, _, _ in pattern_confidences]
        st.session_state.pattern_volatility = np.std(confidences) if len(confidences) > 1 else 0.0
    else:
        st.session_state.pattern_volatility = 0.0

    return bet_selection, confidence, insights

# --- Betting Logic ---
def check_target_hit() -> bool:
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
    unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
    return unit_profit >= st.session_state.target_value

def update_grok_t3_level():
    if len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        losses = st.session_state.t3_results.count('L')
        old_level = st.session_state.t3_level
        if wins > losses:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses > wins:
            st.session_state.t3_level += 1
        if old_level != st.session_state.t3_level:
            st.session_state.t3_level_changes += 1
        st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, st.session_state.t3_level)
        st.session_state.t3_results = []

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], Optional[str]]:
    if len(st.session_state.sequence) < GROK_T3_SEQUENCE_LENGTH:
        return None, f"No bet: Need {GROK_T3_SEQUENCE_LENGTH - len(st.session_state.sequence)} more results"
    if st.session_state.consecutive_losses >= 3 and conf < 45.0:
        return None, f"No bet: Paused after {st.session_state.consecutive_losses} losses"
    if st.session_state.pattern_volatility > 0.6:
        return None, f"No bet: High pattern volatility ({st.session_state.pattern_volatility:.2f})"
    if pred is None:
        return None, f"No bet: No valid prediction (pattern confidences too low)"
    
    if st.session_state.strategy == 'Z1003.1':
        if st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
            return None, "No bet: Stopped after three losses (Z1003.1 rule)"
        bet_amount = st.session_state.base_bet + (st.session_state.z1003_loss_count * 100)
    elif st.session_state.strategy == 'Flatbet':
        bet_amount = st.session_state.base_bet
    elif st.session_state.strategy == 'Grok T3':
        bet_amount = st.session_state.base_bet * st.session_state.t3_level
    else:  # Parlay16
        key = 'base' if st.session_state.parlay_using_base else 'parlay'
        bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)
    
    if st.session_state.safety_net_enabled:
        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if (bet_amount > st.session_state.bankroll or
            st.session_state.bankroll - bet_amount < safe_bankroll * 0.5 or
            bet_amount > st.session_state.bankroll * 0.10):
            if st.session_state.strategy == 'Grok T3':
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
            return None, "No bet: Risk too high for current bankroll. Level/step reset to 1."
    return bet_amount, f"Next Bet: ${bet_amount:.2f} on {pred}"

def place_result(result: str):
    if st.session_state.target_hit:
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
        "z1003_level_changes": st.session_state.z1003_level_changes,
        "pending_bet": st.session_state.pending_bet,
        "wins": st.session_state.wins,
        "losses": st.session_state.losses,
        "prediction_accuracy": st.session_state.prediction_accuracy.copy(),
        "consecutive_losses": st.session_state.consecutive_losses,
        "t3_level_changes": st.session_state.t3_level_changes,
        "parlay_step_changes": st.session_state.parlay_step_changes,
        "pattern_volatility": st.session_state.pattern_volatility,
        "pattern_success": st.session_state.pattern_success.copy(),
        "pattern_attempts": st.session_state.pattern_attempts.copy(),
        "safety_net_percentage": st.session_state.safety_net_percentage,
        "safety_net_enabled": st.session_state.safety_net_enabled,
        "grok_t3_model": st.session_state.grok_t3_model,
        "grok_t3_le": st.session_state.grok_t3_le
    }
    if st.session_state.pending_bet and result != 'T':
        bet_amount, selection = st.session_state.pending_bet
        win = result == selection
        bet_placed = True
        if win:
            st.session_state.bankroll += bet_amount * (0.95 if selection == 'B' else 1.0)
            if st.session_state.strategy == 'Grok T3':
                st.session_state.t3_results.append('W')
                if len(st.session_state.t3_results) == 1:  # First-step win
                    old_level = st.session_state.t3_level
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                    if old_level != st.session_state.t3_level:
                        st.session_state.t3_level_changes += 1
                    st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, st.session_state.t3_level)
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
            st.session_state.wins += 1
            st.session_state.prediction_accuracy[selection] += 1
            st.session_state.consecutive_losses = 0
            for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double', 'LB 6 Rep']:
                if pattern in st.session_state.insights:
                    st.session_state.pattern_success[pattern] += 1
                    st.session_state.pattern_attempts[pattern] += 1
        else:
            st.session_state.bankroll -= bet_amount
            if st.session_state.strategy == 'Grok T3':
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
            st.session_state.losses += 1
            st.session_state.consecutive_losses += 1
            _, conf, _ = predict_grok_t3()
            st.session_state.loss_log.append({
                'sequence': st.session_state.sequence[-10:],
                'prediction': selection,
                'result': result,
                'confidence': f"{conf:.1f}",
                'insights': st.session_state.insights.copy()
            })
            if len(st.session_state.loss_log) > LOSS_LOG_LIMIT:
                st.session_state.loss_log = st.session_state.loss_log[-LOSS_LOG_LIMIT:]
            for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double', 'LB 6 Rep']:
                if pattern in st.session_state.insights:
                    st.session_state.pattern_attempts[pattern] += 1
        st.session_state.prediction_accuracy['total'] += 1
        st.session_state.pending_bet = None
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
        "Z1003_Bet_Factor": None,
        "Previous_State": previous_state,
        "Bet_Placed": bet_placed
    })
    if len(st.session_state.history) > HISTORY_LIMIT:
        st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]
    if check_target_hit():
        st.session_state.target_hit = True
        return
    pred, conf, insights = predict_grok_t3()
    if st.session_state.strategy == 'Z1003.1' and st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
        bet_amount, advice = None, "No bet: Stopped after three losses (Z1003.1 rule)"
    else:
        bet_amount, advice = calculate_bet_amount(pred, conf)
    st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
    st.session_state.advice = advice
    st.session_state.insights = insights
    if st.session_state.strategy == 'Grok T3':
        update_grok_t3_level()

# --- Simulation Logic ---
def simulate_shoe(num_hands: int = 80) -> Dict:
    outcomes = np.random.choice(
        ['P', 'B', 'T'],
        size=num_hands,
        p=[0.4462, 0.4586, 0.0952]
    )
    sequence = []
    correct = total = 0
    pattern_success = defaultdict(int)
    pattern_attempts = defaultdict(int)
    for outcome in outcomes:
        sequence.append(outcome)
        pred, conf, insights = predict_grok_t3()
        if pred and outcome in ['P', 'B']:
            total += 1
            if pred == outcome:
                correct += 1
                for pattern in insights:
                    if pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double', 'LB 6 Rep']:
                        pattern_success[pattern] += 1
                        pattern_attempts[pattern] += 1
            else:
                for pattern in insights:
                    if pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double', 'LB 6 Rep']:
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
        'sequence': sequence
    }
    try:
        with open(SIMULATION_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: Accuracy={accuracy:.1f}%, Correct={correct}/{total}, "
                    f"LB_6_Rep={result['pattern_success'].get('LB 6 Rep', 0)}/{result['pattern_attempts'].get('LB 6 Rep', 0)}, "
                    f"Bigram={result['pattern_success'].get('bigram', 0)}/{result['pattern_attempts'].get('bigram', 0)}, "
                    f"Trigram={result['pattern_success'].get('trigram', 0)}/{result['pattern_attempts'].get('trigram', 0)}, "
                    f"Fourgram={result['pattern_success'].get('fourgram', 0)}/{result['pattern_attempts'].get('fourgram', 0)}, "
                    f"Streak={result['pattern_success'].get('streak', 0)}/{result['pattern_attempts'].get('streak', 0)}, "
                    f"Chop={result['pattern_success'].get('chop', 0)}/{result['pattern_attempts'].get('chop', 0)}, "
                    f"Double={result['pattern_success'].get('double', 0)}/{result['pattern_attempts'].get('double', 0)}\n")
    except PermissionError:
        st.error("Unable to write to simulation log.")
    return result

# --- UI Components ---
def render_setup_form():
    with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
        with st.form("setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.bankroll, step=10.0)
                base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=max(st.session_state.base_bet, 0.10), step=0.10)
            with col2:
                betting_strategy = st.selectbox(
                    "Betting Strategy", STRATEGIES,
                    index=STRATEGIES.index(st.session_state.strategy),
                    help="Grok T3: Adjusts bet size based on wins/losses with pattern analysis. Flatbet: Fixed bet size. Parlay16: 16-step progression. Z1003.1: Resets after first win, stops after three losses."
                )
                target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0)
                target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
            safety_net_enabled = st.checkbox(
                "Enable Safety Net",
                value=st.session_state.safety_net_enabled,
                help="Ensures a percentage of the initial bankroll is preserved."
            )
            safety_net_percentage = st.session_state.safety_net_percentage
            if safety_net_enabled:
                safety_net_percentage = st.number_input(
                    "Safety Net Percentage (%)",
                    min_value=0.0, max_value=50.0, value=st.session_state.safety_net_percentage, step=5.0
                )
            if st.form_submit_button("Start Session"):
                if bankroll <= 0:
                    st.error("Bankroll must be positive.")
                elif base_bet < 0.10:
                    st.error("Base bet must be at least $0.10.")
                elif base_bet > bankroll:
                    st.error("Base bet cannot exceed bankroll.")
                else:
                    # Train Grok T3 model
                    outcomes = generate_baccarat_data()
                    X, y, le = prepare_data(outcomes, GROK_T3_SEQUENCE_LENGTH)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    st.session_state.update({
                        'bankroll': bankroll,
                        'base_bet': base_bet,
                        'initial_base_bet': base_bet,
                        'strategy': betting_strategy,
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
                        'advice': "",
                        'return_to_start': False,
                        'history': [],
                        'wins': 0,
                        'losses': 0,
                        'target_mode': target_mode,
                        'target_value': target_value,
                        'initial_bankroll': bankroll,
                        'target_hit': False,
                        'Prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
                        'consecutive_losses': 0,
                        'loss_log': [],
                        'last_was_tie': False,
                        'insights': {},
                        'pattern_volatility': 0.0,
                        'pattern_success': defaultdict(int),
                        'pattern_attempts': defaultdict(int),
                        'safety_net_percentage': safety_net_percentage,
                        'safety_net_enabled': safety_net_enabled,
                        'grok_t3_model': model,
                        'grok_t3_le': le
                    })
                    for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double', 'LB 6 Rep']:
                        st.session_state.pattern_success[pattern] = 0
                        st.session_state.pattern_attempts[pattern] = 0
                    st.success(f"Session started with {betting_strategy} strategy!")

def render_result_input():
    with st.expander("Enter Result", expanded=True):
        cols = st.columns(4)
        with cols[0]:
            if st.button("Player", key="player_btn", help="Record a Player win"):
                place_result("P")
                st.rerun()
        with cols[1]:
            if st.button("Banker", key="banker_btn", help="Record a Banker win"):
                place_result("B")
                st.rerun()
        with cols[2]:
            if st.button("Tie", key="tie_btn", help="Record a Tie"):
                place_result("T")
                st.rerun()
        with cols[3]:
            if st.button("Undo Last", key="undo_btn", help="Undo the last action"):
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
                                conf = predict_grok_t3()[1]
                                st.session_state.advice = f"Next Bet: ${amount:.2f} on {pred}"
                            else:
                                st.session_state.advice = "No bet pending."
                            st.session_state.last_was_tie = False
                            st.success("Undone last action.")
                            st.rerun()
                        else:
                            st.session_state.sequence.pop()
                            st.session_state.pending_bet = None
                            st.session_state.advice = "No bet pending."
                            st.session_state.last_was_tie = False
                            st.success("Undone last result.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error undoing last action: {str(e)}")

def render_bead_plate():
    with st.expander("Bead Plate", expanded=True):
        sequence = st.session_state.sequence[-90:]
        grid = [[] for _ in range(15)]
        for i, result in enumerate(sequence):
            col_index = i // 6
            if col_index < 15:
                grid[col_index].append(result)
        for col in grid:
            while len(col) < 6:
                col.append('')
        bead_plate_html = "<div class='bead-plate' style='display: flex; flex-direction: row; gap: 5px;'>"
        for col in grid:
            col_html = "<div style='display: flex; flex-direction: column; gap: 5px;'>"
            for result in col:
                style = (
                    "width: 24px; height: 24px; border: 1px solid #e2e8f0; border-radius: 50%;" if result == '' else
                    f"width: 24px; height: 24px; background-color: {'#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'}; border-radius: 50%;"
                )
                col_html += f"<div style='{style}'></div>"
            col_html += "</div>"
            bead_plate_html += col_html
        bead_plate_html += "</div>"
        st.markdown(bead_plate_html, unsafe_allow_html=True)

def render_prediction():
    with st.expander("Prediction", expanded=True):
        if st.session_state.pending_bet:
            amount, side = st.session_state.pending_bet
            color = '#3182ce' if side == 'P' else '#e53e3e'
            st.markdown(f"<div style='background-color: #edf2f7; padding: 15px; border-radius: 8px;'><h4 style='color:{color}; margin:0;'>Prediction: {side} | Bet: ${amount:.2f}</h4></div>", unsafe_allow_html=True)
        elif not st.session_state.target_hit:
            st.info(st.session_state.advice)

def render_insights():
    with st.expander("Prediction Insights"):
        if st.session_state.insights:
            for factor, contribution in st.session_state.insights.items():
                st.markdown(f"**{factor}**: {contribution}")
        if st.session_state.pattern_volatility > 0.6:
            st.warning(f"High Pattern Volatility: {st.session_state.pattern_volatility:.2f} (Betting paused)")

def render_status():
    with st.expander("Session Status", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
            st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
            st.markdown(f"**Safety Net**: {'Enabled' if st.session_state.safety_net_enabled else 'Disabled'}"
                        f"{' | ' + str(st.session_state.safety_net_percentage) + '%' if st.session_state.safety_net_enabled else ''}")
        with col2:
            strategy_status = f"**Strategy**: {st.session_state.strategy}"
            if st.session_state.strategy == 'Grok T3':
                strategy_status += f"<br>Level: {st.session_state.t3_level} | Peak: {st.session_state.t3_peak_level}<br>Changes: {st.session_state.t3_level_changes}"
            elif st.session_state.strategy == 'Parlay16':
                strategy_status += f"<br>Steps: {st.session_state.parlay_step}/16 | Peak: {st.session_state.parlay_peak_step}<br>Changes: {st.session_state.parlay_step_changes} | Wins: {st.session_state.parlay_wins}"
            elif st.session_state.strategy == 'Z1003.1':
                strategy_status += f"<br>Loss Count: {st.session_state.z1003_loss_count}<br>Changes: {st.session_state.z1003_level_changes} | Continue: {st.session_state.z1003_continue}"
            st.markdown(strategy_status, unsafe_allow_html=True)
        st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
        st.markdown(f"**Online Users**: {track_user_session()}")
        if st.session_state.initial_base_bet > 0 and st.session_state.initial_bankroll > 0:
            profit = st.session_state.bankroll - st.session_state.initial_bankroll
            units_profit = profit / st.session_state.initial_base_bet
            st.markdown(f"**Profit**: {units_profit:.2f} units (${profit:.2f})")
        else:
            st.markdown("**Profit**: 0.00 units ($0.00)")

def render_accuracy():
    with st.expander("Prediction Accuracy"):
        total = st.session_state.prediction_accuracy['total']
        if total > 0:
            p_accuracy = (st.session_state.prediction_accuracy['P'] / total) * 100
            b_accuracy = (st.session_state.prediction_accuracy['B'] / total) * 100
            st.markdown(f"**Player Bets**: {st.session_state.prediction_accuracy['P']}/{total} ({p_accuracy:.1f}%)")
            st.markdown(f"**Banker Bets**: {st.session_state.prediction_accuracy['B']}/{total} ({b_accuracy:.1f}%)")
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

def render_loss_log():
    with st.expander("Recent Losses"):
        if st.session_state.loss_log:
            st.dataframe([
                {
                    "Sequence": ", ".join(log['sequence']),
                    "Prediction": log['prediction'],
                    "Result": log['result'],
                    "Confidence": f"{log['confidence']}%",
                    "Insights": "; ".join([f"{k}: {v}" for k, v in log['insights'].items()])
                }
                for log in st.session_state.loss_log[-5:]
            ], use_container_width=True)

def render_history():
    with st.expander("Bet History"):
        if st.session_state.history:
            n = st.slider("Show last N bets", 5, 50, 10)
            st.dataframe([
                {
                    "Bet": h["Bet"] if h["Bet"] else "-",
                    "Result": h["Result"],
                    "Amount": f"${h['Amount']:.2f}" if h["Bet_Placed"] else "-",
                    "Outcome": "Win" if h["Win"] else "Loss" if h["Bet_Placed"] else "-",
                    "T3_Level": h["T3_Level"] if st.session_state.strategy == 'Grok T3' else "-",
                    "Parlay_Step": h["Parlay_Step"] if st.session_state.strategy == 'Parlay16' else "-",
                    "Z1003_Loss_Count": h["Z1003_Loss_Count"] if st.session_state.strategy == 'Z1003.1' else "-",
                }
                for h in st.session_state.history[-n:]
            ], use_container_width=True)

def render_export():
    with st.expander("Export Session"):
        if st.button("Download Session Data"):
            csv_data = "Bet,Result,Amount,Win,T3_Level,Parlay_Step,Z1003_Loss_Count\n"
            for h in st.session_state.history:
                csv_data += f"{h['Bet'] or '-'},{h['Result']},${h['Amount']:.2f},{h['Win']},{h['T3_Level']},{h['Parlay_Step']},{h['Z1003_Loss_Count']}\n"
            st.download_button("Download CSV", csv_data, "session_data.csv", "text/csv")

def render_simulation():
    with st.expander("Run Simulation"):
        num_hands = st.number_input("Number of Hands to Simulate", min_value=10, max_value=200, value=80, step=10)
        if st.button("Run Simulation"):
            result = simulate_shoe(num_hands)
            st.write(f"**Simulation Results**")
            st.write(f"Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']} correct)")
            st.write("**Pattern Performance**:")
            for pattern in result['pattern_success']:
                success = result['pattern_success'][pattern]
                attempts = result['pattern_attempts'][pattern]
                st.write(f"{pattern}: {success}/{attempts} ({success/attempts*100:.1f}%)" if attempts > 0 else f"{pattern}: 0/0 (0%)")
            st.write("Results logged to simulation_log.txt")

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="MANG Baccarat")
    apply_custom_css()
    st.title("MANG Baccarat")
    initialize_session_state()
    col1, col2 = st.columns([2, 1])
    with col1:
        render_setup_form()
        render_result_input()
        render_bead_plate()
        render_prediction()
        render_insights()
    with col2:
        render_status()
        render_accuracy()
        render_loss_log()
        render_history()
        render_export()
        render_simulation()

if __name__ == "__main__":
    main()
