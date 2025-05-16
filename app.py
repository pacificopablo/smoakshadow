import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import numpy as np
from typing import Tuple, Dict, Optional, List
import uuid
import logging
import tempfile

# Placeholder for profit_enhancements module
def adjust_safety_net():
    return st.session_state.safety_net_percentage  # Return current safety net percentage

def recommend_strategy(sequence):
    return "T3"  # Default recommendation

def enhanced_z1003_bet(loss_count, base_bet):
    factors = [1.0, 2.0, 3.0]  # Simple progression
    return base_bet * factors[min(loss_count, len(factors) - 1)]

def calculate_roi():
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    return (profit / st.session_state.initial_bankroll * 100) if st.session_state.initial_bankroll > 0 else 0.0

def render_profit_dashboard():
    st.markdown("**Profit Dashboard**")
    st.markdown(f"ROI: {calculate_roi():.2f}%")

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
SESSION_FILE = os.path.join(tempfile.gettempdir(), "online_users.txt")
SIMULATION_LOG = os.path.join(tempfile.gettempdir(), "simulation_log.txt")
PARLAY_TABLE = {
    i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
        (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
        (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
    ], 1)
}
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Z1003.1"]
SEQUENCE_LIMIT = 100
HISTORY_LIMIT = 1000
LOSS_LOG_LIMIT = 50
WINDOW_SIZE = 50
SHOE_SIZE = 100

# --- CSS for Professional Styling ---
def apply_custom_css():
    try:
        st.markdown("""
        <style>
        body {
            font-family: 'Inter', sans-serif;
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
        h2 {
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
        .stNumberInput input, .stSelectbox select {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            padding: 10px;
            font-size: 14px;
        }
        .stRadio label, .stCheckbox label {
            font-size: 14px;
            color: #4a5568;
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
            h2 {
                font-size: 1.25rem;
            }
            .stButton > button {
                width: 100%;
                padding: 12px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"CSS rendering failed: {e}")
        st.error("Error rendering styles. Please refresh.")

# --- Session Tracking ---
def track_user_session() -> int:
    try:
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(time.time())
        sessions = {}
        current_time = datetime.now()
        if os.path.exists(SESSION_FILE):
            try:
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
                st.warning("Unable to read session file. Session tracking may be limited.")
                return 0
        sessions[st.session_state.session_id] = current_time
        try:
            with open(SESSION_FILE, 'w', encoding='utf-8') as f:
                for session_id, last_seen in sessions.items():
                    f.write(f"{session_id},{last_seen.isoformat()}\n")
        except PermissionError:
            st.warning("Unable to write to session file. Session tracking may be limited.")
            return len(sessions)
        return len(sessions)
    except Exception as e:
        logger.error(f"Session tracking failed: {e}")
        return 0

# --- Session State Management ---
def initialize_session_state():
    try:
        defaults = {
            'bankroll': 0.0,
            'base_bet': 0.10,
            'initial_base_bet': 0.10,
            'sequence': [],
            'pending_bet': None,
            'strategy': 'T3',
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
            'ai_automation_enabled': True,
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
            st.session_state.strategy = 'T3'
    except Exception as e:
        logger.error(f"Session state initialization failed: {e}")
        st.error("Error initializing session. Please refresh.")

def reset_session():
    try:
        # Store current setup values before reset
        setup_values = {
            'initial_bankroll': st.session_state.initial_bankroll,
            'base_bet': st.session_state.base_bet,
            'initial_base_bet': st.session_state.initial_base_bet,
            'strategy': st.session_state.strategy,
            'target_mode': st.session_state.target_mode,
            'target_value': st.session_state.target_value,
            'safety_net_enabled': st.session_state.safety_net_enabled,
            'safety_net_percentage': st.session_state.safety_net_percentage,
            'ai_automation_enabled': st.session_state.ai_automation_enabled
        }

        # Reset session state to defaults
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
            'pattern_success': defaultdict(int),
            'pattern_attempts': defaultdict(int),
            'safety_net_percentage': setup_values['safety_net_percentage'],
            'safety_net_enabled': setup_values['safety_net_enabled'],
            'ai_automation_enabled': setup_values['ai_automation_enabled'],
            'shoe_completed': False
        })
        st.session_state.pattern_success['markov'] = 0
        st.session_state.pattern_attempts['markov'] = 0
        st.session_state.pattern_success['fourgram'] = 0
        st.session_state.pattern_attempts['fourgram'] = 0
    except Exception as e:
        logger.error(f"Session reset failed: {e}")
        st.error("Error resetting session. Please refresh.")

# --- Prediction Logic ---
def analyze_patterns(sequence: List[str]) -> Tuple[Dict, Dict, Dict, Dict, Dict, int, int, int, float, float]:
    try:
        bigram_transitions = defaultdict(lambda: defaultdict(int))
        trigram_transitions = defaultdict(lambda: defaultdict(int))
        fourgram_transitions = defaultdict(lambda: defaultdict(int))
        pattern_transitions = defaultdict(lambda: defaultdict(int))
        markov_transitions = defaultdict(lambda: defaultdict(int))
        streak_count = chop_count = double_count = pattern_changes = 0
        current_streak = last_pattern = None
        player_count = banker_count = 0
        filtered_sequence = [x for x in sequence if x in ['P', 'B']]
        for i in range(len(sequence) - 1):
            if sequence[i] == 'P':
                player_count += 1
            elif sequence[i] == 'B':
                banker_count += 1
            markov_transitions[sequence[i]][sequence[i+1]] += 1
            if i < len(sequence) - 2:
                bigram = tuple(sequence[i:i+2])
                trigram = tuple(sequence[i:i+3])
                next_outcome = sequence[i+2]
                bigram_transitions[bigram][next_outcome] += 1
                if i < len(sequence) - 3:
                    trigram_transitions[trigram][next_outcome] += 1
                    if i < len(sequence) - 4:
                        fourgram = tuple(sequence[i:i+4])
                        fourgram_transitions[fourgram][next_outcome] += 1
        for i in range(1, len(filtered_sequence)):
            if filtered_sequence[i] == filtered_sequence[i-1]:
                if current_streak == filtered_sequence[i]:
                    streak_count += 1
                else:
                    current_streak = filtered_sequence[i]
                    streak_count = 1
                if i > 1 and filtered_sequence[i-1] == filtered_sequence[i-2]:
                    double_count += 1
            else:
                current_streak = None
                streak_count = 0
                if i > 1 and filtered_sequence[i] != filtered_sequence[i-2]:
                    chop_count += 1
            if i < len(filtered_sequence) - 1:
                current_pattern = (
                    'streak' if streak_count >= 2 else
                    'chop' if chop_count >= 2 else
                    'double' if double_count >= 1 else 'other'
                )
                if last_pattern and last_pattern != current_pattern:
                    pattern_changes += 1
                last_pattern = current_pattern
                next_outcome = filtered_sequence[i+1]
                pattern_transitions[current_pattern][next_outcome] += 1
        volatility = pattern_changes / max(len(filtered_sequence) - 2, 1)
        total_outcomes = max(player_count + banker_count, 1)
        shoe_bias = player_count / total_outcomes if player_count > banker_count else -banker_count / total_outcomes
        return (bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions, markov_transitions,
                streak_count, chop_count, double_count, volatility, shoe_bias)
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        return ({}, {}, {}, {}, {}, 0, 0, 0, 0.0, 0.0)

def calculate_weights(streak_count: int, chop_count: int, double_count: int, shoe_bias: float) -> Dict[str, float]:
    try:
        total_bets = max(st.session_state.pattern_attempts.get('fourgram', 1), 1)
        success_ratios = {
            'bigram': st.session_state.pattern_success.get('bigram', 0) / total_bets
                      if st.session_state.pattern_attempts.get('bigram', 0) > 0 else 0.5,
            'trigram': st.session_state.pattern_success.get('trigram', 0) / total_bets
                       if st.session_state.pattern_attempts.get('trigram', 0) > 0 else 0.5,
            'fourgram': st.session_state.pattern_success.get('fourgram', 0) / total_bets
                        if st.session_state.pattern_attempts.get('fourgram', 0) > 0 else 0.5,
            'markov': st.session_state.pattern_success.get('markov', 0) / total_bets
                      if st.session_state.pattern_attempts.get('markov', 0) > 0 else 0.5,
            'streak': 0.6 if streak_count >= 2 else 0.3,
            'chop': 0.4 if chop_count >= 2 else 0.2,
            'double': 0.4 if double_count >= 1 else 0.2
        }
        if success_ratios['fourgram'] > 0.6:
            success_ratios['fourgram'] *= 1.2
        if success_ratios['markov'] > 0.6:
            success_ratios['markov'] *= 1.1
        weights = {k: np.exp(v) / (1 + np.exp(v)) for k, v in success_ratios.items()}
        if shoe_bias > 0.1:
            weights['bigram'] *= 1.1
            weights['trigram'] *= 1.1
            weights['fourgram'] *= 1.15
            weights['markov'] *= 1.1
        elif shoe_bias < -0.1:
            weights['bigram'] *= 0.9
            weights['trigram'] *= 0.9
            weights['fourgram'] *= 0.85
            weights['markov'] *= 0.9
        total_w = sum(weights.values())
        if total_w == 0:
            weights = {
                'bigram': 0.25,
                'trigram': 0.20,
                'fourgram': 0.20,
                'markov': 0.20,
                'streak': 0.10,
                'chop': 0.05,
                'double': 0.05
            }
            total_w = sum(weights.values())
        return {k: max(w / total_w, 0.05) for k, v in weights.items()}
    except Exception as e:
        logger.error(f"Weight calculation failed: {e}")
        return {
            'bigram': 0.25,
            'trigram': 0.20,
            'fourgram': 0.20,
            'markov': 0.20,
            'streak': 0.10,
            'chop': 0.05,
            'double': 0.05
        }

def predict_next() -> Tuple[Optional[str], float, Dict]:
    try:
        sequence = [x for x in st.session_state.sequence if x in ['P', 'B', 'T']]
        if len(sequence) < 4:
            return 'B', 45.86, {}
        recent_sequence = sequence[-WINDOW_SIZE:]
        (bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions, markov_transitions,
         streak_count, chop_count, double_count, volatility, shoe_bias) = analyze_patterns(recent_sequence)
        st.session_state.pattern_volatility = volatility
        prior_p, prior_b = 44.62 / 100, 45.86 / 100
        weights = calculate_weights(streak_count, chop_count, double_count, shoe_bias)
        prob_p = prob_b = total_weight = 0
        insights = {}
        if len(recent_sequence) >= 1:
            current_state = recent_sequence[-1]
            total = sum(markov_transitions[current_state].values())
            if total > 0:
                p_prob = markov_transitions[current_state]['P'] / total
                b_prob = markov_transitions[current_state]['B'] / total
                prob_p += weights['markov'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['markov'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['markov']
                insights['Markov'] = f"{weights['markov']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
        if len(recent_sequence) >= 2:
            bigram = tuple(recent_sequence[-2:])
            total = sum(bigram_transitions[bigram].values())
            if total > 0:
                p_prob = bigram_transitions[bigram]['P'] / total
                b_prob = bigram_transitions[bigram]['B'] / total
                prob_p += weights['bigram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['bigram']
                insights['Bigram'] = f"{weights['bigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
        if len(recent_sequence) >= 3:
            trigram = tuple(recent_sequence[-3:])
            total = sum(trigram_transitions[trigram].values())
            if total > 0:
                p_prob = trigram_transitions[trigram]['P'] / total
                b_prob = trigram_transitions[trigram]['B'] / total
                prob_p += weights['trigram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['trigram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['trigram']
                insights['Trigram'] = f"{weights['trigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
        if len(recent_sequence) >= 4:
            fourgram = tuple(recent_sequence[-4:])
            total = sum(fourgram_transitions[fourgram].values())
            if total > 0:
                p_prob = fourgram_transitions[fourgram]['P'] / total
                b_prob = fourgram_transitions[fourgram]['B'] / total
                prob_p += weights['fourgram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['fourgram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['fourgram']
                insights['Fourgram'] = f"{weights['fourgram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
        if streak_count >= 2:
            streak_prob = min(0.7, 0.5 + streak_count * 0.05) * (0.8 if streak_count > 4 else 1.0)
            current_streak = recent_sequence[-1]
            if current_streak == 'P':
                prob_p += weights['streak'] * streak_prob
                prob_b += weights['streak'] * (1 - streak_prob)
            else:
                prob_b += weights['streak'] * streak_prob
                prob_p += weights['streak'] * (1 - streak_prob)
            total_weight += weights['streak']
            insights['Streak'] = f"{weights['streak']*100:.0f}% ({streak_count} {current_streak})"
        if chop_count >= 2:
            next_pred = 'B' if recent_sequence[-1] == 'P' else 'P'
            if next_pred == 'P':
                prob_p += weights['chop'] * 0.6
                prob_b += weights['chop'] * 0.4
            else:
                prob_b += weights['chop'] * 0.6
                prob_p += weights['chop'] * 0.4
            total_weight += weights['chop']
            insights['Chop'] = f"{weights['chop']*100:.0f}% ({chop_count} alternations)"
        if double_count >= 1 and len(recent_sequence) >= 2 and recent_sequence[-1] == recent_sequence[-2]:
            double_prob = 0.6
            if recent_sequence[-1] == 'P':
                prob_p += weights['double'] * double_prob
                prob_b += weights['double'] * (1 - double_prob)
            else:
                prob_b += weights['double'] * double_prob
                prob_p += weights['double'] * (1 - double_prob)
            total_weight += weights['double']
            insights['Double'] = f"{weights['double']*100:.0f}% ({recent_sequence[-1]}{recent_sequence[-1]})"
        if total_weight > 0:
            prob_p = (prob_p / total_weight) * 100
            prob_b = (prob_b / total_weight) * 100
        else:
            prob_p, prob_b = 44.62, 45.86
        if shoe_bias > 0.1:
            prob_p *= 1.05
            prob_b *= 0.95
        elif shoe_bias < -0.1:
            prob_b *= 1.05
            prob_p *= 0.95
        if abs(prob_p - prob_b) < 2:
            prob_p += 0.5
            prob_b -= 0.5
        current_pattern = (
            'streak' if streak_count >= 2 else
            'chop' if chop_count >= 2 else
            'double' if double_count >= 1 else 'other'
        )
        total = sum(pattern_transitions[current_pattern].values())
        if total > 0:
            p_prob = pattern_transitions[current_pattern]['P'] / total
            b_prob = pattern_transitions[current_pattern]['B'] / total
            prob_p = 0.9 * prob_p + 0.1 * p_prob * 100
            prob_b = 0.9 * prob_b + 0.1 * b_prob * 100
            insights['Pattern Transition'] = f"10% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
        recent_accuracy = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / max(st.session_state.prediction_accuracy['total'], 1)
        threshold = 32.0 + (st.session_state.consecutive_losses * 0.5) - (recent_accuracy * 0.8)
        threshold = min(max(threshold, 32.0), 42.0)
        insights['Threshold'] = f"{threshold:.1f}%"
        if st.session_state.pattern_volatility > 0.5:
            threshold += 1.5
            insights['Volatility'] = f"High (Adjustment: +1.5% threshold)"
        if prob_p > prob_b and prob_p >= threshold:
            return 'P', prob_p, insights
        elif prob_b >= threshold:
            return 'B', prob_b, insights
        return None, max(prob_p, prob_b), insights
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None, 45.86, {}

# --- Betting Logic ---
def check_target_hit() -> bool:
    try:
        if st.session_state.target_mode == "Profit %":
            target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
            return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
        return unit_profit >= st.session_state.target_value
    except Exception as e:
        logger.error(f"Target check failed: {e}")
        return False

def update_t3_level():
    try:
        if len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            old_level = st.session_state.t3_level
            if wins == 3:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
            elif wins == 2 and losses == 1:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses == 2 and wins == 1:
                st.session_state.t3_level = st.session_state.t3_level + 1
            elif losses == 3:
                st.session_state.t3_level = st.session_state.t3_level + 2
            if old_level != st.session_state.t3_level:
                st.session_state.t3_level_changes += 1
            st.session_state.t3_peak_level = max(st.session_state.t3_peak_level, st.session_state.t3_level)
            st.session_state.t3_results = []
    except Exception as e:
        logger.error(f"T3 level update failed: {e}")

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], Optional[str]]:
    try:
        if st.session_state.consecutive_losses >= 3 and conf < 45.0:
            return None, f"No bet: Paused after {st.session_state.consecutive_losses} losses"
        if st.session_state.pattern_volatility > 0.6:
            return None, f"No bet: High pattern volatility"
        if st.session_state.shoe_completed:
            return None, f"No bet: Shoe completed"
        if pred is None or conf < 32.0:
            return None, f"No bet: Confidence too low"
        if st.session_state.strategy == 'Z1003.1':
            if st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
                return None, "No bet: Stopped after three losses (Z1003.1 rule)"
            bet_amount = enhanced_z1003_bet(st.session_state.z1003_loss_count, st.session_state.base_bet)
        elif st.session_state.strategy == 'Flatbet':
            bet_amount = st.session_state.base_bet
        elif st.session_state.strategy == 'T3':
            bet_amount = st.session_state.base_bet * st.session_state.t3_level
        else:  # Parlay16
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
            st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)
        if st.session_state.safety_net_enabled:
            safe_bankroll = st.session_state.initial_bankroll * (adjust_safety_net() / 100)
            if (bet_amount > st.session_state.bankroll or
                st.session_state.bankroll - bet_amount < safe_bankroll * 0.5 or
                bet_amount > st.session_state.bankroll * 0.10):
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
                return None, "No bet: Risk too high for current bankroll. Level/step reset to 1."
        return bet_amount, f"Auto Bet: ${bet_amount:.2f} on {pred}"
    except Exception as e:
        logger.error(f"Bet amount calculation failed: {e}")
        return None, "No bet: Calculation error"

def undo_last_result():
    try:
        if st.session_state.history:
            last_history = st.session_state.history.pop()
            previous_state = last_history["Previous_State"]
            st.session_state.update(previous_state)
            if st.session_state.sequence:
                st.session_state.sequence.pop()
            st.session_state.pending_bet = previous_state.get('pending_bet')
            st.session_state.advice = "Last result undone."
            st.session_state.insights = previous_state.get('insights', {})
            pred, conf, insights = predict_next()
            bet_amount, advice = calculate_bet_amount(pred, conf)
            st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
            st.session_state.advice = advice
            st.session_state.insights = insights
            if st.session_state.strategy == 'T3':
                update_t3_level()
    except Exception as e:
        logger.error(f"Undo last result failed: {e}")
        st.error("Error undoing last result. Please refresh.")

def place_result(result: str):
    try:
        if st.session_state.target_hit:
            reset_session()
            return
        # Check stop loss
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
            "ai_automation_enabled": st.session_state.ai_automation_enabled
        }
        if result != 'T':
            pred, conf, _ = predict_next()
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
                        st.session_state.wins += 1
                        st.session_state.prediction_accuracy[selection] += 1
                        st.session_state.consecutive_losses = 0
                        for pattern in ['bigram', 'trigram', 'fourgram', 'markov', 'streak', 'chop', 'double']:
                            if pattern in st.session_state.insights:
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
                        st.session_state.losses += 1
                        st.session_state.consecutive_losses += 1
                        _, conf, _ = predict_next()
                        st.session_state.loss_log.append({
                            'sequence': st.session_state.sequence[-10:],
                            'prediction': selection,
                            'result': result,
                            'confidence': f"{conf:.1f}",
                            'insights': st.session_state.insights.copy()
                        })
                        if len(st.session_state.loss_log) > LOSS_LOG_LIMIT:
                            st.session_state.loss_log = st.session_state.loss_log[-LOSS_LOG_LIMIT:]
                        for pattern in ['bigram', 'trigram', 'fourgram', 'markov', 'streak', 'chop', 'double']:
                            if pattern in st.session_state.insights:
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
            "Z1003_Bet_Factor": None,
            "Previous_State": previous_state,
            "Bet_Placed": bet_placed
        })
        if len(st.session_state.history) > HISTORY_LIMIT:
            st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]
        if check_target_hit():
            st.session_state.target_hit = True
            reset_session()
            return
        # Recheck stop loss after updating bankroll
        if st.session_state.safety_net_enabled:
            safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
            if st.session_state.bankroll <= safe_bankroll:
                st.session_state.advice = "Session reset: Stop loss reached."
                reset_session()
                return
        pred, conf, insights = predict_next()
        bet_amount, advice = calculate_bet_amount(pred, conf)
        st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
        st.session_state.advice = advice
        st.session_state.insights = insights
        if st.session_state.strategy == 'T3':
            update_t3_level()
        if len(st.session_state.sequence) >= SHOE_SIZE:
            st.session_state.shoe_completed = True
    except Exception as e:
        logger.error(f"Place result failed: {e}")
        st.error("Error processing result. Please refresh.")

# --- Simulation Logic ---
def simulate_shoe(num_hands: int = SHOE_SIZE) -> Dict:
    try:
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
            pred, conf, insights = predict_next()
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
            'losses': st.session_state.losses
        }
        try:
            with open(SIMULATION_LOG, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()}: Accuracy={accuracy:.1f}%, Correct={correct}/{total}, "
                        f"Fourgram={result['pattern_success'].get('fourgram', 0)}/{result['pattern_attempts'].get('fourgram', 0)}, "
                        f"Markov={result['pattern_success'].get('markov', 0)}/{result['pattern_attempts'].get('markov', 0)}, "
                        f"Final Bankroll=${result['final_bankroll']:.2f}, Wins={result['wins']}, Losses={result['losses']}\n")
        except PermissionError:
            st.warning("Unable to write to simulation log. Results not saved to file.")
        return result
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {'accuracy': 0, 'correct': 0, 'total': 0, 'pattern_success': {}, 'pattern_attempts': {}, 'sequence': [], 'final_bankroll': 0, 'wins': 0, 'losses': 0}

# --- UI Components ---
def render_setup_form():
    try:
        with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
            with st.form("setup_form"):
                col1, col2 = st.columns(2)
                with col1:
                    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.initial_bankroll or 100.0, step=10.0)
                    base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=st.session_state.initial_base_bet or 0.10, step=0.10)
                with col2:
                    betting_strategy = st.selectbox(
                        "Betting Strategy", STRATEGIES,
                        index=STRATEGIES.index(st.session_state.strategy),
                        help="T3: Adjusts bet size based on wins/losses. Flatbet: Fixed bet size. Parlay16: 16-step progression. Z1003.1: Resets after first win, stops after three losses."
                    )
                    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=["Profit %", "Units"].index(st.session_state.target_mode))
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
                            'history': [],
                            'wins': 0,
                            'losses': 0,
                            'target_mode': target_mode,
                            'target_value': target_value,
                            'initial_bankroll': bankroll,
                            'target_hit': False,
                            'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
                            'consecutive_losses': 0,
                            'loss_log': [],
                            'last_was_tie': False,
                            'insights': {},
                            'pattern_volatility': 0.0,
                            'pattern_success': defaultdict(int),
                            'pattern_attempts': defaultdict(int),
                            'safety_net_percentage': safety_net_percentage,
                            'safety_net_enabled': safety_net_enabled,
                            'ai_automation_enabled': True,
                            'shoe_completed': False
                        })
                        st.session_state.pattern_success['fourgram'] = 0
                        st.session_state.pattern_attempts['fourgram'] = 0
                        st.session_state.pattern_success['markov'] = 0
                        st.session_state.pattern_attempts['markov'] = 0
                        recommended = recommend_strategy(st.session_state.sequence)
                        st.success(f"Session started with {betting_strategy} strategy! AI Automation: Enabled. Recommended: {recommended}")
                        st.rerun()
    except Exception as e:
        logger.error(f"Setup form rendering failed: {e}")
        st.error("Error rendering setup form. Please refresh.")

def render_result_input():
    try:
        with st.expander("Enter Shoe Results", expanded=True):
            if st.session_state.shoe_completed:
                st.success(f"Shoe of {SHOE_SIZE} hands completed!")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Player", key="player_btn", help="Record a Player win", disabled=st.session_state.shoe_completed):
                    place_result('P')
                    st.rerun()
            with col2:
                if st.button("Banker", key="banker_btn", help="Record a Banker win", disabled=st.session_state.shoe_completed):
                    place_result('B')
                    st.rerun()
            with col3:
                if st.button("Tie", key="tie_btn", help="Record a Tie", disabled=st.session_state.shoe_completed):
                    place_result('T')
                    st.rerun()
            with col4:
                if st.button("Undo", key="undo_btn", help="Undo the last result", disabled=not st.session_state.history or st.session_state.shoe_completed):
                    undo_last_result()
                    st.rerun()
            if st.button("Run Automated Shoe Simulation", key="auto_shoe_btn", disabled=st.session_state.shoe_completed):
                result = simulate_shoe()
                st.session_state.shoe_completed = True
                st.rerun()
            if st.session_state.shoe_completed and st.button("Reset and Start New Shoe", key="new_shoe_btn"):
                reset_session()
                st.session_state.shoe_completed = False
                st.rerun()
    except Exception as e:
        logger.error(f"Result input rendering failed: {e}")
        st.error("Error rendering result input. Please refresh.")

def render_bead_plate():
    try:
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
    except Exception as e:
        logger.error(f"Bead plate rendering failed: {e}")
        st.error("Error rendering bead plate. Please refresh.")

def render_prediction():
    try:
        with st.expander("Latest Prediction", expanded=True):
            if st.session_state.pending_bet:
                amount, pred = st.session_state.pending_bet
                color = '#3182ce' if pred == 'P' else '#e53e3e'
                st.markdown(f"<div style='background-color: #edf2f7; padding: 15px; border-radius: 8px;'><h4 style='color:{color}; margin:0;'>AI Auto Bet: {pred} | Amount: ${amount:.2f}</h4></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color: #edf2f7; padding: 15px; border-radius: 8px;'><h4 style='color:#4a5568; margin:0;'>AI Auto Bet: None</h4></div>", unsafe_allow_html=True)
            st.info(st.session_state.advice)
    except Exception as e:
        logger.error(f"Prediction rendering failed: {e}")
        st.error("Error rendering prediction. Please refresh.")

def render_insights():
    try:
        with st.expander("Prediction Insights"):
            if st.session_state.insights:
                for factor, contribution in st.session_state.insights.items():
                    st.markdown(f"**{factor}**: {contribution}")
            if st.session_state.pattern_volatility > 0.5:
                st.warning(f"High Pattern Volatility: {st.session_state.pattern_volatility:.2f} (Consider pausing bets)")
    except Exception as e:
        logger.error(f"Insights rendering failed: {e}")
        st.error("Error rendering insights. Please refresh.")

def render_status():
    try:
        with st.expander("Session Status", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
                st.markdown(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
                st.markdown(f"**Safety Net**: {'Enabled' if st.session_state.safety_net_enabled else 'Disabled'}"
                            f"{' | ' + str(st.session_state.safety_net_percentage) + '%' if st.session_state.safety_net_enabled else ''}")
            with col2:
                strategy_status = f"**Strategy**: {st.session_state.strategy}"
                if st.session_state.strategy == 'T3':
                    strategy_status += f"<br>Level: {st.session_state.t3_level} | Peak: {st.session_state.t3_peak_level}<br>Changes: {st.session_state.t3_level_changes}"
                elif st.session_state.strategy == 'Parlay16':
                    strategy_status += f"<br>Steps: {st.session_state.parlay_step}/16 | Peak: {st.session_state.parlay_peak_step}<br>Changes: {st.session_state.parlay_step_changes} | Wins: {st.session_state.parlay_wins}"
                elif st.session_state.strategy == 'Z1003.1':
                    strategy_status += f"<br>Loss Count: {st.session_state.z1003_loss_count}<br>Changes: {st.session_state.z1003_level_changes} | Continue: {st.session_state.z1003_continue}"
                st.markdown(strategy_status, unsafe_allow_html=True)
            st.markdown(f"**AI Automation**: {'Enabled' if st.session_state.ai_automation_enabled else 'Disabled'}")
            st.markdown(f"**Wins**: {st.session_state.wins} | **Losses**: {st.session_state.losses}")
            st.markdown(f"**Online Users**: {track_user_session()}")
            if st.session_state.initial_base_bet > 0 and st.session_state.initial_bankroll > 0:
                profit = st.session_state.bankroll - st.session_state.initial_bankroll
                units_profit = profit / st.session_state.initial_base_bet
                st.markdown(f"**Profit**: {units_profit:.2f} units (${profit:.2f})")
            else:
                st.markdown("**Profit**: 0.00 units ($0.00)")
    except Exception as e:
        logger.error(f"Status rendering failed: {e}")
        st.error("Error rendering status. Please refresh.")

def render_accuracy():
    try:
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
    except Exception as e:
        logger.error(f"Accuracy rendering failed: {e}")
        st.error("Error rendering accuracy. Please refresh.")

def render_loss_log():
    try:
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
    except Exception as e:
        logger.error(f"Loss log rendering failed: {e}")
        st.error("Error rendering loss log. Please refresh.")

def render_history():
    try:
        with st.expander("Bet History"):
            if st.session_state.history:
                n = st.slider("Show last N bets", 5, 50, 10)
                st.dataframe([
                    {
                        "Bet": h["Bet"] if h["Bet"] else "-",
                        "Result": h["Result"],
                        "Amount": f"${h['Amount']:.2f}" if h["Bet_Placed"] else "-",
                        "Outcome": "Win" if h["Win"] else "Loss" if h["Bet_Placed"] else "-",
                        "T3_Level": h["T3_Level"] if st.session_state.strategy == 'T3' else "-",
                        "Parlay_Step": h["Parlay_Step"] if st.session_state.strategy == 'Parlay16' else "-",
                        "Z1003_Loss_Count": h["Z1003_Loss_Count"] if st.session_state.strategy == 'Z1003.1' else "-",
                    }
                    for h in st.session_state.history[-n:]
                ], use_container_width=True)
    except Exception as e:
        logger.error(f"History rendering failed: {e}")
        st.error("Error rendering history. Please refresh.")

def render_export():
    try:
        with st.expander("Export Session"):
            if st.button("Download Session Data"):
                csv_data = "Bet,Result,Amount,Win,T3_Level,Parlay_Step,Z1003_Loss_Count\n"
                for h in st.session_state.history:
                    csv_data += f"{h['Bet'] or '-'},{h['Result']},${h['Amount']:.2f},{h['Win']},{h['T3_Level']},{h['Parlay_Step']},{h['Z1003_Loss_Count']}\n"
                st.download_button("Download CSV", csv_data, "session_data.csv", "text/csv")
    except Exception as e:
        logger.error(f"Export rendering failed: {e}")
        st.error("Error rendering export. Please refresh.")

def render_simulation():
    try:
        with st.expander("Run Additional Simulation"):
            num_hands = st.number_input("Number of Hands to Simulate", min_value=10, max_value=200, value=SHOE_SIZE, step=10)
            if st.button("Run Simulation"):
                result = simulate_shoe(num_hands)
                st.write(f"**Simulation Results**")
                st.write(f"Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']} correct)")
                st.write(f"Final Bankroll: ${result['final_bankroll']:.2f}")
                st.write(f"Wins: {result['wins']} | Losses: {result['losses']}")
                st.write("**Pattern Performance**:")
                for pattern in result['pattern_success']:
                    success = result['pattern_success'][pattern]
                    attempts = result['pattern_attempts'][pattern]
                    st.write(f"{pattern}: {success}/{attempts} ({success/attempts*100:.1f}%)" if attempts > 0 else f"{pattern}: 0/0 (0%)")
                st.write("Results logged to simulation_log.txt")
    except Exception as e:
        logger.error(f"Simulation rendering failed: {e}")
        st.error("Error rendering simulation. Please refresh.")

# --- Main Application ---
def main():
    try:
        st.set_page_config(layout="wide", page_title="MANG Baccarat")
        apply_custom_css()
        initialize_session_state()
        st.title("MANG Baccarat")
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
            render_profit_dashboard()
    except Exception as e:
        logger.error(f"Main application failed: {e}")
        st.error("Application failed to load. Please check logs or refresh.")

if __name__ == "__main__":
    main()
