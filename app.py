# Version: 2025-05-14-fix-v12
import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import time
import numpy as np
from typing import Tuple, Dict, Optional, List
import tempfile
import logging
import traceback

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
APP_VERSION = "2025-05-14-fix-v12"

# --- Logging Setup ---
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# --- Session Tracking ---
def track_user_session() -> int:
    """Track active user sessions with fallback for file errors."""
    logging.debug("Entering track_user_session")
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
                        logging.warning(f"Invalid session file line: {line}")
                        continue
    except (PermissionError, OSError) as e:
        logging.error(f"Session file read error: {str(e)}")
        st.warning("Session tracking unavailable. Using fallback.")
        return 1

    sessions[st.session_state.session_id] = current_time

    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            for session_id, last_seen in sessions.items():
                f.write(f"{session_id},{last_seen.isoformat()}\n")
    except (PermissionError, OSError) as e:
        logging.error(f"Session file write error: {str(e)}")
        st.warning("Session tracking may be inaccurate.")
        return len(sessions)

    logging.debug(f"track_user_session: {len(sessions)} active sessions")
    return len(sessions)

# --- Session State Management ---
def initialize_session_state():
    """Initialize session state with default values."""
    logging.debug("Entering initialize_session_state")
    defaults = {
        'bankroll': 0.0,
        'base_bet': 0.0,
        'initial_base_bet': 0.0,
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
        'last_win_confidence': 0.0,
        'recent_pattern_accuracy': defaultdict(float),
        'consecutive_wins': 0,
    }
    defaults['pattern_success']['fourgram'] = 0
    defaults['pattern_attempts']['fourgram'] = 0
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.strategy not in STRATEGIES:
        st.session_state.strategy = 'T3'
    logging.debug("initialize_session_state completed")

def reset_session():
    """Reset session state to initial values."""
    logging.debug("Entering reset_session")
    for key in list(st.session_state.keys()):
        if key != 'session_id':
            del st.session_state[key]
    initialize_session_state()
    st.session_state.update({
        'bankroll': st.session_state.initial_bankroll,
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
        'last_win_confidence': 0.0,
        'consecutive_wins': 0,
    })
    logging.debug("reset_session completed")

# --- Prediction Logic ---
def analyze_patterns(sequence: List[str]) -> Tuple[Dict, Dict, Dict, Dict, int, int, int, float, float, Dict]:
    """Analyze sequence patterns with streak and chop metrics."""
    logging.debug("Entering analyze_patterns")
    try:
        bigram_transitions = defaultdict(lambda: defaultdict(int))
        trigram_transitions = defaultdict(lambda: defaultdict(int))
        fourgram_transitions = defaultdict(lambda: defaultdict(int))
        pattern_transitions = defaultdict(lambda: defaultdict(int))
        streak_count = chop_count = double_count = pattern_changes = 0
        current_streak = last_pattern = None
        player_count = banker_count = 0
        streak_lengths = []
        chop_lengths = []

        filtered_sequence = [x for x in sequence if x in ['P', 'B']]
        for i in range(len(sequence) - 1):
            if sequence[i] == 'P':
                player_count += 1
            elif sequence[i] == 'B':
                banker_count += 1

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

        current_streak_length = 0
        current_chop_length = 0
        for i in range(1, len(filtered_sequence)):
            if filtered_sequence[i] == filtered_sequence[i-1]:
                if current_streak == filtered_sequence[i]:
                    current_streak_length += 1
                else:
                    if current_streak_length > 1:
                        streak_lengths.append(current_streak_length)
                    current_streak = filtered_sequence[i]
                    current_streak_length = 1
                if i > 1 and filtered_sequence[i-1] == filtered_sequence[i-2]:
                    double_count += 1
                if current_chop_length > 1:
                    chop_lengths.append(current_chop_length)
                    current_chop_length = 0
            else:
                current_streak = None
                if current_streak_length > 1:
                    streak_lengths.append(current_streak_length)
                current_streak_length = 0
                if i > 1 and filtered_sequence[i] != filtered_sequence[i-2]:
                    current_chop_length += 1
                    chop_count += 1
                else:
                    if current_chop_length > 1:
                        chop_lengths.append(current_chop_length)
                    current_chop_length = 0

            if i < len(filtered_sequence) - 1:
                current_pattern = (
                    'streak' if current_streak_length >= 2 else
                    'chop' if chop_count >= 2 else
                    'double' if double_count >= 1 else 'other'
                )
                if last_pattern and last_pattern != current_pattern:
                    pattern_changes += 1
                last_pattern = current_pattern
                next_outcome = filtered_sequence[i+1]
                pattern_transitions[current_pattern][next_outcome] += 1

        if current_streak_length > 1:
            streak_lengths.append(current_streak_length)
        if current_chop_length > 1:
            chop_lengths.append(current_chop_length)

        volatility = pattern_changes / max(len(filtered_sequence) - 2, 1)
        total_outcomes = max(player_count + banker_count, 1)
        shoe_bias = player_count / total_outcomes if player_count > banker_count else -banker_count / total_outcomes

        extra_metrics = {
            'avg_streak_length': sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0,
            'avg_chop_length': sum(chop_lengths) / len(chop_lengths) if chop_lengths else 0,
            'streak_frequency': len(streak_lengths) / max(len(filtered_sequence), 1),
            'chop_frequency': len(chop_lengths) / max(len(filtered_sequence), 1)
        }

        logging.debug("analyze_patterns completed")
        return (bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions,
                streak_count, chop_count, double_count, volatility, shoe_bias, extra_metrics)
    except Exception as e:
        logging.error(f"analyze_patterns error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error analyzing patterns. Try resetting the session.")
        return ({}, {}, {}, {}, 0, 0, 0, 0.0, 0.0, {})

def calculate_weights(streak_count: int, chop_count: int, double_count: int, shoe_bias: float) -> Dict[str, float]:
    """Calculate adaptive weights with error handling."""
    logging.debug("Entering calculate_weights")
    try:
        total_bets = max(st.session_state.pattern_attempts.get('fourgram', 1), 1)
        success_ratios = {
            'bigram': st.session_state.pattern_success.get('bigram', 0) / total_bets,
            'trigram': st.session_state.pattern_success.get('trigram', 0) / total_bets,
            'fourgram': st.session_state.pattern_success.get('fourgram', 0) / total_bets,
            'streak': 0.6 if streak_count >= 2 else 0.3,
            'chop': 0.4 if chop_count >= 2 else 0.2,
            'double': 0.4 if double_count >= 1 else 0.2
        }

        recent_bets = st.session_state.history[-10:]
        recent_success = defaultdict(int)
        recent_attempts = defaultdict(int)
        for h in recent_bets:
            if h['Bet_Placed'] and h['Bet'] in ['P', 'B']:
                for pattern in h.get('Previous_State', {}).get('insights', {}):
                    recent_attempts[pattern] += 1
                    if h['Win']:
                        recent_success[pattern] += 1
        for pattern in success_ratios:
            if recent_attempts[pattern] > 0:
                recent_ratio = recent_success[pattern] / recent_attempts[pattern]
                if recent_ratio > 0.7:
                    success_ratios[pattern] *= 1.5
                elif recent_ratio < 0.3:
                    success_ratios[pattern] *= 0.6

        if success_ratios['fourgram'] > 0.6:
            success_ratios['fourgram'] *= 1.3

        weights = {k: np.exp(v) / (1 + np.exp(v)) for k, v in success_ratios.items()}
        if shoe_bias > 0.1:
            weights['bigram'] *= 1.1
            weights['trigram'] *= 1.1
            weights['fourgram'] *= 1.15
        elif shoe_bias < -0.1:
            weights['bigram'] *= 0.9
            weights['trigram'] *= 0.9
            weights['fourgram'] *= 0.85

        total_weight = sum(weights.values())
        if total_weight == 0:
            weights = {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}
            total_weight = sum(weights.values())

        normalized_weights = {k: max(v / total_weight, 0.05) for k, v in weights.items()}

        dominant_pattern = max(normalized_weights, key=normalized_weights.get)
        st.session_state.insights['Dominant Pattern'] = {
            'pattern': dominant_pattern,
            'weight': normalized_weights[dominant_pattern] * 100
        }

        logging.debug("calculate_weights completed")
        return normalized_weights
    except NameError as e:
        logging.error(f"NameError in calculate_weights: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Variable error in weight calculation: {str(e)}. Try resetting the session.")
        return {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}
    except Exception as e:
        logging.error(f"calculate_weights error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error calculating weights. Try resetting the session.")
        return {'bigram': 0.30, 'trigram': 0.25, 'fourgram': 0.25, 'streak': 0.15, 'chop': 0.05, 'double': 0.05}

def predict_next() -> Tuple[Optional[str], float, Dict]:
    """Predict the next outcome with error handling."""
    logging.debug("Entering predict_next")
    try:
        sequence = [x for x in st.session_state.sequence if x in ['P', 'B', 'T']]
        shadow_sequence = [x for x in sequence if x in ['P', 'B']]
        if len(shadow_sequence) < 4:
            return 'B', 45.86, {'Initial': 'Default to Banker (insufficient data)'}

        recent_sequence = shadow_sequence[-WINDOW_SIZE:]
        (bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions,
         streak_count, chop_count, double_count, volatility, shoe_bias, extra_metrics) = analyze_patterns(recent_sequence)
        st.session_state.pattern_volatility = volatility

        prior_p, prior_b = 44.62 / 100, 45.86 / 100
        weights = calculate_weights(streak_count, chop_count, double_count, shoe_bias)
        prob_p = prob_b = total_weight = 0
        insights = {}
        pattern_reliability = {}
        recent_performance = {}

        recent_bets = st.session_state.history[-10:]
        for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
            success = sum(1 for h in recent_bets if h['Bet_Placed'] and h['Win'] and pattern in h.get('Previous_State', {}).get('insights', {}))
            attempts = sum(1 for h in recent_bets if h['Bet_Placed'] and pattern in h.get('Previous_State', {}).get('insights', {}))
            recent_performance[pattern] = success / max(attempts, 1) if attempts > 0 else 0.0

        if len(recent_sequence) >= 2:
            bigram = tuple(recent_sequence[-2:])
            total = sum(bigram_transitions[bigram].values())
            if total > 0:
                p_prob = bigram_transitions[bigram]['P'] / total
                b_prob = bigram_transitions[bigram]['B'] / total
                prob_p += weights['bigram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['bigram']
                reliability = min(total / 5, 1.0)
                pattern_reliability['Bigram'] = reliability
                insights['Bigram'] = {
                    'weight': weights['bigram'] * 100,
                    'p_prob': p_prob * 100,
                    'b_prob': b_prob * 100,
                    'reliability': reliability * 100,
                    'recent_performance': recent_performance['bigram'] * 100
                }

        if len(recent_sequence) >= 3:
            trigram = tuple(recent_sequence[-3:])
            total = sum(trigram_transitions[trigram].values())
            if total > 0:
                p_prob = trigram_transitions[trigram]['P'] / total
                b_prob = trigram_transitions[trigram]['B'] / total
                prob_p += weights['trigram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['trigram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['trigram']
                reliability = min(total / 3, 1.0)
                pattern_reliability['Trigram'] = reliability
                insights['Trigram'] = {
                    'weight': weights['trigram'] * 100,
                    'p_prob': p_prob * 100,
                    'b_prob': b_prob * 100,
                    'reliability': reliability * 100,
                    'recent_performance': recent_performance['trigram'] * 100
                }

        if len(recent_sequence) >= 4:
            fourgram = tuple(recent_sequence[-4:])
            total = sum(fourgram_transitions[fourgram].values())
            if total > 0:
                p_prob = fourgram_transitions[fourgram]['P'] / total
                b_prob = fourgram_transitions[fourgram]['B'] / total
                prob_p += weights['fourgram'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['fourgram'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['fourgram']
                reliability = min(total / 2, 1.0)
                pattern_reliability['Fourgram'] = reliability
                insights['Fourgram'] = {
                    'weight': weights['fourgram'] * 100,
                    'p_prob': p_prob * 100,
                    'b_prob': b_prob * 100,
                    'reliability': reliability * 100,
                    'recent_performance': recent_performance['fourgram'] * 100
                }

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
            reliability = min(streak_count / 5, 1.0)
            pattern_reliability['Streak'] = reliability
            insights['Streak'] = {
                'weight': weights['streak'] * 100,
                'streak_type': current_streak,
                'streak_count': streak_count,
                'reliability': reliability * 100,
                'recent_performance': recent_performance['streak'] * 100
            }

        if chop_count >= 2:
            next_pred = 'B' if recent_sequence[-1] == 'P' else 'P'
            if next_pred == 'P':
                prob_p += weights['chop'] * 0.6
                prob_b += weights['chop'] * 0.4
            else:
                prob_b += weights['chop'] * 0.6
                prob_p += weights['chop'] * 0.4
            total_weight += weights['chop']
            reliability = min(chop_count / 5, 1.0)
            pattern_reliability['Chop'] = reliability
            insights['Chop'] = {
                'weight': weights['chop'] * 100,
                'chop_count': chop_count,
                'next_pred': next_pred,
                'reliability': reliability * 100,
                'recent_performance': recent_performance['chop'] * 100
            }

        if double_count >= 1 and len(recent_sequence) >= 2 and recent_sequence[-1] == recent_sequence[-2]:
            double_prob = 0.6
            if recent_sequence[-1] == 'P':
                prob_p += weights['double'] * double_prob
                prob_b += weights['double'] * (1 - double_prob)
            else:
                prob_b += weights['double'] * double_prob
                prob_p += weights['double'] * (1 - double_prob)
            total_weight += weights['double']
            reliability = min(double_count / 3, 1.0)
            pattern_reliability['Double'] = reliability
            insights['Double'] = {
                'weight': weights['double'] * 100,
                'double_type': recent_sequence[-1],
                'reliability': reliability * 100,
                'recent_performance': recent_performance['double'] * 100
            }

        if total_weight > 0:
            prob_p = (prob_p / total_weight) * 100
            prob_b = (prob_b / total_weight) * 100
        else:
            prob_p, prob_b = 44.62, 45.86

        if shoe_bias > 0.1:
            prob_p *= 1.05
            prob_b *= 0.95
            insights['Shoe Bias'] = {'bias': 'Player', 'adjustment': '+5% P, -5% B'}
        elif shoe_bias < -0.1:
            prob_b *= 1.05
            prob_p *= 0.95
            insights['Shoe Bias'] = {'bias': 'Banker', 'adjustment': '+5% B, -5% P'}

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
            reliability = min(total / 5, 1.0)
            insights['Pattern Transition'] = {
                'weight': 10,
                'p_prob': p_prob * 100,
                'b_prob': b_prob * 100,
                'current_pattern': current_pattern,
                'reliability': reliability * 100,
                'recent_performance': 0.0
            }

        recent_accuracy = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / max(st.session_state.prediction_accuracy['total'], 1)
        threshold = 32.0 + (st.session_state.consecutive_losses * 2.0) - (recent_accuracy * 0.8)
        threshold = min(max(threshold, 32.0), 48.0)
        if recent_performance.get('fourgram', 0) > 0.7:
            threshold -= 2.0
        elif recent_performance.get('fourgram', 0) < 0.3:
            threshold += 2.0
        insights['Threshold'] = {'value': threshold, 'adjusted': f'{threshold:.1f}%'}

        if st.session_state.pattern_volatility > 0.5:
            threshold += 1.5
            insights['Volatility'] = {
                'level': 'High',
                'value': st.session_state.pattern_volatility,
                'adjustment': '+1.5% threshold'
            }

        if prob_p > prob_b and prob_p >= threshold:
            prediction = 'P'
            confidence = prob_p
        elif prob_b >= threshold:
            prediction = 'B'
            confidence = prob_b
        else:
            prediction = None
            confidence = max(prob_p, prob_b)
            insights['No Bet'] = {'reason': f'Confidence below threshold ({confidence:.1f}% < {threshold:.1f}%)'}

        dominant_pattern = max(insights, key=lambda k: insights[k].get('weight', 0) if k not in ['Threshold', 'Volatility', 'Shoe Bias', 'No Bet'] else 0, default=None)
        if dominant_pattern and prediction:
            reliability = pattern_reliability.get(dominant_pattern, 0)
            recommendation = f"Favor {prediction} due to strong {dominant_pattern.lower()} pattern (Reliability: {reliability*100:.1f}%)"
            insights['Recommendation'] = {'text': recommendation}

        logging.debug("predict_next completed")
        return prediction, confidence, insights
    except Exception as e:
        logging.error(f"predict_next error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error predicting next outcome. Try resetting the session.")
        return None, 0.0, {}

# --- Betting Logic ---
def check_target_hit() -> bool:
    """Check if the profit target has been reached."""
    logging.debug("Entering check_target_hit")
    try:
        if st.session_state.target_mode == "Profit %":
            target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
            return st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
        return unit_profit >= st.session_state.target_value
    except Exception as e:
        logging.error(f"check_target_hit error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error checking target. Try resetting the session.")
        return False

def update_t3_level():
    """Update T3 betting level based on recent results."""
    logging.debug("Entering update_t3_level")
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
        logging.debug("update_t3_level completed")
    except Exception as e:
        logging.error(f"update_t3_level error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error updating T3 level. Try resetting the session.")

def calculate_bet_amount(pred: str, conf: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculate the next bet amount, resetting level/step if risk is too high."""
    logging.debug("Entering calculate_bet_amount")
    try:
        if st.session_state.consecutive_losses >= 3 and conf < 45.0:
            return None, f"No bet: Paused after {st.session_state.consecutive_losses} losses"
        if st.session_state.pattern_volatility > 0.6:
            return None, f"No bet: High pattern volatility"
        if pred is None or conf < 32.0:
            return None, f"No bet: Confidence too low"

        if st.session_state.strategy == 'Z1003.1':
            if st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
                return None, "No bet: Stopped after three losses (Z1003.1 rule)"
            bet_amount = st.session_state.base_bet + (st.session_state.z1003_loss_count * 100)
        elif st.session_state.strategy == 'Flatbet':
            bet_amount = st.session_state.base_bet
        elif st.session_state.strategy == 'T3':
            bet_amount = st.session_state.base_bet * st.session_state.t3_level
            logging.debug(f"T3 bet: base_bet={st.session_state.base_bet}, t3_level={st.session_state.t3_level}, bet_amount={bet_amount}")
        else:  # Parlay16
            key = 'base' if st.session_state.parlay_using_base else 'parlay'
            bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
            st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)

        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
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

        logging.debug("calculate_bet_amount completed")
        return bet_amount, f"Next Bet: ${bet_amount:.2f} on {pred}"
    except Exception as e:
        logging.error(f"calculate_bet_amount error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error calculating bet amount. Try resetting the session.")
        return None, "No bet: Calculation error"

def place_result(result: str):
    """Process a game result with error handling."""
    logging.debug("Entering place_result")
    try:
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
            "consecutive_wins": st.session_state.consecutive_wins,
            "last_win_confidence": st.session_state.last_win_confidence,
            "insights": st.session_state.insights.copy(),
        }

        if st.session_state.pending_bet and result != 'T':
            bet_amount, selection = st.session_state.pending_bet
            win = result == selection
            bet_placed = True
            if win:
                st.session_state.bankroll += bet_amount * (0.95 if selection == 'B' else 1.0)
                st.session_state.wins += 1
                st.session_state.consecutive_wins += 1
                st.session_state.consecutive_losses = 0
                st.session_state.last_win_confidence = predict_next()[1]
                logging.debug(f"Win recorded: Total wins={st.session_state.wins}, Consecutive wins={st.session_state.consecutive_wins}")
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
                st.session_state.prediction_accuracy[selection] += 1
                for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
                    if pattern in st.session_state.insights:
                        st.session_state.pattern_success[pattern] += 1
                        st.session_state.pattern_attempts[pattern] += 1
            else:
                st.session_state.bankroll -= bet_amount
                st.session_state.losses += 1
                st.session_state.consecutive_wins = 0
                st.session_state.consecutive_losses += 1
                logging.debug(f"Loss recorded: Total losses={st.session_state.losses}, Consecutive losses={st.session_state.consecutive_losses}")
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
                for pattern in ['bigram', 'trigram', 'fourgram', 'streak', 'chop', 'double']:
                    if pattern in st.session_state.insights:
                        st.session_state.pattern_attempts[pattern] += 1
                if st.session_state.strategy == 'Parlay16':
                    old_step = st.session_state.parlay_step
                    st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                    st.session_state.parlay_wins = 0
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
            "Previous_State": previous_state,
            "Bet_Placed": bet_placed,
            "Consecutive_Wins": st.session_state.consecutive_wins,
        })
        if len(st.session_state.history) > HISTORY_LIMIT:
            st.session_state.history = st.session_state.history[-HISTORY_LIMIT:]

        if check_target_hit():
            st.session_state.target_hit = True
            return

        pred, conf, insights = predict_next()
        if st.session_state.strategy == 'Z1003.1' and st.session_state.z1003_loss_count >= 3 and not st.session_state.z1003_continue:
            bet_amount, advice = None, "No bet: Stopped after three losses (Z1003.1 rule)"
        else:
            bet_amount, advice = calculate_bet_amount(pred, conf)
        st.session_state.pending_bet = (bet_amount, pred) if bet_amount else None
        st.session_state.advice = advice
        st.session_state.insights = insights

        if st.session_state.strategy == 'T3':
            update_t3_level()

        if st.session_state.wins < 0 or st.session_state.losses < 0:
            logging.error(f"Invalid win/loss counts: wins={st.session_state.wins}, losses={st.session_state.losses}")
            st.session_state.wins = max(0, st.session_state.wins)
            st.session_state.losses = max(0, st.session_state.losses)

        logging.debug("place_result completed")
    except Exception as e:
        logging.error(f"place_result error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error processing result. Try resetting the session.")

# --- Simulation Logic ---
def simulate_shoe(num_hands: int = 80) -> Dict:
    """Simulate a Baccarat shoe with error handling."""
    logging.debug("Entering simulate_shoe")
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
            'sequence': sequence
        }

        try:
            with open(SIMULATION_LOG, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()}: Accuracy={accuracy:.1f}%, Correct={correct}/{total}, "
                        f"Fourgram={result['pattern_success'].get('fourgram', 0)}/{result['pattern_attempts'].get('fourgram', 0)}\n")
        except (PermissionError, OSError) as e:
            logging.error(f"Simulation log write error: {str(e)}")
            st.warning("Unable to write to simulation log. Results displayed only.")

        logging.debug("simulate_shoe completed")
        return result
    except Exception as e:
        logging.error(f"simulate_shoe error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error running simulation. Try resetting the session.")
        return {'accuracy': 0, 'correct': 0, 'total': 0, 'pattern_success': {}, 'pattern_attempts': {}, 'sequence': []}

# --- UI Components ---
def render_setup_form():
    """Render the setup form for session configuration."""
    logging.debug("Entering render_setup_form")
    try:
        st.subheader("Setup")
        with st.form("setup_form"):
            bankroll = st.number_input("Enter Bankroll ($)", min_value=0.0, value=st.session_state.bankroll or 10.0, step=0.01, format="%.2f")
            base_bet = st.number_input("Enter Base Bet ($)", min_value=0.01, value=st.session_state.base_bet or 0.20, step=0.01, format="%.2f")
            betting_strategy = st.selectbox(
                "Choose Betting Strategy", STRATEGIES,
                index=STRATEGIES.index(st.session_state.strategy),
                help="T3: Adjusts bet size based on wins/losses. Flatbet: Fixed bet size. Parlay16: Progressive sequence. Z1003.1: Resets after first win, stops after three losses."
            )
            target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0, horizontal=True)
            target_value = st.number_input("Target Value", min_value=1.0, value=float(st.session_state.target_value), step=1.0)
            safety_net_enabled = st.toggle("Enable Safety Net", value=st.session_state.safety_net_enabled, help="Toggle to enable or disable the safety net protection.")
            safety_net_percentage = st.number_input(
                "Safety Net Percentage (%)",
                min_value=0.0, max_value=50.0, value=st.session_state.safety_net_percentage, step=5.0,
                disabled=not safety_net_enabled,
                help="Percentage of initial bankroll to keep as a safety net after each bet."
            )
            start_clicked = st.form_submit_button("Start Session")

            if start_clicked:
                if bankroll <= 0:
                    st.error("Bankroll must be positive.")
                elif base_bet < 0.01:
                    st.error("Base bet must be at least $0.01.")
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
                        'last_win_confidence': 0.0,
                        'recent_pattern_accuracy': defaultdict(float),
                        'consecutive_wins': 0,
                    })
                    st.session_state.pattern_success['fourgram'] = 0
                    st.session_state.pattern_attempts['fourgram'] = 0
                    st.success(f"Session started with {betting_strategy} strategy!")
        logging.debug("render_setup_form completed")
    except Exception as e:
        logging.error(f"render_setup_form error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering setup form. Try resetting the session.")

def render_result_input():
    """Render the result input buttons."""
    logging.debug("Entering render_result_input")
    try:
        st.subheader("Enter Result")
        st.markdown("""
        <style>
        div.stButton > button {
            width: 90px; height: 35px; font-size: 14px; font-weight: bold; border-radius: 6px; border: 1px solid;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); cursor: pointer; transition: all 0.15s ease;
            display: flex; align-items: center; justify-content: center;
        }
        div.stButton > button:hover { transform: scale(1.08); box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3); }
        div.stButton > button:active { transform: scale(0.95); box-shadow: none; }
        div.stButton > button[kind="player_btn"] { background: linear-gradient(to bottom, #007bff, #0056b3); border-color: #0056b3; color: white; }
        div.stButton > button[kind="player_btn"]:hover { background: linear-gradient(to bottom, #339cff, #007bff); }
        div.stButton > button[kind="banker_btn"] { background: linear-gradient(to bottom, #dc3545, #a71d2a); border-color: #a71d2a; color: white; }
        div.stButton > button[kind="banker_btn"]:hover { background: linear-gradient(to bottom, #ff6666, #dc3545); }
        div.stButton > button[kind="tie_btn"] { background: linear-gradient(to bottom, #28a745, #1e7e34); border-color: #1e7e34; color: white; }
        div.stButton > button[kind="tie_btn"]:hover { background: linear-gradient(to bottom, #4caf50, #28a745); }
        div.stButton > button[kind="undo_btn"] { background: linear-gradient(to bottom, #6c757d, #545b62); border-color: #545b62; color: white; }
        div.stButton > button[kind="undo_btn"]:hover { background: linear-gradient(to bottom, #8e959c, #6c757d); }
        @media (max-width: 600px) { div.stButton > button { width: 80%; max-width: 150px; height: 40px; font-size: 12px; } }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Player", key="player_btn"):
                place_result("P")
        with col2:
            if st.button("Banker", key="banker_btn"):
                place_result("B")
        with col3:
            if st.button("Tie", key="tie_btn"):
                place_result("T")
        with col4:
            if st.button("Undo Last", key="undo_btn"):
                try:
                    if not st.session_state.sequence:
                        st.warning("No results to undo.")
                    else:
                        if st.session_state.history:
                            last = st.session_state.history.pop()
                            previous_state = last['Previous_State']
                            for key, value in previous_state.items():
                                st.session_state[key] = value
                            st.session_state.sequence.pop()
                            if last['Bet_Placed'] and not last['Win'] and st.session_state.loss_log:
                                if st.session_state.loss_log[-1]['result'] == last['Result']:
                                    st.session_state.loss_log.pop()
                            if last['Bet_Placed']:
                                if last['Win']:
                                    logging.debug(f"Undo win: Reducing wins from {st.session_state.wins} to {st.session_state.wins - 1}")
                                else:
                                    logging.debug(f"Undo loss: Reducing losses from {st.session_state.losses} to {st.session_state.losses - 1}")
                            if st.session_state.pending_bet:
                                amount, pred = st.session_state.pending_bet
                                conf = predict_next()[1]
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
                    logging.error(f"Undo error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"Error undoing last action: {str(e)}")
        logging.debug("render_result_input completed")
    except Exception as e:
        logging.error(f"render_result_input error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering result input. Try resetting the session.")

def render_bead_plate():
    """Render the current sequence as a bead plate."""
    logging.debug("Entering render_bead_plate")
    try:
        st.subheader("Current Sequence (Bead Plate)")
        sequence = st.session_state.sequence[-90:]
        grid = [[] for _ in range(15)]
        for i, result in enumerate(sequence):
            col_index = i // 6
            if col_index < 15:
                grid[col_index].append(result)
        for col in grid:
            while len(col) < 6:
                col.append('')

        bead_plate_html = "<div style='display: flex; flex-direction: row; gap: 5px; max-width: 100%; overflow-x: auto;'>"
        for col in grid:
            col_html = "<div style='display: flex; flex-direction: column; gap: 5px;'>"
            for result in col:
                style = (
                    "width: 20px; height: 20px; border: 1px solid #ddd; border-radius: 50%;" if result == '' else
                    f"width: 20px; height: 20px; background-color: {'blue' if result == 'P' else 'red' if result == 'B' else 'green'}; border-radius: 50%;"
                )
                col_html += f"<div style='{style}'></div>"
            col_html += "</div>"
            bead_plate_html += col_html
        bead_plate_html += "</div>"
        st.markdown(bead_plate_html, unsafe_allow_html=True)
        logging.debug("render_bead_plate completed")
    except Exception as e:
        logging.error(f"render_bead_plate error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering bead plate. Try resetting the session.")

def render_prediction():
    """Render the current prediction and advice."""
    logging.debug("Entering render_prediction")
    try:
        if st.session_state.pending_bet:
            amount, side = st.session_state.pending_bet
            if amount is not None:
                color = 'blue' if side == 'P' else 'red'
                st.markdown(f"<h4 style='color:{color};'>Prediction: {side} | Bet: ${amount:.2f}</h4>", unsafe_allow_html=True)
            else:
                st.info("No bet placed: Check conditions (e.g., bankroll, risk limits).")
        elif not st.session_state.target_hit:
            st.info(st.session_state.advice)
        logging.debug("render_prediction completed")
    except Exception as e:
        logging.error(f"render_prediction error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering prediction. Try resetting the session.")

def render_insights():
    """Render prediction insights with error handling."""
    logging.debug("Entering render_insights")
    try:
        st.subheader("Prediction Insights")
        
        if not st.session_state.insights:
            st.info("No insights available yet. Enter more results to analyze patterns.")
            return

        try:
            _, _, _, _, _, _, _, _, _, extra_metrics = analyze_patterns(st.session_state.sequence[-WINDOW_SIZE:])
        except Exception as e:
            logging.error(f"analyze_patterns in render_insights error: {str(e)}\n{traceback.format_exc()}")
            st.error("Error analyzing patterns. Try resetting the session.")
            return

        with st.expander("Pattern Analysis", expanded=False):
            if 'Recommendation' in st.session_state.insights:
                st.markdown(f"**Recommendation**: {st.session_state.insights['Recommendation']['text']}")

            for pattern in ['Bigram', 'Trigram', 'Fourgram', 'Streak', 'Chop', 'Double']:
                if pattern in st.session_state.insights:
                    details = st.session_state.insights[pattern]
                    st.write(f"**{pattern}**")
                    st.write(f"- Weight: {details['weight']:.1f}%")
                    if 'p_prob' in details:
                        st.write(f"- Player Probability: {details['p_prob']:.1f}%")
                        st.write(f"- Banker Probability: {details['b_prob']:.1f}%")
                    if 'reliability' in details:
                        st.write(f"- Reliability: {details['reliability']:.1f}%")
                    if 'recent_performance' in details:
                        st.write(f"- Recent Performance: {details['recent_performance']:.1f}%")
                    if 'streak_type' in details:
                        st.write(f"- Streak Type: {details['streak_type']}")
                        st.write(f"- Streak Count: {details['streak_count']}")
                    if 'next_pred' in details:
                        st.write(f"- Next Prediction: {details['next_pred']}")
                        st.write(f"- Chop Count: {details['chop_count']}")
                    if 'double_type' in details:
                        st.write(f"- Double Type: {details['double_type']}")

            if 'Shoe Bias' in st.session_state.insights:
                st.write("**Shoe Bias**")
                st.write(f"- Bias: {st.session_state.insights['Shoe Bias']['bias']}")
                st.write(f"- Adjustment: {st.session_state.insights['Shoe Bias']['adjustment']}")

            if 'Threshold' in st.session_state.insights:
                st.write("**Threshold**")
                st.write(f"- Adjusted Threshold: {st.session_state.insights['Threshold']['adjusted']}")

            if 'Volatility' in st.session_state.insights:
                st.write("**Volatility**")
                st.write(f"- Level: {st.session_state.insights['Volatility']['level']}")
                st.write(f"- Value: {st.session_state.insights['Volatility']['value']:.2f}")
                st.write(f"- Adjustment: {st.session_state.insights['Volatility']['adjustment']}")

            if 'No Bet' in st.session_state.insights:
                st.write("**No Bet Reason**")
                st.write(f"- Reason: {st.session_state.insights['No Bet']['reason']}")

        with st.expander("Extra Metrics", expanded=False):
            st.write(f"- Average Streak Length: {extra_metrics['avg_streak_length']:.2f}")
            st.write(f"- Average Chop Length: {extra_metrics['avg_chop_length']:.2f}")
            st.write(f"- Streak Frequency: {extra_metrics['streak_frequency']:.2f}")
            st.write(f"- Chop Frequency: {extra_metrics['chop_frequency']:.2f}")

        logging.debug("render_insights completed")
    except Exception as e:
        logging.error(f"render_insights error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering insights. Try resetting the session.")

def render_status():
    """Render the session status section."""
    logging.debug("Entering render_status")
    try:
        st.subheader("Session Status")
        st.markdown("""
        <style>
        .status-box {
            border: 2px solid #007bff;
            border-radius: 8px;
            padding: 15px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .status-label {
            font-weight: bold;
            color: #343a40;
        }
        .status-value {
            color: #007bff;
        }
        .status-negative {
            color: #dc3545;
        }
        .status-neutral {
            color: #6c757d;
        }
        @media (max-width: 600px) {
            .status-box {
                padding: 10px;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        profit_loss = st.session_state.bankroll - st.session_state.initial_bankroll
        profit_loss_pct = (profit_loss / st.session_state.initial_bankroll * 100) if st.session_state.initial_bankroll > 0 else 0.0
        total_bets = st.session_state.wins + st.session_state.losses
        win_rate = (st.session_state.wins / total_bets * 100) if total_bets > 0 else 0.0

        if st.session_state.target_mode == "Profit %":
            target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
            progress = (profit_loss / target_profit * 100) if target_profit > 0 else 0.0
            target_text = f"{st.session_state.target_value}% Profit (${target_profit:.2f})"
        else:
            target_units = st.session_state.target_value
            units_earned = profit_loss / st.session_state.initial_base_bet if st.session_state.initial_base_bet > 0 else 0.0
            progress = (units_earned / target_units * 100) if target_units > 0 else 0.0
            target_text = f"{target_units} Units"

        status_html = "<div class='status-box'>"
        status_html += f"<p><span class='status-label'>Bankroll:</span> <span class='status-value'>${st.session_state.bankroll:.2f}</span></p>"
        status_html += f"<p><span class='status-label'>Profit/Loss:</span> <span class={'status-value' if profit_loss >= 0 else 'status-negative'}>${profit_loss:.2f} ({profit_loss_pct:.1f}%)</span></p>"
        status_html += f"<p><span class='status-label'>Wins/Losses:</span> <span class='status-value'>{st.session_state.wins}/{st.session_state.losses}</span> (Win Rate: {win_rate:.1f}%)</p>"
        status_html += f"<p><span class='status-label'>Strategy:</span> <span class='status-value'>{st.session_state.strategy}</span></p>"

        if st.session_state.strategy == 'T3':
            status_html += f"<p><span class='status-label'>T3 Level:</span> <span class='status-value'>{st.session_state.t3_level}</span> (Peak: {st.session_state.t3_peak_level}, Changes: {st.session_state.t3_level_changes})</p>"
        elif st.session_state.strategy == 'Parlay16':
            status_html += f"<p><span class='status-label'>Parlay Step:</span> <span class='status-value'>{st.session_state.parlay_step}</span> (Peak: {st.session_state.parlay_peak_step}, Wins: {st.session_state.parlay_wins})</p>"
        elif st.session_state.strategy == 'Z1003.1':
            status_html += f"<p><span class='status-label'>Z1003 Loss Count:</span> <span class='status-value'>{st.session_state.z1003_loss_count}</span> (Bet Factor: {st.session_state.z1003_bet_factor:.2f})</p>"

        status_html += f"<p><span class='status-label'>Target:</span> <span class='status-value'>{target_text}</span> (Progress: {progress:.1f}%)</p>"
        status_html += f"<p><span class='status-label'>Safety Net:</span> <span class='status-value'>{'Enabled' if st.session_state.safety_net_enabled else 'Disabled'}</span> ({st.session_state.safety_net_percentage}%)</p>"
        status_html += f"<p><span class='status-label'>Consecutive Wins/Losses:</span> <span class='status-value'>{st.session_state.consecutive_wins}/{st.session_state.consecutive_losses}</span></p>"
        status_html += f"<p><span class='status-label'>Pattern Volatility:</span> <span class='status-value'>{st.session_state.pattern_volatility:.2f}</span></p>"
        status_html += "</div>"

        st.markdown(status_html, unsafe_allow_html=True)
        logging.debug("render_status completed")
    except Exception as e:
        logging.error(f"render_status error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error rendering status section. Try resetting the session.")

# --- Main Application ---
def main():
    """Main application logic."""
    logging.debug("Entering main")
    try:
        st.set_page_config(page_title="Baccarat Predictor", layout="wide")
        st.title(f"Baccarat Predictor v{APP_VERSION}")
        initialize_session_state()

        num_users = track_user_session()
        st.sidebar.write(f"Active Users: {num_users}")

        if st.sidebar.button("Reset Session"):
            reset_session()
            st.success("Session reset successfully!")
            st.rerun()

        if st.session_state.target_hit:
            st.success("Target hit! Session ended. Reset to start a new session.")
            return

        render_setup_form()
        if st.session_state.bankroll > 0:
            st.sidebar.subheader("Session Stats")
            st.sidebar.write(f"Bankroll: ${st.session_state.bankroll:.2f}")
            st.sidebar.write(f"Wins: {st.session_state.wins}")
            st.sidebar.write(f"Losses: {st.session_state.losses}")
            st.sidebar.write(f"Strategy: {st.session_state.strategy}")
            if st.session_state.strategy == 'T3':
                st.sidebar.write(f"T3 Level: {st.session_state.t3_level}")
                st.sidebar.write(f"T3 Peak Level: {st.session_state.t3_peak_level}")
                st.sidebar.write(f"T3 Level Changes: {st.session_state.t3_level_changes}")
            elif st.session_state.strategy == 'Parlay16':
                st.sidebar.write(f"Parlay Step: {st.session_state.parlay_step}")
                st.sidebar.write(f"Parlay Peak Step: {st.session_state.parlay_peak_step}")
                st.sidebar.write(f"Parlay Step Changes: {st.session_state.parlay_step_changes}")
                st.sidebar.write(f"Parlay Wins: {st.session_state.parlay_wins}")
            elif st.session_state.strategy == 'Z1003.1':
                st.sidebar.write(f"Z1003 Loss Count: {st.session_state.z1003_loss_count}")
                st.sidebar.write(f"Z1003 Bet Factor: {st.session_state.z1003_bet_factor:.2f}")
                st.sidebar.write(f"Z1003 Level Changes: {st.session_state.z1003_level_changes}")
            st.sidebar.write(f"Consecutive Wins: {st.session_state.consecutive_wins}")
            st.sidebar.write(f"Consecutive Losses: {st.session_state.consecutive_losses}")
            st.sidebar.write(f"Pattern Volatility: {st.session_state.pattern_volatility:.2f}")

            col1, col2 = st.columns([2, 1])
            with col1:
                render_status()
                render_result_input()
                render_prediction()
                render_bead_plate()
            with col2:
                render_insights()

            with st.expander("Loss Log", expanded=False):
                if st.session_state.loss_log:
                    for i, log in enumerate(st.session_state.loss_log):
                        st.write(f"**Loss {i+1}**")
                        st.write(f"- Sequence: {''.join(log['sequence'])}")
                        st.write(f"- Prediction: {log['prediction']}")
                        st.write(f"- Result: {log['result']}")
                        st.write(f"- Confidence: {log['confidence']}%")
                        st.write("- Insights:")
                        for key, value in log['insights'].items():
                            st.write(f"  - {key}: {value}")
                else:
                    st.info("No losses recorded yet.")

            with st.expander("Run Simulation", expanded=False):
                if st.button("Simulate Shoe (80 hands)"):
                    result = simulate_shoe()
                    st.write(f"**Simulation Results**")
                    st.write(f"- Accuracy: {result['accuracy']:.1f}%")
                    st.write(f"- Correct Predictions: {result['correct']}/{result['total']}")
                    st.write(f"- Sequence: {''.join(result['sequence'])}")
                    st.write("- Pattern Performance:")
                    for pattern in result['pattern_success']:
                        st.write(f"  - {pattern}: {result['pattern_success'][pattern]}/{result['pattern_attempts'][pattern]}")

        logging.debug("main completed")
    except Exception as e:
        logging.error(f"main error: {str(e)}\n{traceback.format_exc()}")
        st.error("Error running application. Try resetting the session.")

if __name__ == "__main__":
    main()
