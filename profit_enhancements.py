import streamlit as st
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Constants from app.py
STRATEGIES = ["T3", "Flatbet", "Parlay16", "Z1003.1"]
SHOE_SIZE = 100
WINDOW_SIZE = 50

def adjust_safety_net() -> float:
    """Dynamically adjust safety net percentage based on recent performance."""
    total_bets = max(st.session_state.prediction_accuracy['total'], 1)
    win_rate = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / total_bets
    volatility = st.session_state.pattern_volatility
    consecutive_losses = st.session_state.consecutive_losses
    
    base_percentage = st.session_state.safety_net_percentage
    if consecutive_losses >= 3:
        return min(base_percentage + 10, 50.0)
    elif win_rate > 0.6 and volatility < 0.3:
        return max(base_percentage - 5, 5.0)
    return base_percentage

def recommend_strategy(sequence: List[str]) -> str:
    """Recommend a betting strategy based on shoe patterns."""
    if len(sequence) < 4:
        return "Flatbet"
    
    _, _, _, _, _, streak_count, chop_count, double_count, volatility, shoe_bias = analyze_patterns(sequence[-WINDOW_SIZE:])
    
    if volatility > 0.5 or abs(shoe_bias) < 0.1:
        return "Flatbet"
    elif streak_count >= 3:
        return "Parlay16"
    elif chop_count >= 3:
        return "Z1003.1"
    return "T3"

def enhanced_z1003_bet(loss_count: int, base_bet: float) -> float:
    """Modified Z1003.1 with loss recovery mode."""
    if loss_count == 0:
        return base_bet
    recovery_factor = min(1.5 ** loss_count, 4.0)
    return base_bet * recovery_factor

def calculate_roi() -> Dict[str, float]:
    """Calculate ROI for each strategy based on history."""
    roi = {s: 0.0 for s in STRATEGIES}
    strategy_counts = defaultdict(int)
    
    for h in st.session_state.history:
        if h['Bet_Placed']:
            strategy = st.session_state.strategy
            # Fixed: Use history's strategy if available, else current strategy
            strategy = h.get('Strategy', st.session_state.strategy)
            profit = h['Amount'] * (0.95 if h['Bet'] == 'B' and h['Win'] else 1.0 if h['Win'] else -1.0)
            roi[strategy] += profit
            strategy_counts[strategy] += 1
    
    for strategy in roi:
        if strategy_counts[strategy] > 0:
            roi[strategy] /= strategy_counts[strategy]
    return roi

def render_profit_dashboard():
    """Render a dashboard with profitability metrics."""
    with st.expander("Profitability Dashboard", expanded=True):
        roi = calculate_roi()
        st.markdown("**Strategy ROI (Average Profit per Bet)**")
        for strategy, value in roi.items():
            st.markdown(f"{strategy}: ${value:.2f}")
        
        total_bets = max(st.session_state.prediction_accuracy['total'], 1)
        win_rate = (st.session_state.wins / total_bets) * 100
        st.markdown(f"**Overall Win Rate**: {win_rate:.1f}%")
        
        profit = st.session_state.bankroll - st.session_state.initial_bankroll
        st.markdown(f"**Total Profit**: ${profit:.2f}")
        
        recommended_strategy = recommend_strategy(st.session_state.sequence)
        st.markdown(f"**Recommended Strategy**: {recommended_strategy}")

# Expose analyze_patterns for strategy recommendation
def analyze_patterns(sequence: List[str]) -> Tuple[Dict, Dict, Dict, Dict, Dict, int, int, int, float, float]:
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