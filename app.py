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
    
    _, _, _, _, _, streak_count, chop_count, double_count, volatility, shoe_bias = st.session_state.analyze_patterns(sequence[-WINDOW_SIZE:])
    
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
