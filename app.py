import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random
from collections import defaultdict
from datetime import datetime
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PARLAY_TABLE = {
    i: {'base': b, 'parlay': p} for i, (b, p) in enumerate([
        (1, 2), (1, 2), (1, 2), (2, 4), (3, 6), (4, 8), (6, 12), (8, 16),
        (12, 24), (16, 32), (22, 44), (30, 60), (40, 80), (52, 104), (70, 140), (95, 190)
    ], 1)
}
STRATEGIES = ["T3", "Flatbet", "Parlay16"]

# Placeholder for profit_enhancements module
def adjust_safety_net():
    return st.session_state.get('safety_net_percentage', 10.0)

def recommend_strategy(sequence):
    return "T3"

def calculate_roi():
    profit = st.session_state.bankroll - st.session_state.initial_bankroll
    return (profit / st.session_state.initial_bankroll * 100) if st.session_state.initial_bankroll > 0 else 0.0

# Simulate Baccarat game data for training
def generate_baccarat_data(num_games=10000):
    outcomes = ['P', 'B']
    return [random.choice(outcomes) for _ in range(num_games)]

# Preprocess data: Create sequences for prediction
def prepare_data(outcomes, sequence_length=10):
    le = LabelEncoder()
    encoded_outcomes = le.fit_transform(outcomes)
    X, y = [], []
    for i in range(len(encoded_outcomes) - sequence_length):
        X.append(encoded_outcomes[i:i + sequence_length])
        y.append(encoded_outcomes[i + sequence_length])
    return np.array(X), np.array(y), le

# Simplified prediction logic
def predict_next():
    sequence = st.session_state.user_sequence
    if len(sequence) < st.session_state.sequence_length or not st.session_state.model:
        return None, 45.86, {}
    encoded_input = st.session_state.le.transform(sequence[-st.session_state.sequence_length:])
    input_array = np.array([encoded_input])
    prediction_probs = st.session_state.model.predict_proba(input_array)[0]
    predicted_class = np.argmax(prediction_probs)
    predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction_probs) * 100
    insights = {"Model Confidence": f"{confidence:.1f}%"}
    return predicted_outcome, confidence, insights

# Initialize session state
def initialize_session_state():
    defaults = {
        'sequence_length': 10,
        'user_sequence': [],
        'bet_history': [],
        'pending_bet': None,
        'bankroll': 0.0,
        'base_bet': 0.10,
        'initial_bankroll': 0.0,
        'initial_base_bet': 0.10,
        'stop_loss': 0.8,
        'win_limit': 1.5,
        'bets_placed': 0,
        'bets_won': 0,
        'model': None,
        'le': None,
        't3_level': 1,
        't3_results': [],
        'session_active': False,
        'strategy': 'T3',
        'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
        'consecutive_losses': 0,
        'loss_log': [],
        'insights': {},
        'safety_net_percentage': 10.0,
        'safety_net_enabled': True,
        'ai_automation_enabled': True,
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_using_base': True,
        'parlay_step_changes': 0,
        'parlay_peak_step': 1
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Apply custom CSS
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

# UI Components
def render_setup_form():
    try:
        with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
            with st.form("setup_form"):
                col1, col2 = st.columns(2)
                with col1:
                    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=st.session_state.initial_bankroll or 100.0, step=10.0)
                    base_bet = st.number_input("Base Bet ($)", min_value=0.10, value=st.session_state.initial_base_bet or 0.10, step=0.10)
                with col2:
                    betting_strategy = st.selectbox("Betting Strategy", STRATEGIES, index=STRATEGIES.index(st.session_state.strategy), help="T3: Adjusts bet size based on wins/losses. Flatbet: Fixed bet size. Parlay16: 16-step progression.")
                    target_mode = st.radio("Target Type", ["Profit %", "Units"], index=0)
                    target_value = st.number_input("Target Value", min_value=1.0, value=10.0, step=1.0)
                safety_net_enabled = st.checkbox("Enable Safety Net", value=st.session_state.safety_net_enabled)
                safety_net_percentage = st.session_state.safety_net_percentage
                if safety_net_enabled:
                    safety_net_percentage = st.number_input("Safety Net Percentage (%)", min_value=0.0, max_value=50.0, value=st.session_state.safety_net_percentage, step=5.0)
                if st.form_submit_button("Start Session"):
                    if bankroll <= 0:
                        st.error("Bankroll must be positive.")
                    elif base_bet < 0.10:
                        st.error("Base bet must be at least $0.10.")
                    elif base_bet > bankroll * 0.05:
                        st.error("Base bet cannot exceed 5% of bankroll.")
                    else:
                        st.session_state.update({
                            'bankroll': bankroll,
                            'base_bet': base_bet,
                            'initial_base_bet': base_bet,
                            'initial_bankroll': bankroll,
                            'user_sequence': [],
                            'bet_history': [],
                            'pending_bet': None,
                            'bets_placed': 0,
                            'bets_won': 0,
                            't3_level': 1,
                            't3_results': [],
                            'session_active': True,
                            'strategy': betting_strategy,
                            'target_mode': target_mode,
                            'target_value': target_value,
                            'prediction_accuracy': {'P': 0, 'B': 0, 'total': 0},
                            'consecutive_losses': 0,
                            'loss_log': [],
                            'insights': {},
                            'safety_net_percentage': safety_net_percentage,
                            'safety_net_enabled': safety_net_enabled,
                            'ai_automation_enabled': True,
                            'parlay_step': 1,
                            'parlay_wins': 0,
                            'parlay_using_base': True,
                            'parlay_step_changes': 0,
                            'parlay_peak_step': 1
                        })
                        outcomes = generate_baccarat_data()
                        X, y, st.session_state.le = prepare_data(outcomes, st.session_state.sequence_length)
                        st.session_state.model = RandomForestClassifier(n_estimators=100, random_state=42)
                        st.session_state.model.fit(X, y)
                        st.success(f"Session started with {betting_strategy} strategy! AI Automation: Enabled.")
                        st.rerun()
    except Exception as e:
        logger.error(f"Setup form rendering failed: {e}")
        st.error("Error rendering setup form. Please refresh.")

def render_result_input():
    try:
        with st.expander("Enter Shoe Results", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Player", key="player_btn", help="Record a Player win"):
                    add_result('P')
                    st.rerun()
            with col2:
                if st.button("Banker", key="banker_btn", help="Record a Banker win"):
                    add_result('B')
                    st.rerun()
            with col3:
                if st.button("Undo", key="undo_btn", help="Undo the last result", disabled=not st.session_state.bet_history):
                    undo_result()
                    st.rerun()
    except Exception as e:
        logger.error(f"Result input rendering failed: {e}")
        st.error("Error rendering result input. Please refresh.")

def render_bead_plate():
    try:
        with st.expander("Bead Plate", expanded=True):
            sequence = st.session_state.user_sequence[-90:]
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
                if st.session_state.strategy == 'T3':
                    st.markdown(f"**Strategy**: T3<br>Level: {st.session_state.t3_level}<br>Results: {st.session_state.t3_results}", unsafe_allow_html=True)
                elif st.session_state.strategy == 'Parlay16':
                    st.markdown(f"**Strategy**: Parlay16<br>Step: {st.session_state.parlay_step}/16 | Wins: {st.session_state.parlay_wins}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Strategy**: Flatbet", unsafe_allow_html=True)
                st.markdown(f"**AI Automation**: {'Enabled' if st.session_state.ai_automation_enabled else 'Disabled'}")
            st.markdown(f"**Wins**: {st.session_state.bets_won} | **Losses**: {st.session_state.bets_placed - st.session_state.bets_won}")
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
            if st.session_state.bet_history:
                accuracy_data = []
                correct = total = 0
                for h in st.session_state.bet_history[-50:]:
                    if h[2] and h[3]:
                        total += 1
                        if h[3] == 'win':
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
                        "Confidence": f"{log['confidence']}%"
                    }
                    for log in st.session_state.loss_log[-5:]
                ], use_container_width=True)
    except Exception as e:
        logger.error(f"Loss log rendering failed: {e}")
        st.error("Error rendering loss log. Please refresh.")

def render_history():
    try:
        with st.expander("Bet History"):
            if st.session_state.bet_history:
                n = st.slider("Show last N bets", 5, 50, 10)
                st.dataframe([
                    {
                        "Bet": h[2] if h[2] else "-",
                        "Result": h[0],
                        "Amount": f"${h[1]:.2f}" if h[1] > 0 else "-",
                        "Outcome": h[3] if h[3] else "-",
                        "T3_Level": h[4] if st.session_state.strategy == 'T3' else "-",
                        "Parlay_Step": h[5] if st.session_state.strategy == 'Parlay16' else "-"
                    }
                    for h in st.session_state.bet_history[-n:]
                ], use_container_width=True)
    except Exception as e:
        logger.error(f"History rendering failed: {e}")
        st.error("Error rendering history. Please refresh.")

def render_export():
    try:
        with st.expander("Export Session"):
            if st.button("Download Session Data"):
                csv_data = "Result,Bet,Amount,Outcome,T3_Level,Parlay_Step\n"
                for h in st.session_state.bet_history:
                    t3_level = h[4] if st.session_state.strategy == 'T3' else '-'
                    parlay_step = h[5] if st.session_state.strategy == 'Parlay16' else '-'
                    csv_data += f"{h[0]},{h[2] or '-'},${h[1]:.2f},{h[3] or '-'},{t3_level},{parlay_step}\n"
                st.download_button("Download CSV", csv_data, "session_data.csv", "text/csv")
    except Exception as e:
        logger.error(f"Export rendering failed: {e}")
        st.error("Error rendering export. Please refresh.")

def render_simulation():
    try:
        with st.expander("Run Simulation"):
            num_hands = st.number_input("Number of Hands to Simulate", min_value=10, max_value=200, value=100, step=10)
            if st.button("Run Simulation"):
                outcomes = np.random.choice(['P', 'B'], size=num_hands, p=[0.5, 0.5])
                initial_bankroll = st.session_state.bankroll
                wins = losses = 0
                for outcome in outcomes:
                    add_result(outcome)
                    if st.session_state.bet_history[-1][3] == 'win':
                        wins += 1
                    elif st.session_state.bet_history[-1][3] == 'loss':
                        losses += 1
                st.write(f"**Simulation Results**")
                st.write(f"Final Bankroll: ${st.session_state.bankroll:.2f}")
                st.write(f"Wins: {wins} | Losses: {losses}")
    except Exception as e:
        logger.error(f"Simulation rendering failed: {e}")
        st.error("Error rendering simulation. Please refresh.")

def render_profit_dashboard():
    try:
        st.markdown("**Profit Dashboard**")
        st.markdown(f"ROI: {calculate_roi():.2f}%")
    except Exception as e:
        logger.error(f"Profit dashboard rendering failed: {e}")
        st.error("Error rendering profit dashboard. Please refresh.")

# Core Logic
def add_result(result):
    if st.session_state.model is None:
        st.error("Please start a session with bankroll and bet.")
        return
    if st.session_state.safety_net_enabled:
        safe_bankroll = st.session_state.initial_bankroll * (st.session_state.safety_net_percentage / 100)
        if st.session_state.bankroll <= safe_bankroll:
            st.error(f"Bankroll below {st.session_state.safety_net_percentage:.0f}%. Session ended. Reset or exit.")
            st.session_state.session_active = False
            return
    if st.session_state.target_mode == "Profit %":
        target_profit = st.session_state.initial_bankroll * (st.session_state.target_value / 100)
        if st.session_state.bankroll >= st.session_state.initial_bankroll + target_profit:
            st.success(f"Target profit of {st.session_state.target_value}% reached. Session ended.")
            st.session_state.session_active = False
            return
    elif st.session_state.target_mode == "Units":
        unit_profit = (st.session_state.bankroll - st.session_state.initial_bankroll) / st.session_state.initial_base_bet
        if unit_profit >= st.session_state.target_value:
            st.success(f"Target of {st.session_state.target_value} units reached. Session ended.")
            st.session_state.session_active = False
            return

    bet_amount = 0
    bet_selection = None
    bet_outcome = None
    if st.session_state.pending_bet:
        bet_amount, bet_selection = st.session_state.pending_bet
        st.session_state.bets_placed += 1
        if result == bet_selection:
            if bet_selection == 'B':
                st.session_state.bankroll += bet_amount * 0.95
            else:
                st.session_state.bankroll += bet_amount
            st.session_state.bets_won += 1
            bet_outcome = 'win'
            if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 0:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
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
            st.session_state.prediction_accuracy[bet_selection] += 1
            st.session_state.consecutive_losses = 0
        else:
            st.session_state.bankroll -= bet_amount
            bet_outcome = 'loss'
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
            st.session_state.consecutive_losses += 1
            st.session_state.loss_log.append({
                'sequence': st.session_state.user_sequence[-10:],
                'prediction': bet_selection,
                'result': result,
                'confidence': st.session_state.insights.get('Model Confidence', '0%')
            })
            if len(st.session_state.loss_log) > 50:
                st.session_state.loss_log = st.session_state.loss_log[-50:]
        st.session_state.prediction_accuracy['total'] += 1
        if st.session_state.strategy == 'T3' and len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins > losses:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses > wins:
                st.session_state.t3_level += 1
            st.session_state.t3_results = []
        st.session_state.pending_bet = None

    st.session_state.user_sequence.append(result)
    if len(st.session_state.user_sequence) > 100:
        st.session_state.user_sequence.pop(0)

    if len(st.session_state.user_sequence) >= st.session_state.sequence_length:
        pred, conf, insights = predict_next()
        st.session_state.insights = insights
        bet_selection = None
        if len(st.session_state.user_sequence) >= 6:
            sixth_prior = st.session_state.user_sequence[-6]
            outcome_index = st.session_state.le.transform([sixth_prior])[0]
            sixth_confidence = st.session_state.model.predict_proba(np.array([st.session_state.le.transform(st.session_state.user_sequence[-st.session_state.sequence_length:])]))[0][outcome_index] * 100
            if sixth_confidence > 40:
                bet_selection = sixth_prior
        if bet_selection:
            if st.session_state.strategy == 'Flatbet':
                bet_amount = st.session_state.base_bet
            elif st.session_state.strategy == 'T3':
                bet_amount = st.session_state.base_bet * st.session_state.t3_level
            elif st.session_state.strategy == 'Parlay16':
                key = 'base' if st.session_state.parlay_using_base else 'parlay'
                bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
            if bet_amount <= st.session_state.bankroll:
                st.session_state.pending_bet = (bet_amount, bet_selection)
                st.session_state.advice = f"Auto Bet: ${bet_amount:.2f} on {bet_selection}"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = "No bet: Insufficient bankroll."
        else:
            st.session_state.pending_bet = None
            st.session_state.advice = "Skip betting (no 6th prior or low confidence)."

    st.session_state.bet_history.append((result, bet_amount, bet_selection, bet_outcome, st.session_state.t3_level, st.session_state.parlay_step))
    if len(st.session_state.bet_history) > 1000:
        st.session_state.bet_history = st.session_state.bet_history[-1000:]

def undo_result():
    if not st.session_state.user_sequence:
        st.warning("No results to undo.")
        return
    st.session_state.user_sequence.pop()
    if st.session_state.bet_history:
        last_bet = st.session_state.bet_history.pop()
        result, bet_amount, bet_selection, bet_outcome, t3_level, parlay_step = last_bet
        if bet_amount > 0:
            st.session_state.bets_placed -= 1
            if bet_outcome == 'win':
                if bet_selection == 'B':
                    st.session_state.bankroll -= bet_amount * 0.95
                else:
                    st.session_state.bankroll -= bet_amount
                st.session_state.bets_won -= 1
                st.session_state.prediction_accuracy[bet_selection] -= 1
                st.session_state.prediction_accuracy['total'] -= 1
                if st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_wins -= 1
                    if st.session_state.parlay_wins < 0:
                        st.session_state.parlay_wins = 0
                    st.session_state.parlay_using_base = True
            elif bet_outcome == 'loss':
                st.session_state.bankroll += bet_amount
                st.session_state.consecutive_losses -= 1
                if st.session_state.loss_log and st.session_state.loss_log[-1]['result'] == result:
                    st.session_state.loss_log.pop()
                if st.session_state.strategy == 'Parlay16':
                    st.session_state.parlay_step = max(1, st.session_state.parlay_step - 1)
                    st.session_state.parlay_using_base = False
            if st.session_state.strategy == 'T3':
                st.session_state.t3_level = t3_level
                st.session_state.t3_results = []
            elif st.session_state.strategy == 'Parlay16':
                st.session_state.parlay_step = parlay_step
    if len(st.session_state.user_sequence) >= st.session_state.sequence_length - 1:
        pred, conf, insights = predict_next()
        st.session_state.insights = insights
        bet_selection = None
        if len(st.session_state.user_sequence) >= 6:
            sixth_prior = st.session_state.user_sequence[-6]
            outcome_index = st.session_state.le.transform([sixth_prior])[0]
            sixth_confidence = st.session_state.model.predict_proba(np.array([st.session_state.le.transform(st.session_state.user_sequence[-st.session_state.sequence_length:])]))[0][outcome_index] * 100
            if sixth_confidence > 40:
                bet_selection = sixth_prior
        if bet_selection:
            if st.session_state.strategy == 'Flatbet':
                bet_amount = st.session_state.base_bet
            elif st.session_state.strategy == 'T3':
                bet_amount = st.session_state.base_bet * st.session_state.t3_level
            elif st.session_state.strategy == 'Parlay16':
                key = 'base' if st.session_state.parlay_using_base else 'parlay'
                bet_amount = st.session_state.initial_base_bet * PARLAY_TABLE[st.session_state.parlay_step][key]
            if bet_amount <= st.session_state.bankroll:
                st.session_state.pending_bet = (bet_amount, bet_selection)
                st.session_state.advice = f"Auto Bet: ${bet_amount:.2f} on {bet_selection}"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = "No bet: Insufficient bankroll."
        else:
            st.session_state.pending_bet = None
            st.session_state.advice = "Skip betting (no 6th prior or low confidence)."

def get_advice():
    if len(st.session_state.user_sequence) < st.session_state.sequence_length:
        return f"Need {st.session_state.sequence_length - len(st.session_state.user_sequence)} more results"
    elif st.session_state.pending_bet:
        bet_amount, bet_selection = st.session_state.pending_bet
        if st.session_state.strategy == 'Flatbet':
            return f"Bet ${bet_amount:.2f} on {bet_selection} (Flatbet)."
        elif st.session_state.strategy == 'T3':
            return f"Bet ${bet_amount:.2f} on {bet_selection} (T3 Level {st.session_state.t3_level}, mirroring 6th prior)."
        elif st.session_state.strategy == 'Parlay16':
            return f"Bet ${bet_amount:.2f} on {bet_selection} (Parlay16 Step {st.session_state.parlay_step})."
    else:
        return "Skip betting (no 6th prior or low confidence)."

# Main Application
def main():
    try:
        st.set_page_config(layout="wide", page_title="MANG BACCARAT GROUP")
        apply_custom_css()
        initialize_session_state()
        st.title("MANG BACCARAT GROUP")
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
        st.error("Application failed to load. Please refresh.")

if __name__ == "__main__":
    main()
