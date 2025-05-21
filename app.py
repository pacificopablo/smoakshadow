import streamlit as st
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import uuid
import time

# Constants
SEQUENCE_LENGTH = 6
T3_LEVELS = [1, 2, 3, 4]
PARLAY16_BET_MULTIPLIERS = [1, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
MOON_LEVELS = [1, 2, 3, 4, 5]
FOUR_TIER_LEVELS = [1, 2, 3, 4]
FOUR_TIER_STEPS = [1, 2, 3, 4]
FLATBET_LEVELUP_LEVELS = [1, 2, 3, 4, 5]

def initialize_session_state():
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 0
        st.session_state.initial_bankroll = 0
        st.session_state.base_bet = 0
        st.session_state.sequence = []
        st.session_state.transition_counts = {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0}
        st.session_state.pending_bet = None
        st.session_state.bet_history = []
        st.session_state.advice = "Start a session to get advice."
        st.session_state.model = None
        st.session_state.le = None
        st.session_state.money_management = 'Flatbet'
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.parlay_step = 1
        st.session_state.parlay_wins = 0
        st.session_state.parlay_peak_step = 1
        st.session_state.parlay_step_changes = 0
        st.session_state.moon_level = 1
        st.session_state.moon_peak_level = 1
        st.session_state.moon_level_changes = 0
        st.session_state.four_tier_level = 1
        st.session_state.four_tier_step = 1
        st.session_state.four_tier_losses = 0
        st.session_state.flatbet_levelup_level = 1
        st.session_state.flatbet_levelup_net_loss = 0
        st.session_state.bets_placed = 0
        st.session_state.bets_won = 0
        st.session_state.shoe_completed = False
        st.session_state.stop_loss_percentage = 0.5
        st.session_state.target_profit_option = 'None'
        st.session_state.target_profit_percentage = 0
        st.session_state.target_profit_units = 0
        st.session_state.safety_net_enabled = False
        st.session_state.smart_skip_enabled = False
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.model, st.session_state.le = train_model()

def reset_session():
    st.session_state.update({
        'bankroll': st.session_state.initial_bankroll,
        'sequence': [],
        'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
        'pending_bet': None,
        'bet_history': [],
        'advice': "Start placing results to get advice.",
        't3_level': 1,
        't3_results': [],
        'parlay_step': 1,
        'parlay_wins': 0,
        'parlay_peak_step': 1,
        'parlay_step_changes': 0,
        'moon_level': 1,
        'moon_peak_level': 1,
        'moon_level_changes': 0,
        'four_tier_level': 1,
        'four_tier_step': 1,
        'four_tier_losses': 0,
        'flatbet_levelup_level': 1,
        'flatbet_levelup_net_loss': 0,
        'bets_placed': 0,
        'bets_won': 0,
        'shoe_completed': False,
        'session_id': str(uuid.uuid4())
    })
    st.session_state.model, st.session_state.le = train_model()

def generate_baccarat_data(num_games=10000):
    outcomes = ['P', 'B', 'T']
    weights = [0.4462, 0.4586, 0.0952]  # Real Baccarat probabilities
    result = []
    i = 0
    while i < num_games:
        outcome = random.choices(outcomes, weights=weights, k=1)[0]
        streak_length = random.choices([1, 2, 3, 4, 5, 6], weights=[0.4, 0.3, 0.15, 0.1, 0.03, 0.02])[0]
        result.extend([outcome] * streak_length)
        i += streak_length
    return result[:num_games]

def prepare_data(outcomes, sequence_length=6):
    try:
        le = LabelEncoder()
        encoded_outcomes = le.fit_transform(outcomes)
        X, y = [], []
        for i in range(len(encoded_outcomes) - sequence_length):
            X.append(encoded_outcomes[i:i + sequence_length])
            y.append(encoded_outcomes[i + sequence_length])
        return np.array(X), np.array(y), le
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return np.array([]), np.array([]), None

def train_model():
    try:
        outcomes = generate_baccarat_data()
        X, y, le = prepare_data(outcomes, SEQUENCE_LENGTH)
        if X.size == 0 or y.size == 0 or le is None:
            st.error("Failed to generate training data.")
            return None, None
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, le
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def calculate_bet_amount(selection):
    try:
        if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
            return st.session_state.base_bet
        if st.session_state.money_management == 'Flatbet':
            return st.session_state.base_bet
        elif st.session_state.money_management == 'T3':
            return st.session_state.base_bet * T3_LEVELS[st.session_state.t3_level - 1]
        elif st.session_state.money_management == 'Parlay16':
            return st.session_state.base_bet * PARLAY16_BET_MULTIPLIERS[st.session_state.parlay_step - 1]
        elif st.session_state.money_management == 'Moon':
            return st.session_state.base_bet * MOON_LEVELS[st.session_state.moon_level - 1]
        elif st.session_state.money_management == 'FourTier':
            return st.session_state.base_bet * FOUR_TIER_LEVELS[st.session_state.four_tier_level - 1]
        elif st.session_state.money_management == 'FlatbetLevelUp':
            return st.session_state.base_bet * FLATBET_LEVELUP_LEVELS[st.session_state.flatbet_levelup_level - 1]
        return st.session_state.base_bet
    except Exception as e:
        st.error(f"Error calculating bet amount: {str(e)}")
        return st.session_state.base_bet

def place_result(result):
    try:
        st.session_state.sequence.append(result)
        if len(st.session_state.sequence) >= 1:
            prev_result = st.session_state.sequence[-1]
            if result in ['P', 'B'] and prev_result in ['P', 'B']:
                transition = f"{prev_result}{result}"
                st.session_state.transition_counts[transition] += 1

        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
        if len(valid_sequence) < SEQUENCE_LENGTH:
            st.session_state.pending_bet = None
            st.session_state.advice = f"Need {SEQUENCE_LENGTH - len(valid_sequence)} more Player or Banker results"
        elif len(valid_sequence) >= SEQUENCE_LENGTH and result in ['P', 'B']:
            prediction_sequence = valid_sequence[-SEQUENCE_LENGTH:]
            encoded_input = st.session_state.le.transform(prediction_sequence)
            input_array = np.array([encoded_input])
            prediction_probs = st.session_state.model.predict_proba(input_array)[0]
            predicted_class = np.argmax(prediction_probs)
            predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
            model_confidence = np.max(prediction_probs) * 100

            lb6_selection = None
            lb6_confidence = 0
            sixth_prior = 'N/A'
            if len(valid_sequence) >= 6:
                sixth_prior = valid_sequence[-6]
                outcome_index = st.session_state.le.transform([sixth_prior])[0]
                lb6_confidence = prediction_probs[outcome_index] * 100
                if lb6_confidence > 40 and sixth_prior == predicted_outcome:
                    lb6_selection = sixth_prior

            markov_selection = None
            markov_confidence = 0
            last_outcome = valid_sequence[-1]
            total_from_p = st.session_state.transition_counts['PP'] + st.session_state.transition_counts['PB']
            total_from_b = st.session_state.transition_counts['BP'] + st.session_state.transition_counts['BB']
            if last_outcome == 'P' and total_from_p > 0:
                prob_p_to_p = st.session_state.transition_counts['PP'] / total_from_p
                prob_p_to_b = st.session_state.transition_counts['PB'] / total_from_p
                if prob_p_to_p > prob_p_to_b and prob_p_to_p > 0.5 and 'P' == predicted_outcome:
                    markov_selection = 'P'
                    markov_confidence = prob_p_to_p * 100
                elif prob_p_to_b > prob_p_to_p and prob_p_to_b > 0.5 and 'B' == predicted_outcome:
                    markov_selection = 'B'
                    markov_confidence = prob_p_to_b * 100
            elif last_outcome == 'B' and total_from_b > 0:
                prob_b_to_p = st.session_state.transition_counts['BP'] / total_from_b
                prob_b_to_b = st.session_state.transition_counts['BB'] / total_from_b
                if prob_b_to_p > prob_b_to_b and prob_b_to_p > 0.5 and 'P' == predicted_outcome:
                    markov_selection = 'P'
                    markov_confidence = prob_b_to_p * 100
                elif prob_b_to_b > prob_b_to_b and prob_b_to_b > 0.5 and 'B' == predicted_outcome:
                    markov_selection = 'B'
                    markov_confidence = prob_b_to_b * 100

            is_streak = len(valid_sequence) >= 4 and len(set(valid_sequence[-4:])) == 1
            confidence_threshold = 45 if is_streak else 50
            smart_skip_threshold = 55 if is_streak else 60

            strategy_used = None
            bet_selection = None
            confidence = 0.0
            if (model_confidence > confidence_threshold or (is_streak and markov_confidence > 60)) and (lb6_selection or markov_selection):
                bet_selection = predicted_outcome
                confidence = max(model_confidence, lb6_confidence, markov_confidence)
                if is_streak and markov_selection:
                    strategy_used = 'Model+Markov(Streak)'
                elif lb6_selection and markov_selection:
                    strategy_used = 'Model+LB6+Markov'
                elif lb6_selection:
                    strategy_used = 'Model+LB6'
                elif markov_selection:
                    strategy_used = 'Model+Markov'

            if bet_selection and not (st.session_state.smart_skip_enabled and confidence < smart_skip_threshold):
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
                    st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%)"
                else:
                    st.session_state.pending_bet = None
                    st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (low confidence or no matching strategy: Model {model_confidence:.1f}% ({predicted_outcome}), LB6 {lb6_confidence:.1f}% ({sixth_prior}), Markov {markov_confidence:.1f}% ({markov_selection if markov_selection else 'N/A'})"

        if st.session_state.pending_bet:
            bet_amount, bet_selection = st.session_state.pending_bet
            st.session_state.bet_history.append({
                "Hand": len(st.session_state.sequence),
                "Bet": bet_amount,
                "Selection": bet_selection,
                "Result": result,
                "Outcome": "Pending",
                "Bankroll": st.session_state.bankroll
            })
            st.session_state.bets_placed += 1
            st.session_state.bankroll -= bet_amount
            if result == bet_selection:
                st.session_state.bankroll += bet_amount * (1.95 if bet_selection == 'B' else 2.0)
                st.session_state.bets_won += 1
                st.session_state.bet_history[-1]["Outcome"] = "Win"
                if st.session_state.money_management == 'T3':
                    st.session_state.t3_results.append('W')
                    if len(st.session_state.t3_results) >= 2 and st.session_state.t3_results[-2:] == ['W', 'W'] and st.session_state.t3_level < len(T3_LEVELS):
                        st.session_state.t3_level += 1
                    elif len(st.session_state.t3_results) >= 2 and st.session_state.t3_results[-2:] == ['L', 'W']:
                        st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                elif st.session_state.money_management == 'Parlay16':
                    st.session_state.parlay_wins += 1
                    st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                    st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, st.session_state.parlay_step)
                    if st.session_state.parlay_step > 1:
                        st.session_state.parlay_step_changes += 1
                elif st.session_state.money_management == 'Moon':
                    st.session_state.moon_level = min(st.session_state.moon_level + 1, len(MOON_LEVELS))
                    st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                    if st.session_state.moon_level > 1:
                        st.session_state.moon_level_changes += 1
                elif st.session_state.money_management == 'FourTier':
                    st.session_state.four_tier_step = min(st.session_state.four_tier_step + 1, 4)
                    if st.session_state.four_tier_step == 1:
                        st.session_state.four_tier_level = min(st.session_state.four_tier_level + 1, len(FOUR_TIER_LEVELS))
                    st.session_state.four_tier_losses = 0
                elif st.session_state.money_management == 'FlatbetLevelUp':
                    st.session_state.flatbet_levelup_net_loss = max(0, st.session_state.flatbet_levelup_net_loss - bet_amount)
                    if st.session_state.flatbet_levelup_net_loss < st.session_state.base_bet * FLATBET_LEVELUP_LEVELS[st.session_state.flatbet_levelup_level - 1] * 2:
                        st.session_state.flatbet_levelup_level = min(st.session_state.flatbet_levelup_level + 1, len(FLATBET_LEVELUP_LEVELS))
            else:
                st.session_state.bet_history[-1]["Outcome"] = "Loss"
                if st.session_state.money_management == 'T3':
                    st.session_state.t3_results.append('L')
                    if len(st.session_state.t3_results) >= 2 and st.session_state.t3_results[-2:] == ['L', 'L']:
                        st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                elif st.session_state.money_management == 'Parlay16':
                    st.session_state.parlay_step = 1
                    st.session_state.parlay_step_changes += 1
                elif st.session_state.money_management == 'Moon':
                    st.session_state.moon_level = max(1, st.session_state.moon_level - 1)
                    if st.session_state.moon_level < st.session_state.moon_peak_level:
                        st.session_state.moon_level_changes += 1
                elif st.session_state.money_management == 'FourTier':
                    st.session_state.four_tier_losses += 1
                    if st.session_state.four_tier_losses >= 2:
                        st.session_state.four_tier_level = max(1, st.session_state.four_tier_level - 1)
                        st.session_state.four_tier_step = 1
                        st.session_state.four_tier_losses = 0
                elif st.session_state.money_management == 'FlatbetLevelUp':
                    st.session_state.flatbet_levelup_net_loss += bet_amount
                    if st.session_state.flatbet_levelup_net_loss > st.session_state.base_bet * FLATBET_LEVELUP_LEVELS[st.session_state.flatbet_levelup_level - 1] * 2:
                        st.session_state.flatbet_levelup_level = max(1, st.session_state.flatbet_levelup_level - 1)

        stop_loss = st.session_state.initial_bankroll * (1 - st.session_state.stop_loss_percentage)
        target_profit = None
        if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0:
            target_profit = st.session_state.initial_bankroll * (1 + st.session_state.target_profit_percentage)
        elif st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0:
            target_profit = st.session_state.initial_bankroll + st.session_state.target_profit_units

        if st.session_state.bankroll <= stop_loss:
            st.session_state.shoe_completed = True
            st.session_state.advice = f"Stop loss reached. Bankroll: ${st.session_state.bankroll:.2f}"
            st.session_state.pending_bet = None
        elif target_profit and st.session_state.bankroll >= target_profit:
            st.session_state.shoe_completed = True
            st.session_state.advice = f"Target profit reached. Bankroll: ${st.session_state.bankroll:.2f}"
            st.session_state.pending_bet = None
        elif len(st.session_state.sequence) >= 60:
            st.session_state.shoe_completed = True
            if st.session_state.safety_net_enabled:
                st.session_state.advice = f"Shoe completed. Using Safety Net (Flatbet)."
            else:
                st.session_state.advice = f"Shoe completed. Reset to start a new session."
                st.session_state.pending_bet = None
    except Exception as e:
        st.error(f"Error processing result: {str(e)}")

def render_session_setup():
    with st.expander("Session Setup", expanded=st.session_state.bankroll == 0):
        try:
            bankroll = st.number_input("Bankroll ($)", min_value=0.0, step=10.0, value=1000.0)
            base_bet = st.number_input("Base Bet ($)", min_value=0.0, step=1.0, value=10.0)
            stop_loss_percentage = st.slider("Stop Loss (% of Bankroll)", 0, 100, 50) / 100
            target_profit_option = st.selectbox("Target Profit", ['None', 'Profit %', 'Units'])
            target_profit_percentage = 0
            target_profit_units = 0
            if target_profit_option == 'Profit %':
                target_profit_percentage = st.slider("Target Profit (% of Bankroll)", 0, 100, 50) / 100
            elif target_profit_option == 'Units':
                target_profit_units = st.number_input("Target Profit (Units)", min_value=0.0, step=10.0, value=100.0)
            safety_net_enabled = st.checkbox("Enable Safety Net", value=False)
            smart_skip_enabled = st.checkbox("Enable Smart Skip", value=False)
            money_management = st.selectbox("Money Management Strategy", ['Flatbet', 'T3', 'Parlay16', 'Moon', 'FourTier', 'FlatbetLevelUp'])
            if st.button("Start Session"):
                st.session_state.update({
                    'bankroll': bankroll,
                    'initial_bankroll': bankroll,
                    'base_bet': base_bet,
                    'stop_loss_percentage': stop_loss_percentage,
                    'target_profit_option': target_profit_option,
                    'target_profit_percentage': target_profit_percentage,
                    'target_profit_units': target_profit_units,
                    'safety_net_enabled': safety_net_enabled,
                    'smart_skip_enabled': smart_skip_enabled,
                    'money_management': money_management,
                    'shoe_completed': False,
                    'sequence': [],
                    'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
                    'pending_bet': None,
                    'bet_history': [],
                    'bets_placed': 0,
                    'bets_won': 0,
                    't3_level': 1,
                    't3_results': [],
                    'parlay_step': 1,
                    'parlay_wins': 0,
                    'parlay_peak_step': 1,
                    'parlay_step_changes': 0,
                    'moon_level': 1,
                    'moon_peak_level': 1,
                    'moon_level_changes': 0,
                    'four_tier_level': 1,
                    'four_tier_step': 1,
                    'four_tier_losses': 0,
                    'flatbet_levelup_level': 1,
                    'flatbet_levelup_net_loss': 0,
                    'advice': "Start placing results to get advice."
                })
                st.session_state.model, st.session_state.le = train_model()
                st.success("Session started!")
            if st.session_state.bankroll > 0 and st.button("Reset Session"):
                reset_session()
                st.success("Session reset!")
        except Exception as e:
            st.error(f"Error setting up session: {str(e)}")

def render_result_input():
    with st.expander("Result Input", expanded=True):
        try:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Player (P)"):
                    place_result('P')
            with col2:
                if st.button("Banker (B)"):
                    place_result('B')
            with col3:
                if st.button("Tie (T)"):
                    place_result('T')
            with col4:
                if st.button("Undo Last"):
                    if st.session_state.sequence:
                        last_result = st.session_state.sequence.pop()
                        if st.session_state.bet_history and st.session_state.bet_history[-1]["Hand"] == len(st.session_state.sequence) + 1:
                            last_bet = st.session_state.bet_history.pop()
                            st.session_state.bankroll = last_bet["Bankroll"]
                            if last_bet["Outcome"] == "Win":
                                st.session_state.bets_won -= 1
                            st.session_state.bets_placed -= 1
                            if st.session_state.money_management == 'T3' and st.session_state.t3_results:
                                st.session_state.t3_results.pop()
                                st.session_state.t3_level = max(1, min(len(T3_LEVELS), st.session_state.t3_level))
                            elif st.session_state.money_management == 'Parlay16':
                                st.session_state.parlay_step = max(1, st.session_state.parlay_step - 1)
                                st.session_state.parlay_wins = max(0, st.session_state.parlay_wins - 1)
                            elif st.session_state.money_management == 'Moon':
                                st.session_state.moon_level = max(1, st.session_state.moon_level - 1)
                            elif st.session_state.money_management == 'FourTier':
                                st.session_state.four_tier_step = max(1, st.session_state.four_tier_step - 1)
                                st.session_state.four_tier_losses = max(0, st.session_state.four_tier_losses - 1)
                            elif st.session_state.money_management == 'FlatbetLevelUp':
                                st.session_state.flatbet_levelup_net_loss = max(0, st.session_state.flatbet_levelup_net_loss - last_bet["Bet"])
                        if len(st.session_state.sequence) >= 1 and last_bet["Result"] in ['P', 'B'] and st.session_state.sequence[-1] in ['P', 'B']:
                            transition = f"{st.session_state.sequence[-1]}{last_bet['Result']}"
                            st.session_state.transition_counts[transition] = max(0, st.session_state.transition_counts[transition] - 1)
                        st.session_state.shoe_completed = False
                        st.session_state.advice = "Last result undone. Enter next result."
                        st.session_state.pending_bet = None
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")

def render_prediction():
    with st.expander("Prediction", expanded=True):
        try:
            if st.session_state.bankroll == 0:
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
                    f"Advice: {advice}</p></div>",
                    unsafe_allow_html=True
                )
                if 'Skip betting' in advice:
                    valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
                    if len(valid_sequence) >= SEQUENCE_LENGTH:
                        prediction_sequence = valid_sequence[-SEQUENCE_LENGTH:]
                        encoded_input = st.session_state.le.transform(prediction_sequence)
                        input_array = np.array([encoded_input])
                        prediction_probs = st.session_state.model.predict_proba(input_array)[0]
                        predicted_class = np.argmax(prediction_probs)
                        predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
                        model_confidence = np.max(prediction_probs) * 100
                        lb6_confidence = 0
                        sixth_prior = 'N/A'
                        if len(valid_sequence) >= 6:
                            sixth_prior = valid_sequence[-6]
                            outcome_index = st.session_state.le.transform([sixth_prior])[0]
                            lb6_confidence = prediction_probs[outcome_index] * 100
                        markov_confidence = 0
                        markov_selection = 'N/A'
                        last_outcome = valid_sequence[-1]
                        total_from_p = st.session_state.transition_counts['PP'] + st.session_state.transition_counts['PB']
                        total_from_b = st.session_state.transition_counts['BP'] + st.session_state.transition_counts['BB']
                        if last_outcome == 'P' and total_from_p > 0:
                            prob_p_to_p = st.session_state.transition_counts['PP'] / total_from_p
                            prob_p_to_b = st.session_state.transition_counts['PB'] / total_from_p
                            if prob_p_to_p > prob_p_to_b and prob_p_to_p > 0.5:
                                markov_selection = 'P'
                                markov_confidence = prob_p_to_p * 100
                            elif prob_p_to_b > prob_p_to_p and prob_p_to_b > 0.5:
                                markov_selection = 'B'
                                markov_confidence = prob_p_to_b * 100
                        elif last_outcome == 'B' and total_from_b > 0:
                            prob_b_to_p = st.session_state.transition_counts['BP'] / total_from_b
                            prob_b_to_b = st.session_state.transition_counts['BB'] / total_from_b
                            if prob_b_to_p > prob_b_to_b and prob_b_to_p > 0.5:
                                markov_selection = 'P'
                                markov_confidence = prob_b_to_p * 100
                            elif prob_b_to_b > prob_b_to_b and prob_b_to_b > 0.5:
                                markov_selection = 'B'
                                markov_confidence = prob_b_to_b * 100
                        is_streak = len(valid_sequence) >= 4 and len(set(valid_sequence[-4:])) == 1
                        st.markdown(
                            f"**Debug Info**: Model: {model_confidence:.1f}% ({predicted_outcome}), "
                            f"LB6: {lb6_confidence:.1f}% ({sixth_prior}), "
                            f"Markov: {markov_confidence:.1f}% ({markov_selection}), "
                            f"Transitions: {st.session_state.transition_counts}, "
                            f"Streak: {'Yes' if is_streak else 'No'}"
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
            with col2:
                st.markdown(f"**Safety Net**: {'On' if st.session_state.safety_net_enabled else 'Off'}")
                st.markdown(f"**Hands Played**: {len(st.session_state.sequence)}")
                valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
                streak_info = "No streak detected"
                if len(valid_sequence) >= 4 and len(set(valid_sequence[-4:])) == 1:
                    streak_length = len([x for x in valid_sequence[::-1] if x == valid_sequence[-1]])
                    streak_info = f"Streak detected: {valid_sequence[-1]} x {streak_length}"
                st.markdown(f"**Streak Status**: {streak_info}")
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
                st.markdown(strategy_status, unsafe_allow_html=True)
                st.markdown(f"**Bets Placed**: {st.session_state.bets_placed}")
                st.markdown(f"**Bets Won**: {st.session_state.bets_won}")
                st.markdown(f"**Online Users**: {track_user_session()}")
        except Exception as e:
            st.error(f"Error rendering status: {str(e)}")

def render_bead_plate():
    with st.expander("Bead Plate"):
        try:
            if not st.session_state.sequence:
                st.info("No results yet. Enter results to see the bead plate.")
                return

            # Define bead plate grid (6 rows, dynamic columns)
            rows = 6
            results = st.session_state.sequence
            num_results = len(results)
            cols = (num_results + rows - 1) // rows  # Ceiling division

            # Initialize grid
            grid = [['' for _ in range(cols)] for _ in range(rows)]
            for i, result in enumerate(results):
                row = i % rows
                col = i // rows
                grid[row][col] = result

            # Generate HTML for bead plate
            html = """
            <style>
                .bead-plate-table {
                    border-collapse: collapse;
                    margin: 10px 0;
                }
                .bead-plate-table td {
                    width: 30px;
                    height: 30px;
                    text-align: center;
                    vertical-align: middle;
                    border: 1px solid #ccc;
                    font-weight: bold;
                    font-size: 14px;
                }
                .player {
                    background-color: #3182ce;
                    color: white;
                    border-radius: 50%;
                }
                .banker {
                    background-color: #e53e3e;
                    color: white;
                    border-radius: 50%;
                }
                .tie {
                    background-color: #38a169;
                    color: white;
                    border-radius: 50%;
                }
                .empty {
                    background-color: #f7fafc;
                }
            </style>
            <table class='bead-plate-table'>
            """

            for row in range(rows):
                html += "<tr>"
                for col in range(cols):
                    result = grid[row][col]
                    if result == 'P':
                        html += "<td class='player'>P</td>"
                    elif result == 'B':
                        html += "<td class='banker'>B</td>"
                    elif result == 'T':
                        html += "<td class='tie'>T</td>"
                    else:
                        html += "<td class='empty'></td>"
                html += "</tr>"
            html += "</table>"

            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering bead plate: {str(e)}")

def render_history():
    with st.expander("Bet History"):
        try:
            if st.session_state.bet_history:
                df = pd.DataFrame(st.session_state.bet_history)
                df = df[['Hand', 'Bet', 'Selection', 'Result', 'Outcome', 'Bankroll']]
                df.columns = ['Hand', 'Bet ($)', 'Selection', 'Result', 'Outcome', 'Bankroll ($)']
                st.dataframe(df)
            else:
                st.info("No bets placed yet.")
        except Exception as e:
            st.error(f"Error rendering history: {str(e)}")

def track_user_session():
    return 1  # Placeholder for user tracking

def main():
    st.set_page_config(page_title="Baccarat Betting Assistant", layout="wide")
    st.title("Baccarat Betting Assistant")
    initialize_session_state()
    render_session_setup()
    render_result_input()
    render_prediction()
    render_status()
    render_bead_plate()
    render_history()

if __name__ == "__main__":
    main()
