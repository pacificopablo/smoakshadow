import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random

# Simulate Baccarat game data for training
def generate_baccarat_data(num_games=10000):
    outcomes = ['P', 'B']  # Player or Banker
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

# Initialize session state
def initialize_session_state():
    defaults = {
        'sequence_length': 10,
        'user_sequence': [],
        'bet_history': [],  # Tracks (result, bet_amount, bet_selection, bet_outcome, t3_level, t3_results)
        'pending_bet': None,  # Stores (bet_amount, bet_selection)
        'bankroll': 0.0,
        'base_bet': 0.0,
        'initial_bankroll': 0.0,
        'stop_loss': 0.8,  # Stop at 80%
        'win_limit': 1.5,  # Stop at 150%
        'bets_placed': 0,
        'bets_won': 0,
        'model': None,
        'le': None,
        't3_level': 1,
        't3_results': [],
        'session_active': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Main Streamlit app
def main():
    st.set_page_config(page_title="Grok vs. Baccarat", layout="centered")
    st.title("EDMELG BACCARAT")
    initialize_session_state()

    # Session setup
    st.subheader("Session Setup")
    bankroll_input = st.text_input("Enter Initial Bankroll ($):", key="bankroll_input")
    base_bet_input = st.text_input("Enter Base Bet ($):", key="base_bet_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Session", key="start_session"):
            try:
                bankroll = float(bankroll_input)
                base_bet = float(base_bet_input)
                if bankroll <= 0 or base_bet <= 0:
                    st.error("Bankroll and bet must be positive numbers.")
                    return
                if base_bet > bankroll * 0.05:
                    st.error("Base bet cannot exceed 5% of bankroll.")
                    return
                st.session_state.bankroll = bankroll
                st.session_state.base_bet = base_bet
                st.session_state.initial_bankroll = bankroll
                st.session_state.user_sequence = []
                st.session_state.bet_history = []
                st.session_state.pending_bet = None
                st.session_state.bets_placed = 0
                st.session_state.bets_won = 0
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.session_state.session_active = True

                # Train model
                outcomes = generate_baccarat_data()
                X, y, st.session_state.le = prepare_data(outcomes, st.session_state.sequence_length)
                st.session_state.model = RandomForestClassifier(n_estimators=100, random_state=42)
                st.session_state.model.fit(X, y)

                st.success(f"Session Started! Bankroll: ${bankroll:.2f}, Base Bet: ${base_bet:.2f}")
            except ValueError:
                st.error("Please enter valid numbers for bankroll and bet.")

    with col2:
        if st.button("Reset Session", key="reset_session"):
            st.session_state.bankroll = 0.0
            st.session_state.base_bet = 0.0
            st.session_state.initial_bankroll = 0.0
            st.session_state.user_sequence = []
            st.session_state.bet_history = []
            st.session_state.pending_bet = None
            st.session_state.bets_placed = 0
            st.session_state.bets_won = 0
            st.session_state.t3_level = 1
            st.session_state.t3_results = []
            st.session_state.model = None
            st.session_state.le = None
            st.session_state.session_active = False
            st.success("Session Reset. Enter new bankroll and bet to start.")

    # Result input
    if st.session_state.session_active:
        st.subheader("Enter Results")
        col3, col4, col5 = st.columns(3)
        with col3:
            if st.button("P (Player)", key="player_button"):
                add_result('P')
        with col4:
            if st.button("B (Banker)", key="banker_button"):
                add_result('B')
        with col5:
            if st.button("Undo Last Result", key="undo_button"):
                undo_result()

        # Display status
        st.subheader("Session Status")
        st.write(f"**Sequence**: {st.session_state.user_sequence if st.session_state.user_sequence else 'None'}")
        st.write(f"**Model Prediction**: Hidden")
        st.write(f"**Advice**: {get_advice()}")
        st.write(f"**T3 Status**: Level {st.session_state.t3_level}, Results: {st.session_state.t3_results}")
        st.write(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
        st.write(f"**Base Bet**: ${st.session_state.base_bet:.2f}")
        st.write(f"**Session**: {st.session_state.bets_placed} bets, {st.session_state.bets_won} wins")

def add_result(result):
    if st.session_state.model is None:
        st.error("Please start a session with bankroll and bet.")
        return
    if st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss:
        st.error(f"Bankroll below {st.session_state.stop_loss*100:.0f}%. Session ended. Reset or exit.")
        st.session_state.session_active = False
        return
    if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
        st.success(f"Bankroll above {st.session_state.win_limit*100:.0f}%. Session ended. Reset or exit.")
        st.session_state.session_active = False
        return

    # Resolve pending bet if exists
    bet_amount = 0
    bet_selection = None
    bet_outcome = None
    if st.session_state.pending_bet:
        bet_amount, bet_selection = st.session_state.pending_bet
        st.session_state.bets_placed += 1
        if result == bet_selection:
            if bet_selection == 'B':
                st.session_state.bankroll += bet_amount * 0.95
            else:  # Player
                st.session_state.bankroll += bet_amount
            st.session_state.bets_won += 1
            bet_outcome = 'win'
            # Modified T3 logic: Decrease level by 1 on first-step win, minimum 1
            if len(st.session_state.t3_results) == 0:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            st.session_state.t3_results.append('W')
        else:
            st.session_state.bankroll -= bet_amount
            bet_outcome = 'loss'
            st.session_state.t3_results.append('L')
        # Update T3 level after 3 results
        if len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins > losses:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses > wins:
                st.session_state.t3_level += 1
            st.session_state.t3_results = []
        st.session_state.pending_bet = None

    st.session_state.user_sequence.append(result)
    if len(st.session_state.user_sequence) > st.session_state.sequence_length:
        st.session_state.user_sequence.pop(0)

    # Make prediction and apply LB 6 Rep (Mirror) with T3
    if len(st.session_state.user_sequence) == st.session_state.sequence_length:
        encoded_input = st.session_state.le.transform(st.session_state.user_sequence)
        input_array = np.array([encoded_input])
        prediction_probs = st.session_state.model.predict_proba(input_array)[0]
        predicted_class = np.argmax(prediction_probs)
        predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction_probs) * 100

        # LB 6 Rep: Bet same as 6th prior result
        bet_selection = None
        if len(st.session_state.user_sequence) >= 6:
            sixth_prior = st.session_state.user_sequence[-6]
            outcome_index = st.session_state.le.transform([sixth_prior])[0]
            sixth_confidence = prediction_probs[outcome_index] * 100
            if sixth_confidence > 40:
                bet_selection = sixth_prior

        if bet_selection:
            bet_amount = st/session_state.base_bet * st.session_state.t3_level
            if bet_amount <= st.session_state.bankroll:
                st.session_state.pending_bet = (bet_amount, bet_selection)
            else:
                st.session_state.pending_bet = None
        else:
            st.session_state.pending_bet = None

    # Store bet history with T3 state
    st.session_state.bet_history.append((result, bet_amount, bet_selection, bet_outcome, st.session_state.t3_level, st.session_state.t3_results[:]))

def undo_result():
    if not st.session_state.user_sequence:
        st.warning("No results to undo.")
        return
    st.session_state.user_sequence.pop()
    if st.session_state.bet_history:
        last_bet = st.session_state.bet_history.pop()
        result, bet_amount, bet_selection, bet_outcome, t3_level, t3_results = last_bet
        if bet_amount > 0:
            st.session_state.bets_placed -= 1
            if bet_outcome == 'win':
                if

 bet_selection == 'B':
                    st.session_state.bankroll -= bet_amount * 0.95
                else:  # Player
                    st.session_state.bankroll -= bet_amount
                st.session_state.bets_won -= 1
            elif bet_outcome == 'loss':
                st.session_state.bankroll += bet_amount
            # Restore T3 state
            st.session_state.t3_level = t3_level
            st.session_state.t3_results = t3_results[:]
    if st.session_state.pending_bet and len(st.session_state.user_sequence) >= st.session_state.sequence_length - 1:
        st.session_state.pending_bet = None

def get_advice():
    if len(st.session_state.user_sequence) < st.session_state.sequence_length:
        return f"Need {st.session_state.sequence_length - len(st.session_state.user_sequence)} more results"
    elif st.session_state.pending_bet:
        bet_amount, bet_selection = st.session_state.pending_bet
        return f"Bet ${bet_amount:.2f} on {bet_selection} (T3 Level {st.session_state.t3_level}, mirroring 6th prior)."
    else:
        return "Skip betting (no 6th prior or low confidence)."

if __name__ == "__main__":
    main()
