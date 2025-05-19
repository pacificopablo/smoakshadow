import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import uuid
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Constants
SESSION_FILE = "baccarat_sessions_ml.json"
DEFAULT_BANKROLL = 500.0
DEFAULT_BASE_BET = 10.0
LOSS_LIMIT = 0.5  # Stop at 50% bankroll loss
WIN_LIMIT = 1.0    # Stop at 100% bankroll gain
SAFETY_NET_LOSSES = 3  # Pause betting after 3 consecutive losses
LOOKBACK = 5  # Use last 5 outcomes for ML prediction

# Initialize ML model
def init_ml_model():
    # Simulate a small training dataset (in practice, use real or larger simulated data)
    np.random.seed(42)
    outcomes = ['Banker', 'Player', 'Tie']
    probs = [0.4586, 0.4462, 0.0952]
    simulated_data = np.random.choice(outcomes, size=1000, p=probs)
    
    # Create features: last LOOKBACK outcomes
    X = []
    y = []
    for i in range(LOOKBACK, len(simulated_data)):
        X.append(simulated_data[i-LOOKBACK:i])
        y.append(simulated_data[i])
    
    # Encode features and target
    le = LabelEncoder()
    le.fit(outcomes)
    X_encoded = np.array([le.transform(seq) for seq in X])
    y_encoded = le.transform(y)
    
    # Train model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_encoded, y_encoded)
    return model, le

# Initialize session state
def initialize_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.bankroll = DEFAULT_BANKROLL
        st.session_state.base_bet = DEFAULT_BASE_BET
        st.session_state.results = []
        st.session_state.consecutive_losses = 0
        st.session_state.session_active = True
        st.session_state.total_bets = 0
        st.session_state.wins = 0
        st.session_state.losses = 0
        st.session_state.ties = 0
        st.session_state.model, st.session_state.le = init_ml_model()

# Save session data
def save_session():
    session_data = {
        'session_id': st.session_state.session_id,
        'bankroll': st.session_state.bankroll,
        'results': st.session_state.results,
        'total_bets': st.session_state.total_bets,
        'wins': st.session_state.wins,
        'losses': st.session_state.losses,
        'ties': st.session_state.ties
    }
    try:
        with open(SESSION_FILE, 'w') as f:
            import json
            json.dump(session_data, f)
    except Exception as e:
        st.error(f"Error saving session: {e}")

# Load session data
def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                import json
                data = json.load(f)
                st.session_state.session_id = data['session_id']
                st.session_state.bankroll = data['bankroll']
                st.session_state.results = data['results']
                st.session_state.total_bets = data['total_bets']
                st.session_state.wins = data['wins']
                st.session_state.losses = data['losses']
                st.session_state.ties = data['ties']
                st.session_state.model, st.session_state.le = init_ml_model()
        except Exception as e:
            st.error(f"Error loading session: {e}")
            initialize_session()
    else:
        initialize_session()

# Predict next outcome using ML
def predict_next_outcome():
    if len(st.session_state.results) < LOOKBACK:
        return "Banker", [0.4586, 0.4462, 0.0952]  # Default to Banker with standard probs
    recent_outcomes = [r['outcome'] for r in st.session_state.results[-LOOKBACK:]]
    X = st.session_state.le.transform(recent_outcomes).reshape(1, -1)
    probs = st.session_state.model.predict_proba(X)[0]
    predicted_idx = np.argmax(probs)
    predicted_outcome = st.session_state.le.inverse_transform([predicted_idx])[0]
    return predicted_outcome, probs

# Calculate bet amount with safety net
def calculate_bet():
    if st.session_state.consecutive_losses >= SAFETY_NET_LOSSES:
        return 0.0, "Paused due to consecutive losses", "None"
    if st.session_state.bankroll <= st.session_state.base_bet:
        return 0.0, "Insufficient bankroll", "None"
    if st.session_state.bankroll <= DEFAULT_BANKROLL * (1 - LOSS_LIMIT):
        return 0.0, "Loss limit reached", "None"
    if st.session_state.bankroll >= DEFAULT_BANKROLL * (1 + WIN_LIMIT):
        return 0.0, "Win limit reached", "None"
    predicted_outcome, probs = predict_next_outcome()
    max_prob = max(probs)
    if predicted_outcome == "Tie" and max_prob > 0.5:
        return 0.0, "Skip Tie bet (high house edge)", "None"
    return st.session_state.base_bet, f"Bet on {predicted_outcome}", predicted_outcome

# Process game outcome
def process_outcome(outcome, bet_amount, bet_selection):
    if not st.session_state.session_active or bet_amount == 0:
        return
    st.session_state.total_bets += 1
    result = {
        'hand': st.session_state.total_bets,
        'outcome': outcome,
        'bet_amount': bet_amount,
        'bet_selection': bet_selection,
        'bankroll_before': st.session_state.bankroll
    }
    if outcome == bet_selection:
        if outcome == "Banker":
            st.session_state.bankroll += bet_amount * 0.95  # 5% commission
            st.session_state.wins += 1
            st.session_state.consecutive_losses = 0
        elif outcome == "Player":
            st.session_state.bankroll += bet_amount
            st.session_state.wins += 1
            st.session_state.consecutive_losses = 0
    elif outcome == "Tie":
        st.session_state.ties += 1
        st.session_state.consecutive_losses = 0
    else:
        st.session_state.bankroll -= bet_amount
        st.session_state.losses += 1
        st.session_state.consecutive_losses += 1
    result['bankroll_after'] = st.session_state.bankroll
    st.session_state.results.append(result)
    save_session()

# Display bead plate
def display_bead_plate():
    if not st.session_state.results:
        st.write("No results to display.")
        return
    df = pd.DataFrame(st.session_state.results)
    outcomes = df['outcome'].tolist()
    bead_display = []
    for i, outcome in enumerate(outcomes, 1):
        color = 'red' if outcome == 'Banker' else 'blue' if outcome == 'Player' else 'green'
        bead_display.append(f'<span style="color: {color}; font-size: 20px;">●</span>')
    st.markdown("### Bead Plate")
    st.markdown(" ".join(bead_display), unsafe_allow_html=True)

# Display statistics
def display_stats():
    if not st.session_state.results:
        return
    df = pd.DataFrame(st.session_state.results)
    st.markdown("### Session Statistics")
    st.write(f"**Total Hands**: {st.session_state.total_bets}")
    st.write(f"**Wins**: {st.session_state.wins}")
    st.write(f"**Losses**: {st.session_state.losses}")
    st.write(f"**Ties**: {st.session_state.ties}")
    win_rate = st.session_state.wins / st.session_state.total_bets * 100 if st.session_state.total_bets > 0 else 0
    st.write(f"**Win Rate**: {win_rate:.2f}%")
    net_profit = st.session_state.bankroll - DEFAULT_BANKROLL
    st.write(f"**Net Profit/Loss**: ${net_profit:.2f}")
    
    # Bankroll trend plot
    df['hand'] = df['hand'].astype(int)
    fig = px.line(df, x='hand', y='bankroll_after', title='Bankroll Trend', labels={'bankroll_after': 'Bankroll ($)', 'hand': 'Hand Number'})
    st.plotly_chart(fig)

# Main app
def main():
    st.set_page_config(page_title="Baccarat Tracker with ML", layout="wide")
    st.title("Baccarat Tracker with Machine Learning Bet Selection")
    
    # Load or initialize session
    load_session()
    
    # Session setup
    st.sidebar.header("Session Setup")
    bankroll = st.sidebar.number_input("Initial Bankroll", min_value=100.0, value=DEFAULT_BANKROLL, step=100.0)
    base_bet = st.sidebar.number_input("Base Bet", min_value=5.0, value=DEFAULT_BASE_BET, step=5.0)
    if st.sidebar.button("Start/Reset Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.bankroll = bankroll
        st.session_state.base_bet = base_bet
        st.session_state.results = []
        st.session_state.consecutive_losses = 0
        st.session_state.session_active = True
        st.session_state.total_bets = 0
        st.session_state.wins = 0
        st.session_state.losses = 0
        st.session_state.ties = 0
        st.session_state.model, st.session_state.le = init_ml_model()
        save_session()
    
    # Display current status
    st.markdown(f"**Current Bankroll**: ${st.session_state.bankroll:.2f}")
    bet_amount, bet_status, bet_selection = calculate_bet()
    st.markdown(f"**Recommended Bet**: {bet_status} (${bet_amount:.2f})")
    if bet_selection != "None":
        predicted_outcome, probs = predict_next_outcome()
        st.markdown(f"**ML Prediction Probabilities**: Banker {probs[0]:.2%}, Player {probs[1]:.2%}, Tie {probs[2]:.2%}")
    
    # Input game outcome
    st.header("Record Outcome")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Banker Win") and bet_amount > 0 and st.session_state.session_active:
            process_outcome("Banker", bet_amount, bet_selection)
    with col2:
        if st.button("Player Win") and bet_amount > 0 and st.session_state.session_active:
            process_outcome("Player", bet_amount, bet_selection)
    with col3:
        if st.button("Tie") and bet_amount > 0 and st.session_state.session_active:
            process_outcome("Tie", bet_amount, bet_selection)
    
    # Display results
    display_bead_plate()
    display_stats()
    
    # End session
    if st.button("End Session"):
        st.session_state.session_active = False
        st.warning("Session ended. Start a new session to continue.")

if __name__ == "__main__":
    main()
