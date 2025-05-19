import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import uuid
import os

# Constants
SESSION_FILE = "baccarat_sessions.json"
DEFAULT_BANKROLL = 500.0
DEFAULT_BASE_BET = 10.0
LOSS_LIMIT = 0.5  # Stop at 50% bankroll loss
WIN_LIMIT = 1.0    # Stop at 100% bankroll gain
SAFETY_NET_LOSSES = 3  # Pause betting after 3 consecutive losses

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
        except Exception as e:
            st.error(f"Error loading session: {e}")
            initialize_session()
    else:
        initialize_session()

# Calculate bet amount with safety net
def calculate_bet():
    if st.session_state.consecutive_losses >= SAFETY_NET_LOSSES:
        return 0.0, "Paused due to consecutive losses"
    if st.session_state.bankroll <= st.session_state.base_bet:
        return 0.0, "Insufficient bankroll"
    if st.session_state.bankroll <= DEFAULT_BANKROLL * (1 - LOSS_LIMIT):
        return 0.0, "Loss limit reached"
    if st.session_state.bankroll >= DEFAULT_BANKROLL * (1 + WIN_LIMIT):
        return 0.0, "Win limit reached"
    return st.session_state.base_bet, "Bet on Banker"

# Process game outcome
def process_outcome(outcome, bet_amount):
    if not st.session_state.session_active:
        return
    st.session_state.total_bets += 1
    result = {
        'hand': st.session_state.total_bets,
        'outcome': outcome,
        'bet_amount': bet_amount,
        'bankroll_before': st.session_state.bankroll
    }
    if outcome == "Banker":
        st.session_state.bankroll += bet_amount * 0.95  # 5% commission
        st.session_state.wins += 1
        st.session_state.consecutive_losses = 0
    elif outcome == "Player":
        st.session_state.bankroll -= bet_amount
        st.session_state.losses += 1
        st.session_state.consecutive_losses += 1
    else:  # Tie
        st.session_state.ties += 1
        st.session_state.consecutive_losses = 0
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
    st.write(f"**Wins (Banker)**: {st.session_state.wins}")
    st.write(f"**Losses (Player)**: {st.session_state.losses}")
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
    st.set_page_config(page_title="Baccarat Tracker", layout="wide")
    st.title("Baccarat Tracker with Money Management")
    
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
        save_session()
    
    # Display current status
    st.markdown(f"**Current Bankroll**: ${st.session_state.bankroll:.2f}")
    bet_amount, bet_status = calculate_bet()
    st.markdown(f"**Recommended Bet**: {bet_status} (${bet_amount:.2f})")
    
    # Input game outcome
    st.header("Record Outcome")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Banker Win") and bet_amount > 0 and st.session_state.session_active:
            process_outcome("Banker", bet_amount)
    with col2:
        if st.button("Player Win") and bet_amount > 0 and st.session_state.session_active:
            process_outcome("Player", bet_amount)
    with col3:
        if st.button("Tie") and bet_amount > 0 and st.session_state.session_active:
            process_outcome("Tie", bet_amount)
    
    # Display results
    display_bead_plate()
    display_stats()
    
    # End session
    if st.button("End Session"):
        st.session_state.session_active = False
        st.warning("Session ended. Start a new session to continue.")

if __name__ == "__main__":
    main()
