import streamlit as st
import pandas as pd
from datetime import date

st.title("Z 1003.1 Baccarat Tracker")

# --- Session Info ---
st.header("1. Session Info")
session_date = st.date_input("Date", date.today())
table_id = st.text_input("Table ID")
bankroll = st.number_input("Starting Bankroll", min_value=0)
base_bet = st.number_input("Base Bet", min_value=0)
trigger_found = st.checkbox("Trigger Found?")

# --- Trigger Observation ---
st.header("2. Trigger Observation (12 Hands)")
trigger_data = []
for i in range(1, 13):
    col1, col2 = st.columns([1, 3])
    with col1:
        result = st.text_input(f"Hand #{i} Result", key=f"res_{i}")
    with col2:
        notes = st.text_input(f"Hand #{i} Notes", key=f"note_{i}")
    trigger_data.append({"Hand": i, "Result": result, "Notes": notes})

# --- Betting Log ---
st.header("3. Betting Log")
if "bets" not in st.session_state:
    st.session_state.bets = []

bet_side = st.selectbox("Bet Side", ["", "Player", "Banker"])
bet_amount = st.number_input("Bet Amount", min_value=0)
bet_result = st.selectbox("Result", ["", "Win", "Loss"])

if st.button("Add Bet"):
    if bet_side and bet_result:
        st.session_state.bets.append({
            "Bet #": len(st.session_state.bets) + 1,
            "Side": bet_side,
            "Amount": bet_amount,
            "Result": bet_result
        })

if st.session_state.bets:
    bets_df = pd.DataFrame(st.session_state.bets)
    st.dataframe(bets_df)

# --- Summary ---
st.header("4. Round Summary")
if st.session_state.bets:
    wins = sum(1 for bet in st.session_state.bets if bet["Result"] == "Win")
    losses = sum(1 for bet in st.session_state.bets if bet["Result"] == "Loss")
    profit = sum(
        bet["Amount"] if bet["Result"] == "Win" else -bet["Amount"]
        for bet in st.session_state.bets
    )
    first_win = st.session_state.bets[0]["Result"] == "Win" if st.session_state.bets else False
    resets = losses // 3

    st.write(f"Total Bets: {len(st.session_state.bets)}")
    st.write(f"Total Wins: {wins}")
    st.write(f"Total Losses: {losses}")
    st.write(f"Total Profit: ${profit}")
    st.write(f"First Win: {'Yes' if first_win else 'No'}")
    st.write(f"Reset Count: {resets}")
