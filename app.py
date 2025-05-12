import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

st.set_page_config(page_title="Z 1003.1 Baccarat Tracker", layout="wide")
st.title("Z 1003.1 Baccarat Tracker")

# --- Session Initialization ---
if "bets" not in st.session_state:
    st.session_state.bets = []
if "profits" not in st.session_state:
    st.session_state.profits = [0]
if "t3_level" not in st.session_state:
    st.session_state.t3_level = 1
if "start_bankroll" not in st.session_state:
    st.session_state.start_bankroll = 0

# --- Session Info ---
st.header("1. Session Info")
session_date = st.date_input("Date", date.today())
table_id = st.text_input("Table ID")
start_bankroll = st.number_input("Starting Bankroll", value=st.session_state.start_bankroll, min_value=0)
base_bet = st.number_input("Base Bet", min_value=0)
trigger_found = st.checkbox("Trigger Found?")

# Update session bankroll
st.session_state.start_bankroll = start_bankroll
current_bankroll = st.session_state.start_bankroll + st.session_state.profits[-1]
st.markdown(f"**Current Bankroll:** ${current_bankroll}**")

# --- Trigger Observation ---
st.header("2. Trigger Observation (12 Hands)")
trigger_data = []
for i in range(1, 13):
    col1, col2 = st.columns([1, 3])
    with col1:
        result = st.text_input(f"Hand #{i} Result (P/B/T)", key=f"res_{i}")
    with col2:
        notes = st.text_input(f"Hand #{i} Notes", key=f"note_{i}")
    trigger_data.append({"Hand": i, "Result": result, "Notes": notes})

# --- Betting Log ---
st.header("3. Betting Log")

bet_side = st.selectbox("Your Bet Side", ["", "Player", "Banker"])
bet_amount = st.number_input("Bet Amount", min_value=0, step=100)
actual_result = st.selectbox("Actual Result", ["", "Player", "Banker", "Tie"])

if st.button("Add Bet"):
    if bet_side and actual_result:
        result = "Win" if bet_side == actual_result else "Loss"
        if actual_result == "Tie":
            result = "Loss"

        profit = bet_amount if result == "Win" else -bet_amount
        new_total = st.session_state.profits[-1] + profit

        st.session_state.bets.append({
            "Bet #": len(st.session_state.bets) + 1,
            "Bet Side": bet_side,
            "Amount": bet_amount,
            "Actual Result": actual_result,
            "Result": result,
            "Level": st.session_state.t3_level,
            "Profit": profit,
            "Bankroll After": st.session_state.start_bankroll + new_total
        })

        st.session_state.profits.append(new_total)

        # T3 Logic
        if len(st.session_state.bets) % 3 == 0:
            last_3 = st.session_state.bets[-3:]
            win_count = sum(1 for b in last_3 if b["Result"] == "Win")
            if win_count == 3:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
            elif win_count == 2:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif win_count == 1:
                st.session_state.t3_level += 1
            elif win_count == 0:
                st.session_state.t3_level += 2

# --- Bet History ---
if st.session_state.bets:
    st.subheader("Betting History")
    bets_df = pd.DataFrame(st.session_state.bets)
    st.dataframe(bets_df, use_container_width=True)

    # CSV Export
    csv = bets_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "baccarat_bets.csv", "text/csv")

# --- Summary & Chart ---
st.header("4. Round Summary")
if st.session_state.bets:
    wins = sum(1 for b in st.session_state.bets if b["Result"] == "Win")
    losses = sum(1 for b in st.session_state.bets if b["Result"] == "Loss")
    total_profit = st.session_state.profits[-1]
    first_win = st.session_state.bets[0]["Result"] == "Win"

    st.markdown(f"**Total Bets:** {len(st.session_state.bets)}")
    st.markdown(f"**Wins:** {wins} | **Losses:** {losses}")
    st.markdown(f"**Total Profit:** ${total_profit}")
    st.markdown(f"**Current T3 Level:** {st.session_state.t3_level}")
    st.markdown(f"**Bankroll After All Bets:** ${st.session_state.start_bankroll + total_profit}")

    # Chart
    st.subheader("Profit Over Time")
    fig, ax = plt.subplots()
    ax.plot(st.session_state.profits, marker='o')
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_xlabel("Bets")
    ax.set_title("Profit Progress")
    ax.axhline(0, color='gray', linestyle='--')
    st.pyplot(fig)
