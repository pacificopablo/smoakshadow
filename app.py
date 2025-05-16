import streamlit as st
import random
import pandas as pd

st.set_page_config(page_title="Baccarat Predictor", layout="wide")

# --- SESSION STATE INIT ---
if "history" not in st.session_state:
    st.session_state.history = []
if "bankroll" not in st.session_state:
    st.session_state.bankroll = [1000]  # Starting bankroll
if "t3_level" not in st.session_state:
    st.session_state.t3_level = 1
if "ai_enabled" not in st.session_state:
    st.session_state.ai_enabled = False

# --- CONSTANTS ---
HOUSE_EDGE = {"Banker": 0.0106, "Player": 0.0124, "Tie": 0.144}
BASE_BET = 10
STOP_WIN = 300
STOP_LOSS = -200

# --- T3 STAKING ---
def get_t3_bet():
    return BASE_BET * st.session_state.t3_level

def update_t3(wins):
    if wins == 3:
        st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
    elif wins == 2:
        st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
    elif wins == 1:
        pass
    else:
        st.session_state.t3_level += 2

# --- PREDICTION LOGIC ---
def ai_predict():
    """Simple AI prediction with 55% Banker bias."""
    return random.choices(["Banker", "Player"], weights=[0.55, 0.45])[0]

def predict_next_outcome():
    """Advanced predictor using historical patterns and Banker bias."""
    if len(st.session_state.history) < 3:
        return ai_predict()  # Fallback to simple AI if not enough history

    history = st.session_state.history
    last_results = [r["Game Result"] for r in history[-10:]]  # Last 10 results

    # Initialize weights
    banker_weight = 0.3 * 0.55  # 30% from simple AI's Banker bias
    player_weight = 0.3 * 0.45  # 30% from simple AI's Player bias

    # Pattern 1: Streak detection (3+ same results)
    if len(last_results) >= 3 and all(x == last_results[-1] for x in last_results[-3:]):
        if last_results[-1] == "Banker":
            banker_weight += 0.7 * 0.8  # 70% pattern weight, 80% to continue streak
            player_weight += 0.7 * 0.2
        elif last_results[-1] == "Player":
            player_weight += 0.7 * 0.8
            banker_weight += 0.7 * 0.2

    # Pattern 2: Alternation detection (e.g., B-P-B or P-B-P)
    elif len(last_results) >= 3 and last_results[-1] != last_results[-2] and last_results[-2] != last_results[-3]:
        if last_results[-1] == "Banker":
            player_weight += 0.7 * 0.6  # Slightly favor Player after alternation
            banker_weight += 07 * 0.4
        elif last_results[-1] == "Player":
            banker_weight += 0.7 * 0.6  # Slightly favor Banker after alternation
            player_weight += 0.7 * 0.4

    # Pattern 3: Recent bias (proportion in last 10 results)
    else:
        banker_count = last_results.count("Banker")
        player_count = last_results.count("Player")
        total = banker_count + player_count
        if total > 0:
            banker_weight += 0.7 * (banker_count / total)
            player_weight += 0.7 * (player_count / total)
        else:
            banker_weight += 0.7 * 0.5  # Neutral if no Banker/Player results
            player_weight += 0.7 * 0.5

    # Normalize weights and predict
    total_weight = banker_weight + player_weight
    if total_weight == 0:
        total_weight = 1  # Avoid division by zero
    banker_weight /= total_weight
    player_weight /= total_weight

    return random.choices(["Banker", "Player"], weights=[banker_weight, player_weight])[0]

# --- SIDEBAR OPTIONS ---
st.sidebar.title("Settings")
ai_toggle = st.sidebar.checkbox("Enable AI Prediction", value=st.session_state.ai_enabled)
t3_toggle = st.sidebar.checkbox("Use T3 Strategy", value=True)

st.session_state.ai_enabled = ai_toggle

# --- MAIN UI ---
st.title("Baccarat Predictor — Optimized for Profit per Shoe")

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if ai_toggle:
        selected_bet = predict_next_outcome()
        st.info(f"AI predicts: **{selected_bet}**")
        st.caption("Prediction based on historical patterns and 55% Banker bias.")
    else:
        selected_bet = st.radio("Your Bet:", ["Banker", "Player", "Tie"])

with col2:
    actual_result = st.radio("Actual Game Result:", ["Banker", "Player", "Tie"])

with col3:
    bet_amount = get_t3_bet() if t3_toggle else BASE_BET
    st.metric("Current Bet", f"${bet_amount}")

# --- PLACE BET BUTTON ---
if st.button("Place Bet"):
    win = selected_bet == actual_result
    net = bet_amount if win else -bet_amount
    new_balance = st.session_state.bankroll[-1] + net
    st.session_state.bankroll.append(new_balance)

    st.session_state.history.append({
        "Bet On": selected_bet,
        "Game Result": actual_result,
        "Result": "Win" if win else "Loss",
        "Wager": bet_amount,
        "Net": net,
        "Balance": new_balance
    })

    # Update T3 logic
    if t3_toggle:
        round_idx = (len(st.session_state.history) - 1) % 3
        if round_idx == 2:
            round_results = st.session_state.history[-3:]
            wins = sum(1 for r in round_results if r["Result"] == "Win")
            update_t3(wins)

    # Banker streak alert
    if len(st.session_state.history) >= 3:
        last3 = [r["Game Result"] for r in st.session_state.history[-3:]]
        if all(x == "Banker" for x in last3):
            st.info("**Banker streak detected (3+). Consider riding it!**")

    # Auto stop: profit or loss reached
    profit = new_balance - st.session_state.bankroll[0]
    if profit >= STOP_WIN:
        st.success(f"Profit target reached: +${profit}. Stop betting this shoe!")
    elif profit <= STOP_LOSS:
        st.error(f"Loss limit hit: ${profit}. Stop and reassess.")

# --- METRICS ---
df = pd.DataFrame(st.session_state.history)
total_bets = len(df)
wins = df[df["Result"] == "Win"].shape[0] if total_bets > 0 else 0
user_win_rate = wins / total_bets if total_bets > 0 else 0
avg_house_edge = df["Bet On"].map(HOUSE_EDGE).mean() if total_bets > 0 else 0
user_edge = user_win_rate - (1 - avg_house_edge)

st.subheader("Performance Metrics")
col4, col5, col6 = st.columns(3)
col4.metric("Total Bets", total_bets)
col5.metric("Win Rate", f"{user_win_rate:.2%}")
col6.metric("User Edge vs House", f"{user_edge:.2%}")

# --- BANKROLL CHART ---
st.subheader("Bankroll Over Time")
st.line_chart(st.session_state.bankroll)

# --- EDGE OVER TIME CHART ---
if total_bets >= 3:
    edge_df = pd.DataFrame({
        "Bet #": range(1, total_bets + 1),
        "User Edge": [
            (df.iloc[:i+1]["Result"].value_counts().get("Win", 0) / (i+1)) - 
            (1 - df.iloc[:i+1]["Bet On"].map(HOUSE_EDGE).mean())
            for i in range(total_bets)
        ]
    })
    st.subheader("Edge Over Time")
    st.line_chart(edge_df.set_index("Bet #"))

# --- HISTORY TABLE ---
st.subheader("Bet History (Latest 20)")
if not df.empty and all(col in df.columns for col in ["Bet On", "Game Result", "Result", "Wager", "Net", "Balance"]):
    st.dataframe(df[["Bet On", "Game Result", "Result", "Wager", "Net", "Balance"]].tail(20), use_container_width=True)
else:
    st.write("No bet history available yet.")
