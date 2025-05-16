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
# Baccarat outcome probabilities (standard 8-deck, no commission considered)
OUTCOME_PROBS = {"Banker": 0.4586, "Player": 0.4462, "Tie": 0.0952}

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
            banker_weight += 0.7 * 0.4
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

# --- RANDOM GAME RESULT ---
def get_random_result():
    """Generate random game result based on Baccarat probabilities."""
    return random.choices(
        ["Banker", "Player", "Tie"],
        weights=[OUTCOME_PROBS["Banker"], OUTCOME_PROBS["Player"], OUTCOME_PROBS["Tie"]]
    )[0]

# --- RECORD RESULT LOGIC ---
def record_result(result):
    """Record a game result without placing a bet."""
    st.session_state.history.append({
        "Bet On": None,  # No bet placed
        "Game Result": result,
        "Result": None,  # No win/loss
        "Wager": 0,
        "Net": 0,
        "Balance": st.session_state.bankroll[-1]
    })
    st.success(f"Recorded Game Result: **{result}**")

# --- PLACE BET LOGIC ---
def place_bet(selected_bet, bet_amount, t3_toggle):
    actual_result = get_random_result()
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

    # Display result
    result_text = f"Bet: **{selected_bet}**, Game Result: **{actual_result}**, You **{'Win' if win else 'Lose'}**!"
    if win:
        st.success(result_text)
    else:
        st.error(result_text)

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

# --- SIDEBAR OPTIONS ---
st.sidebar.title("Settings")
ai_toggle = st.sidebar.checkbox("Enable AI Prediction", value=st.session_state.ai_enabled)
t3_toggle = st.sidebar.checkbox("Use T3 Strategy", value=True)

st.session_state.ai_enabled = ai_toggle

# New Shoe Button
if st.sidebar.button("New Shoe"):
    st.session_state.history = []
    st.session_state.bankroll = [1000]
    st.session_state.t3_level = 1
    st.sidebar.success("New shoe started!")

# --- MAIN UI ---
st.title("Baccarat Predictor — Optimized for Profit per Shoe")

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if ai_toggle:
        ai_bet = predict_next_outcome()
        st.info(f"AI predicts: **{ai_bet}**")
        st.caption("Prediction based on historical patterns and 55% Banker bias.")
    else:
        st.write("Enable AI Prediction to see suggestions.")

with col2:
    st.subheader("Record Game Result")
    if st.button("Record Player"):
        record_result("Player")
    if st.button("Record Banker"):
        record_result("Banker")
    if st.button("Record Tie"):
        record_result("Tie")

    st.subheader("Place Your Bet")
    bet_amount = get_t3_bet() if t3_toggle else BASE_BET
    if ai_toggle and st.button("Bet AI Prediction"):
        place_bet(ai_bet, bet_amount, t3_toggle)
    if st.button("Bet Player"):
        place_bet("Player", bet_amount, t3_toggle)
    if st.button("Bet Banker"):
        place_bet("Banker", bet_amount, t3_toggle)
    if st.button("Bet Tie"):
        place_bet("Tie", bet_amount, t3_toggle)

with col3:
    st.metric("Current Bet Amount", f"${bet_amount}")

# --- METRICS ---
df = pd.DataFrame(st.session_state.history)
total_bets = len(df[df["Bet On"].notnull()])
wins = df[df["Result"] == "Win"].shape[0] if total_bets > 0 else 0
user_win_rate = wins / total_bets if total_bets > 0 else 0
avg_house_edge = df[df["Bet On"].notnull()]["Bet On"].map(HOUSE_EDGE).mean() if total_bets > 0 else 0
user_edge = user_win_rate - (1 - avg_house_edge) if total_bets > 0 else 0

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
            (df[df["Bet On"].notnull()].iloc[:i+1]["Result"].value_counts().get("Win", 0) / (i+1)) - 
            (1 - df[df["Bet On"].notnull()].iloc[:i+1]["Bet On"].map(HOUSE_EDGE).mean())
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
