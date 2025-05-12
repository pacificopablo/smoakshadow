import streamlit as st
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []  # List of tuples: [(outcome, points), ...]
if 't3_level' not in st.session_state:
    st.session_state.t3_level = 1
if 'round_results' not in st.session_state:
    st.session_state.round_results = []
if 'bet_sequence' not in st.session_state:
    st.session_state.bet_sequence = []  # Track wins/losses: ["W", "L", ...]
if 'current_bet' not in st.session_state:
    st.session_state.current_bet = 500  # Base bet: $500
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 10000  # Starting bankroll: $10,000
if 'total_wins' not in st.session_state:
    st.session_state.total_wins = 0
if 'total_losses' not in st.session_state:
    st.session_state.total_losses = 0

# UI Title
st.title("Algo Z100-Inspired Baccarat Predictor with Z 1003.1 Algorithm")

# Game Result Entry with Form
st.subheader("Enter Game Result")
with st.form("game_result"):
    outcome = st.selectbox("Result:", ["Player Wins", "Banker Wins", "Tie"])
    points = st.number_input("Points (0-9):", min_value=0, max_value=9, value=0)
    submit = st.form_submit_button("Add Result")
    if submit:
        outcome_map = {"Player Wins": "P", "Banker Wins": "B", "Tie": "T"}
        st.session_state.history.append((outcome_map[outcome], points))

# Bead Plate Function (Casino-Style with Plotly)
def create_bead_plate(history, rows=6, cols=12):
    if history:
        cols = max(cols, (len(history) + rows - 1) // rows)
    
    x_positions, y_positions, colors, points_text = [], [], [], []
    for i, (outcome, points) in enumerate(history):
        row = i // cols
        col = i % cols
        if row < rows:
            x_positions.append(col)
            y_positions.append(rows - 1 - row)
            colors.append({"P": "blue", "B": "red", "T": "green"}[outcome])
            points_text.append(str(points))
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="markers+text",
            marker=dict(
                color=colors,
                size=28,
                line=dict(width=1, color="black")
            ),
            text=points_text,
            textposition="middle center",
            textfont=dict(color="white", size=16, family="Arial"),
            hoverinfo="text",
            texttemplate="%{text}",
        )
    )
    
    next_pos = len(history)
    next_row = next_pos // cols
    next_col = next_pos % cols
    arrow_x, arrow_y, arrow_dx, arrow_dy = None, None, 0, 0
    if next_row < rows:
        arrow_x = next_col
        arrow_y = rows - 1 - next_row
        if next_row < rows - 1 and next_col == cols - 1:
            arrow_dx = 0.9
            arrow_dy = 0
        elif next_row < rows - 1:
            arrow_dx = 0
            arrow_dy = -0.9
        else:
            arrow_dx = 0.9
            arrow_dy = 0
    
    if arrow_x is not None and arrow_y is not None:
        fig.add_shape(
            type="line",
            x0=arrow_x, y0=arrow_y,
            x1=arrow_x + arrow_dx, y1=arrow_y + arrow_dy,
            line=dict(color="white", width=4),
            xref="x", yref="y"
        )
        if arrow_dx > 0:
            fig.add_shape(
                type="path",
                path=f"M {arrow_x + arrow_dx} {arrow_y} L {arrow_x + arrow_dx - 0.3} {arrow_y + 0.3} L {arrow_x + arrow_dx - 0.3} {arrow_y - 0.3} Z",
                fillcolor="white",
                line=dict(color="white"),
                xref="x", yref="y"
            )
        else:
            fig.add_shape(
                type="path",
                path=f"M {arrow_x} {arrow_y + arrow_dy} L {arrow_x + 0.3} {arrow_y + arrow_dy + 0.3} L {arrow_x - 0.3} {arrow_y + arrow_dy + 0.3} Z",
                fillcolor="white",
                line=dict(color="white"),
                xref="x", yref="y"
            )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, cols-0.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, rows-0.5]),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="#4B0082",
        width=900,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    for row in range(rows):
        for col in range(cols):
            if (rows - 1 - row) not in y_positions or col not in x_positions or (col, rows - 1 - row) not in zip(x_positions, y_positions):
                fig.add_shape(
                    type="rect",
                    x0=col - 0.35, y0=(rows - 1 - row) - 0.35,
                    x1=col + 0.35, y1=(rows - 1 - row) + 0.35,
                    fillcolor="lightgray",
                    line=dict(color="lightgray", width=1),
                    xref="x", yref="y"
                )
    
    return fig

# Show Bead Plate in Game History
st.subheader("Game History")
st.write("Blue Circle: Player (P), Red Circle: Banker (B), Green Circle: Tie (T)")
if st.session_state.history:
    bead_plate = create_bead_plate(st.session_state.history)
    st.plotly_chart(bead_plate, use_container_width=True)
else:
    st.write("No history yet. Add game outcomes to see the bead plate.")

# Show raw history
st.write("Raw History: " + " ".join([f"{outcome}({points})" for outcome, points in st.session_state.history]))

# Z 1003.1 Bet Selection Logic
def z1003_bet_selection(history, bet_sequence):
    # Step 1: Check if we should stop or wait
    if not history:
        return "Waiting for history...", 500  # No history yet
    
    # Check recent bet outcomes
    if len(bet_sequence) >= 3 and bet_sequence[-3:] == ["L", "L", "L"]:
        # Stop after three losses, reset sequence and bet
        st.session_state.bet_sequence = []
        st.session_state.current_bet = 500
        return "Stop: Three losses in a row. Resetting...", 500
    
    if bet_sequence and bet_sequence[-1] == "W":
        # Reset after a win, wait for a new streak
        st.session_state.bet_sequence = []
        st.session_state.current_bet = 500
        # Look for a new streak (at least 2 same outcomes in a row)
        outcomes = [outcome for outcome, _ in history[-3:]]
        if len(outcomes) >= 2 and outcomes[-1] == outcomes[-2] and outcomes[-1] in ["P", "B"]:
            return outcomes[-1], 500  # Bet on the streak
        return "Waiting for a new streak...", 500
    
    # Step 2: Analyze bead plate for bet selection (last 3 outcomes)
    last_three = [outcome for outcome, _ in history[-3:]]
    if len(last_three) < 3:
        return "Waiting for more history...", 500
    
    # Avoid symmetrical patterns (e.g., P-B-P or B-P-B)
    if last_three == ["P", "B", "P"] or last_three == ["B", "P", "B"]:
        # If symmetrical, bet against the last outcome to break the pattern
        prediction = "B" if last_three[-1] == "P" else "P"
    else:
        # Otherwise, bet on the most frequent outcome in the last 3 (favoring streaks)
        p_count = last_three.count("P")
        b_count = last_three.count("B")
        prediction = "P" if p_count > b_count else "B"
    
    # Step 3: Adjust bet amount
    if bet_sequence and bet_sequence[-1] == "L":
        # Increase bet by $100 after a loss
        st.session_state.current_bet += 100
    else:
        # Reset bet to $500 if starting a new sequence
        st.session_state.current_bet = 500
    
    return prediction, st.session_state.current_bet

# Predictor Output
st.subheader("Prediction")
prediction, bet_amount = z1003_bet_selection(st.session_state.history, st.session_state.bet_sequence)
if "Waiting" in prediction or "Stop" in prediction:
    st.info(prediction)
else:
    st.success(f"Next Bet: {prediction} | Bet Amount: ${bet_amount}")

# Record Bet Outcome
st.subheader("Record Bet Outcome")
col4, col5 = st.columns(2)
with col4:
    if st.button("Win"):
        st.session_state.bet_sequence.append("W")
        st.session_state.total_wins += 1
        st.session_state.bankroll += bet_amount
        update_t3("W")
with col5:
    if st.button("Loss"):
        st.session_state.bet_sequence.append("L")
        st.session_state.total_losses += 1
        st.session_state.bankroll -= bet_amount
        update_t3("L")

# Performance Tracking
st.subheader("Performance Metrics")
total_bets = st.session_state.total_wins + st.session_state.total_losses
win_rate = (st.session_state.total_wins / total_bets * 100) if total_bets > 0 else 0
st.markdown(f"**Bankroll:** ${st.session_state.bankroll}")
st.markdown(f"**Total Wins:** {st.session_state.total_wins}")
st.markdown(f"**Total Losses:** {st.session_state.total_losses}")
st.markdown(f"**Win Rate:** {win_rate:.1f}%")
st.markdown(f"**Current Bet Sequence:** {st.session_state.bet_sequence}")

# T3 System Logic
def update_t3(bet_result):
    st.session_state.round_results.append(bet_result)
    if len(st.session_state.round_results) == 3:
        wins = st.session_state.round_results.count("W")
        losses = 3 - wins
        if wins == 3:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 2)
        elif wins == 2:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif wins == 1:
            st.session_state.t3_level += 1
        elif wins == 0:
            st.session_state.t3_level += 2
        st.session_state.round_results = []

st.markdown(f"**T3 Level:** {st.session_state.t3_level}")
st.markdown(f"**Current Round Results:** {st.session_state.round_results}")

# Reset Button
if st.button("Reset All"):
    st.session_state.history = []
    st.session_state.t3_level = 1
    st.session_state.round_results = []
    st.session_state.bet_sequence = []
    st.session_state.current_bet = 500
    st.session_state.bankroll = 10000
    st.session_state.total_wins = 0
    st.session_state.total_losses = 0
