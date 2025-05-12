import streamlit as st
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 't3_level' not in st.session_state:
    st.session_state.t3_level = 1
if 'round_results' not in st.session_state:
    st.session_state.round_results = []

# UI Title
st.title("Algo Z100-Inspired Baccarat Predictor")

# Game Result Entry
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Player Wins"):
        st.session_state.history.append("P")
with col2:
    if st.button("Banker Wins"):
        st.session_state.history.append("B")
with col3:
    if st.button("Tie"):
        st.session_state.history.append("T")

# Bead Plate Function (Casino-Style with Plotly)
def create_bead_plate(history, rows=6, cols=12):
    # Calculate required columns based on history length
    if history:
        cols = max(cols, (len(history) + rows - 1) // rows)
    
    # Initialize lists for scatter plot
    x_positions = []
    y_positions = []
    colors = []
    markers = []
    sizes = []
    
    # Map history to grid positions
    for i, outcome in enumerate(history):
        row = i // cols
        col = i % cols
        if row < rows:  # Only include within row limit
            x_positions.append(col)
            y_positions.append(rows - 1 - row)  # Invert y-axis for top-to-bottom
            if outcome == "P":
                colors.append("blue")
                markers.append("circle")
                sizes.append(20)  # Larger circle for Player
            elif outcome == "B":
                colors.append("red")
                markers.append("circle")
                sizes.append(20)  # Larger circle for Banker
            elif outcome == "T":
                colors.append("green")
                markers.append("circle")  # Small dot for Tie
                sizes.append(8)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add scatter plot for outcomes
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="markers",
            marker=dict(
                color=colors,
                size=sizes,
                symbol=markers,
                line=dict(width=1, color="black")  # Black outline for clarity
            ),
            text=[f"{outcome} ({x},{rows-1-y})" for outcome, x, y in zip(history[:len(x_positions)], x_positions, y_positions)],
            hoverinfo="text"
        )
    )
    
    # Customize layout
    fig.update_layout(
        title="",
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="lightgray",
            tickvals=list(range(cols)),
            ticktext=[""] * cols,  # Hide x-axis labels
            range=[-0.5, cols - 0.5]
        ),
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor="lightgray",
            tickvals=list(range(rows)),
            ticktext=[""] * rows,  # Hide y-axis labels
            range=[-0.5, rows - 0.5]
        ),
        showlegend=False,
        plot_bgcolor="white",
        width=600,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

# Show Bead Plate
st.subheader("Bead Plate (Game History)")
st.write("Blue Circle: Player (P), Red Circle: Banker (B), Green Dot: Tie (T)")
if st.session_state.history:
    bead_plate = create_bead_plate(st.session_state.history)
    st.plotly_chart(bead_plate, use_container_width=True)
else:
    st.write("No history yet. Add game outcomes to see the bead plate.")

# Show raw history (optional, for reference)
st.write("Raw History: " + " ".join(st.session_state.history))

# Prediction Engine (Hybrid AI)
def predict_next_move(history):
    if len(history) < 4:
        return "Waiting", 0.0
    last_four = history[-4:]
    pattern_vote = "B" if last_four.count("B") > last_four.count("P") else "P"
    random_vote = random.choice(["B", "P"])
    votes = [pattern_vote, random_vote]
    prediction = max(set(votes), key=votes.count)
    confidence = round(votes.count(prediction) / len(votes), 2)
    if confidence < 0.515:
        return "Waiting", confidence
    return prediction, confidence

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

# Predictor Output
st.subheader("Prediction")
prediction, confidence = predict_next_move(st.session_state.history)
if prediction == "Waiting":
    st.info("Waiting for strong signal...")
else:
    st.success(f"Next Bet: {prediction} (Confidence: {confidence * 100:.1f}%)")

# Simulated Result Feedback (for T3 system)
st.subheader("Record Bet Outcome")
col4, col5 = st.columns(2)
with col4:
    if st.button("Win"):
        update_t3("W")
with col5:
    if st.button("Loss"):
        update_t3("L")

st.markdown(f"**T3 Level:** {st.session_state.t3_level}")
st.markdown(f"**Current Round Results:** {st.session_state.round_results}")

# Reset Button
if st.button("Reset All"):
    st.session_state.history = []
    st.session_state.t3_level = 1
    st.session_state.round_results = []
