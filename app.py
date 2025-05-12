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

# UI Title
st.title("Algo Z100-Inspired Baccarat Predictor")

# Game Result Entry with Point Input
st.subheader("Enter Game Result")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Player Wins"):
        points = st.number_input("Enter Player points (0-9):", min_value=0, max_value=9, value=0, key="player_points")
        if points is not None:
            st.session_state.history.append(("P", points))
with col2:
    if st.button("Banker Wins"):
        points = st.number_input("Enter Banker points (0-9):", min_value=0, max_value=9, value=0, key="banker_points")
        if points is not None:
            st.session_state.history.append(("B", points))
with col3:
    if st.button("Tie"):
        points = st.number_input("Enter Tie points (0-9):", min_value=0, max_value=9, value=0, key="tie_points")
        if points is not None:
            st.session_state.history.append(("T", points))

# Bead Plate Function (Casino-Style with Plotly)
def create_bead_plate(history, rows=6, cols=12):
    # Calculate required columns based on history length
    if history:
        cols = max(cols, (len(history) + rows - 1) // rows)
    
    # Initialize lists for scatter plot (circles and text)
    x_positions = []
    y_positions = []
    colors = []
    points_text = []
    
    # Map history to grid positions
    for i, (outcome, points) in enumerate(history):
        row = i // cols
        col = i % cols
        if row < rows:  # Only include within row limit
            x_positions.append(col)
            y_positions.append(rows - 1 - row)  # Invert y-axis for top-to-bottom
            colors.append({"P": "blue", "B": "red", "T": "green"}[outcome])
            points_text.append(str(points))
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add scatter plot for shaded circles
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="markers+text",
            marker=dict(
                color=colors,
                size=30,  # Larger size to fit numbers
                line=dict(width=1, color="black")  # Black outline
            ),
            text=points_text,
            textposition="middle center",
            textfont=dict(color="white", size=14),  # White numbers
            hoverinfo="text",
            texttemplate="%{text}",
        )
    )
    
    # Calculate next position for arrows
    next_pos = len(history)
    next_row = next_pos // cols
    next_col = next_pos % cols
    arrow_x, arrow_y, arrow_dx, arrow_dy = None, None, 0, 0
    if next_row < rows:
        arrow_x = next_col
        arrow_y = rows - 1 - next_row
        if next_row < rows - 1:  # Point down if column not full
            arrow_dx = 0
            arrow_dy = -0.8
        else:  # Point left if moving to next column
            arrow_dx = 0.8
            arrow_dy = 0
    
    # Add arrows
    if arrow_x is not None and arrow_y is not None:
        fig.add_shape(
            type="line",
            x0=arrow_x, y0=arrow_y,
            x1=arrow_x + arrow_dx, y1=arrow_y + arrow_dy,
            line=dict(color="white", width=3),
            xref="x", yref="y"
        )
        # Add arrowhead
        fig.add_shape(
            type="path",
            path=f"M {arrow_x + arrow_dx} {arrow_y + arrow_dy} L {arrow_x + arrow_dx - 0.2 if arrow_dx else arrow_x} {arrow_y + arrow_dy + 0.2 if arrow_dy else arrow_y + 0.2} L {arrow_x + arrow_dx + 0.2 if arrow_dx else arrow_x} {arrow_y + arrow_dy - 0.2 if arrow_dy else arrow_y - 0.2} Z",
            fillcolor="white",
            line=dict(color="white"),
            xref="x", yref="y"
        )
    
    # Customize layout
    fig.update_layout(
        title="",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, cols - 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, rows - 0.5]
        ),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="#4B0082",  # Purple background like the image
        width=800,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Add empty cells as light gray squares
    for row in range(rows):
        for col in range(cols):
            if (rows - 1 - row) not in y_positions or col not in x_positions or (col, rows - 1 - row) not in zip(x_positions, y_positions):
                fig.add_shape(
                    type="rect",
                    x0=col - 0.4, y0=(rows - 1 - row) - 0.4,
                    x1=col + 0.4, y1=(rows - 1 - row) + 0.4,
                    fillcolor="lightgray",
                    line=dict(color="lightgray"),
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

# Show raw history (optional, for reference)
st.write("Raw History: " + " ".join([f"{outcome}({points})" for outcome, points in st.session_state.history]))

# Prediction Engine (Hybrid AI)
def predict_next_move(history):
    if len(history) < 4:
        return "Waiting", 0.0
    last_four = [outcome for outcome, _ in history[-4:]]
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
