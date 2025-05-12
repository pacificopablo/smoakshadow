import streamlit as st
import random
import pandas as pd
import numpy as np

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

# Show history
st.subheader("Game History")
st.write(" ".join(st.session_state.history))

# Prediction Engine (Hybrid AI)
def predict_next_move(history):
    if len(history) < 4:
        return "Waiting", 0.0

    # Pattern vote (very simple example)
    last_four = history[-4:]
    pattern_vote = "B" if last_four.count("B") > last_four.count("P") else "P"

    # Random vote
    random_vote = random.choice(["B", "P"])

    # Weighted decision
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
