import streamlit as st
from place_result_function_aigood import place_result  # Save artifact as a separate file
# Add session state initialization above
st.title("Mang Baccarat")
col1, col2, col3 = st.columns(3)
if col1.button("Player"):
    place_result("P")
if col2.button("Banker"):
    place_result("B")
if col3.button("Tie"):
    place_result("T")
st.write(f"Bankroll: ${st.session_state.bankroll:.2f}")
st.write(st.session_state.advice)
