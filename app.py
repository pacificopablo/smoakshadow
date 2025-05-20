import streamlit as st
from collections import defaultdict, Counter

class BaccaratPredictor:
    def __init__(self):
        # Initialize session state variables if not already set
        if 'history' not in st.session_state:
            st.session_state.history = []
            st.session_state.transitions = defaultdict(Counter)
            st.session_state.bet = 1
            st.session_state.profit = 0
            st.session_state.last_prediction = None
            st.session_state.strategy = "Flat Bet"
            st.session_state.masterline_step = 0
            st.session_state.in_force2 = False
            st.session_state.force2_failed = False
            st.session_state.break_countdown = 0

    def update(self, result):
        # Update transitions with the new result
        if st.session_state.history:
            prev = st.session_state.history[-1]
            st.session_state.transitions[prev][result] += 1

        # Append result to history
        st.session_state.history.append(result)

        # Handle break countdown
        if st.session_state.break_countdown > 0:
            st.session_state.break_countdown -= 1
            return "?", f"Status: Break ({st.session_state.break_countdown} left)\nPrediction paused."

        # Update bet and profit based on strategy
        win = st.session_state.last_prediction == result if result in ("P", "B") and st.session_state.last_prediction else False
        strategy = st.session_state.strategy

        if result in ("P", "B") and st.session_state.last_prediction in ("P", "B"):
            if strategy == "Flat Bet":
                st.session_state.bet = 1
                st.session_state.profit += st.session_state.bet if win else -st.session_state.bet

            elif strategy == "D'Alembert":
                if win:
                    st.session_state.profit += st.session_state.bet
                    st.session_state.bet = max(1, st.session_state.bet - 2)
                else:
                    st.session_state.profit -= st.session_state.bet
                    st.session_state.bet += 1

            elif strategy == "-1 +2":
                if win:
                    st.session_state.profit += st.session_state.bet
                    st.session_state.bet += 2
                else:
                    st.session_state.profit -= st.session_state.bet
                    st.session_state.bet = max(1, st.session_state.bet - 1)

            elif strategy == "Suchi Masterline":
                self.handle_masterline(win)

        # Predict next outcome
        prediction, explanation = self.predict_next()
        st.session_state.last_prediction = prediction if prediction in ("P", "B") else None
        return prediction, explanation

    def handle_masterline(self, win):
        if st.session_state.in_force2:
            if win:
                st.session_state.profit += 2
                st.session_state.in_force2 = False
                st.session_state.bet = 1
                st.session_state.masterline_step = 0
            else:
                st.session_state.profit -= 2
                st.session_state.force2_failed = True
                st.session_state.in_force2 = False
                st.session_state.break_countdown = 3
        elif st.session_state.force2_failed:
            st.session_state.force2_failed = False
            st.session_state.bet = 1
            st.session_state.masterline_step = 0
        elif win:
            ladder = [1, 3, 2, 5]
            st.session_state.profit += ladder[st.session_state.masterline_step]
            st.session_state.masterline_step += 1
            if st.session_state.masterline_step > 3:
                st.session_state.masterline_step = 0
                st.session_state.bet = 1
            else:
                st.session_state.bet = ladder[st.session_state.masterline_step]
        else:
            if st.session_state.masterline_step == 0:
                st.session_state.in_force2 = True
                st.session_state.bet = 2
            else:
                st.session_state.profit -= st.session_state.bet
                st.session_state.break_countdown = 3
                st.session_state.masterline_step = 0
                st.session_state.bet = 1

    def predict_next(self):
        if st.session_state.break_countdown > 0:
            return "?", "In break. Prediction paused."

        if not st.session_state.history:
            return "?", "No history yet."

        last = st.session_state.history[-1]
        counts = st.session_state.transitions[last]
        total = sum(counts.values())

        if not counts:
            return "?", f"No data available after '{last}' to predict next."

        probabilities = {k: v / total for k, v in counts.items()}
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        prediction = sorted_probs[0][0]

        explanation = f"Last result: {last}\n"
        explanation += "Transition probabilities:\n"
        for outcome, prob in sorted_probs:
            explanation += f"  {last} → {outcome}: {prob:.2f}\n"

        return prediction, explanation

def render_bead_plate():
    history = st.session_state.history
    if not history:
        st.markdown("**Bead Plate**: No results yet.")
        return

    # Initialize 6-row grid
    rows = 6
    cols = (len(history) + rows - 1) // rows  # Calculate required columns
    bead_plate = [[" " for _ in range(cols)] for _ in range(rows)]

    # Fill bead plate: left-to-right, top-to-bottom
    for i, outcome in enumerate(history):
        row = i % rows
        col = i // rows
        symbol = "🟢" if outcome == "T" else "🔵" if outcome == "P" else "🔴"
        bead_plate[row][col] = symbol

    # Render bead plate using markdown table
    st.markdown("**Bead Plate**")
    table = "| " + " | ".join([""] * cols) + " |\n"
    table += "|---" * cols + "|\n"
    for row in bead_plate:
        table += "| " + " | ".join(row) + " |\n"
    st.markdown(table)

def main():
    st.title("Baccarat Predictor with Masterline Strategy")

    # Initialize predictor
    predictor = BaccaratPredictor()

    # Strategy selection
    st.session_state.strategy = st.selectbox(
        "Select Betting Strategy",
        ["Flat Bet", "D'Alembert", "-1 +2", "Suchi Masterline"],
        index=["Flat Bet", "D'Alembert", "-1 +2", "Suchi Masterline"].index(st.session_state.strategy)
    )

    # Display current state
    status = "Normal" if st.session_state.break_countdown == 0 else f"Break ({st.session_state.break_countdown} left)"
    st.write(f"**Status:** {status}")
    st.write(f"**Current Bet:** {st.session_state.bet}")
    st.write(f"**Profit by Unit:** {st.session_state.profit}")

    # Input buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("P", use_container_width=True):
            prediction, explanation = predictor.update("P")
            st.session_state.prediction = prediction
            st.session_state.explanation = explanation
    with col2:
        if st.button("B", use_container_width=True):
            prediction, explanation = predictor.update("B")
            st.session_state.prediction = prediction
            st.session_state.explanation = explanation
    with col3:
        if st.button("T", use_container_width=True):
            prediction, explanation = predictor.update("T")
            st.session_state.prediction = prediction
            st.session_state.explanation = explanation

    # Display prediction
    prediction = st.session_state.get("prediction", "?")
    st.markdown(f"**Next Prediction:** {prediction}")

    # Display explanation
    explanation = st.session_state.get("explanation", "Explanation:\n")
    st.text_area("Explanation", explanation, height=150, disabled=True)

    # Display bead plate
    render_bead_plate()

    # Reset button
    if st.button("Reset"):
        predictor.reset()
        st.session_state.prediction = "?"
        st.session_state.explanation = "Explanation:\n"

if __name__ == "__main__":
    main()
