import streamlit as st
import uuid
from collections import defaultdict, Counter

def initialize_session_state():
    """Initialize session state with default values."""
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
        st.session_state.transition_counts = {'PP': 0, 'PB': 0, 'PT': 0, 'BP': 0, 'BB': 0, 'BT': 0, 'TP': 0, 'TB': 0, 'TT': 0}
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.bet = 1  # Bet multiplier, to be scaled by base_bet
        st.session_state.bankroll = None  # To be set by user in dollars
        st.session_state.base_bet = None  # To be set by user in dollars
        st.session_state.last_prediction = None
        st.session_state.strategy = "Flat Bet"
        st.session_state.masterline_step = 0
        st.session_state.in_force2 = False
        st.session_state.force2_failed = False
        st.session_state.break_countdown = 0
        st.session_state.t3_level = 1  # T3 bet multiplier
        st.session_state.t3_results = []  # Tracks W/L for T3
        st.session_state.transitions = defaultdict(Counter)
        st.session_state.explanation = "No explanation available."
        st.session_state.session_started = False

def reset_session():
    """Reset session state to initial values."""
    st.session_state.update({
        'sequence': [],
        'transition_counts': {'PP': 0, 'PB': 0, 'PT': 0, 'BP': 0, 'BB': 0, 'BT': 0, 'TP': 0, 'TB': 0, 'TT': 0},
        'session_id': str(uuid.uuid4()),
        'bet': 1,
        'bankroll': None,
        'base_bet': None,
        'last_prediction': None,
        'strategy': "Flat Bet",
        'masterline_step': 0,
        'in_force2': False,
        'force2_failed': False,
        'break_countdown': 0,
        't3_level': 1,
        't3_results': [],
        'transitions': defaultdict(Counter),
        'explanation': "No explanation available.",
        'session_started': False
    })

def place_result(result):
    """Append a game result, update transition counts, and handle betting logic."""
    try:
        # Update history and transitions
        if st.session_state.sequence:
            prev = st.session_state.sequence[-1]
            st.session_state.transitions[prev][result] += 1
            transition = f"{prev}{result}"
            st.session_state.transition_counts[transition] += 1
        st.session_state.sequence.append(result)

        # Handle betting logic if not in break countdown
        if st.session_state.break_countdown > 0:
            st.session_state.break_countdown -= 1
            return

        # Determine if the last prediction was correct
        win = st.session_state.last_prediction == result if result in ("P", "B") and st.session_state.last_prediction else False
        strategy = st.session_state.strategy
        actual_bet = st.session_state.bet * st.session_state.base_bet

        # Check for insufficient bankroll before betting
        if st.session_state.bankroll < actual_bet and result in ("P", "B"):
            st.error("Insufficient bankroll to place the bet! Please reset the session or adjust settings.")
            return

        # Update bankroll based on strategy
        if result in ("P", "B") and st.session_state.last_prediction in ("P", "B"):
            commission = 0.95 if result == "B" else 1.0  # 5% commission on Banker wins
            if strategy == "Flat Bet":
                st.session_state.bet = 1
                if win:
                    st.session_state.bankroll += actual_bet * commission
                else:
                    st.session_state.bankroll -= actual_bet

            elif strategy == "D'Alembert":
                if win:
                    st.session_state.bankroll += actual_bet * commission
                    st.session_state.bet = max(1, st.session_state.bet - 2)
                else:
                    st.session_state.bankroll -= actual_bet
                    st.session_state.bet += 1

            elif strategy == "-1 +2":
                if win:
                    st.session_state.bankroll += actual_bet * commission
                    st.session_state.bet += 2
                else:
                    st.session_state.bankroll -= actual_bet
                    st.session_state.bet = max(1, st.session_state.bet - 1)

            elif strategy == "Suchi Masterline":
                handle_masterline(win, result)

            elif strategy == "T3":
                handle_t3(win, result)

        # Update prediction
        prediction, explanation = predict_next()
        st.session_state.last_prediction = prediction if prediction in ("P", "B") else None
        st.session_state.explanation = explanation
    except Exception as e:
        st.error(f"Error processing result: {e}")

def handle_masterline(win, result):
    """Handle Suchi Masterline betting strategy with commission on Banker wins."""
    actual_bet = st.session_state.bet * st.session_state.base_bet
    commission = 0.95 if result == "B" else 1.0  # 5% commission on Banker wins
    if st.session_state.bankroll < actual_bet:
        st.error("Insufficient bankroll for Masterline bet! Please reset the session or adjust settings.")
        return
    if st.session_state.in_force2:
        if win:
            st.session_state.bankroll += (2 * st.session_state.base_bet) * commission
            st.session_state.in_force2 = False
            st.session_state.bet = 1
            st.session_state.masterline_step = 0
        else:
            st.session_state.bankroll -= 2 * st.session_state.base_bet
            st.session_state.force2_failed = True
            st.session_state.in_force2 = False
            st.session_state.break_countdown = 3
    elif st.session_state.force2_failed:
        st.session_state.force2_failed = False
        st.session_state.bet = 1
        st.session_state.masterline_step = 0
    elif win:
        ladder = [1, 3, 2, 5]
        st.session_state.bankroll += (ladder[st.session_state.masterline_step] * st.session_state.base_bet) * commission
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
            st.session_state.bankroll -= actual_bet
            st.session_state.break_countdown = 3
            st.session_state.masterline_step = 0
            st.session_state.bet = 1

def handle_t3(win, result):
    """Handle T3 betting strategy with commission on Banker wins."""
    actual_bet = st.session_state.bet * st.session_state.base_bet
    commission = 0.95 if result == "B" else 1.0  # 5% commission on Banker wins
    if st.session_state.bankroll < actual_bet:
        st.error("Insufficient bankroll for T3 bet! Please reset the session or adjust settings.")
        return
    if win:
        st.session_state.bankroll += actual_bet * commission
        if len(st.session_state.t3_results) == 0:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)  # Decrease level on first-step win
        st.session_state.t3_results.append('W')
    else:
        st.session_state.bankroll -= actual_bet
        st.session_state.t3_results.append('L')

    # Update T3 level after 3 results
    if len(st.session_state.t3_results) == 3:
        wins = st.session_state.t3_results.count('W')
        losses = st.session_state.t3_results.count('L')
        if wins > losses:
            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
        elif losses > wins:
            st.session_state.t3_level += 1
        st.session_state.t3_results = []

    # Set next bet multiplier
    st.session_state.bet = st.session_state.t3_level

def predict_next():
    """Predict the next outcome based on transition probabilities."""
    if st.session_state.break_countdown > 0:
        return "?", "In break. Prediction paused."

    if not st.session_state.sequence:
        return "?", "No history yet."

    last = st.session_state.sequence[-1]
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

def render_session_setup():
    """Render UI for starting, resetting a session, and selecting strategy."""
    with st.expander("Session Setup", expanded=not st.session_state.session_started):
        try:
            st.markdown("**Configure Session**")
            bankroll = st.number_input("Initial Bankroll ($)", min_value=1.0, value=100.0, step=1.0, format="%.2f")
            base_bet = st.number_input("Base Bet ($)", min_value=0.1, value=1.0, step=0.1, format="%.2f")
            strategy = st.selectbox(
                "Select Betting Strategy",
                ["Flat Bet", "D'Alembert", "-1 +2", "Suchi Masterline", "T3"],
                index=["Flat Bet", "D'Alembert", "-1 +2", "Suchi Masterline", "T3"].index(st.session_state.strategy)
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Session", disabled=bankroll < base_bet):
                    if bankroll < base_bet:
                        st.error("Initial bankroll must be at least equal to the base bet.")
                        return
                    st.session_state.bankroll = bankroll
                    st.session_state.base_bet = base_bet
                    st.session_state.strategy = strategy
                    st.session_state.bet = 1
                    st.session_state.t3_level = 1
                    st.session_state.t3_results = []
                    st.session_state.session_started = True
                    st.success("Session started!")
            with col2:
                if st.session_state.session_started and st.button("Reset Session"):
                    reset_session()
                    st.success("Session reset!")
        except Exception as e:
            st.error(f"Error setting up session: {e}")

def render_result_input():
    """Render UI for inputting game results."""
    with st.expander("Result Input", expanded=True):
        try:
            if not st.session_state.session_started:
                st.warning("Please start a session with a valid bankroll and base bet.")
                return
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Player (P)"):
                    place_result('P')
            with col2:
                if st.button("Banker (B)"):
                    place_result('B')
            with col3:
                if st.button("Tie (T)"):
                    place_result('T')
            with col4:
                if st.button("Undo Last"):
                    if st.session_state.sequence:
                        last_result = st.session_state.sequence.pop()
                        if len(st.session_state.sequence) >= 1 and last_result in ['P', 'B', 'T'] and st.session_state.sequence[-1] in ['P', 'B', 'T']:
                            transition = f"{st.session_state.sequence[-1]}{last_result}"
                            st.session_state.transition_counts[transition] = max(0, st.session_state.transition_counts[transition] - 1)
                            st.session_state.transitions[st.session_state.sequence[-1]][last_result] -= 1
                        # Reset betting state for simplicity after undo
                        st.session_state.bet = 1
                        st.session_state.masterline_step = 0
                        st.session_state.in_force2 = False
                        st.session_state.force2_failed = False
                        st.session_state.break_countdown = 0
                        st.session_state.t3_level = 1
                        st.session_state.t3_results = []
                        st.session_state.last_prediction = None
                        st.session_state.explanation = "No explanation available."
        except Exception as e:
            st.error(f"Error processing input: {e}")

def render_prediction():
    """Render the prediction section with next prediction and explanation."""
    with st.expander("Prediction", expanded=True):
        try:
            prediction, _ = predict_next()
            st.markdown(f"**Next Prediction**: {prediction}")
            st.markdown("**Explanation**:")
            st.text(st.session_state.get('explanation', "No explanation available."))
        except Exception as e:
            st.error(f"Error rendering prediction: {e}")

def render_status():
    """Render session status with hands played, streak status, and betting info."""
    with st.expander("Session Status", expanded=True):
        try:
            st.markdown(f"**Hands Played**: {len(st.session_state.sequence)}")
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
            streak_info = "No streak detected"
            if len(valid_sequence) >= 4 and len(set(valid_sequence[-4:])) == 1:
                streak_length = len([x for x in valid_sequence[::-1] if x == valid_sequence[-1]])
                streak_info = f"Streak detected: {valid_sequence[-1]} x {streak_length}"
            st.markdown(f"**Streak Status**: {streak_info}")
            st.markdown(f"**Transition Counts**: PP: {st.session_state.transition_counts['PP']}, "
                        f"PB: {st.session_state.transition_counts['PB']}, "
                        f"PT: {st.session_state.transition_counts['PT']}, "
                        f"BP: {st.session_state.transition_counts['BP']}, "
                        f"BB: {st.session_state.transition_counts['BB']}, "
                        f"BT: {st.session_state.transition_counts['BT']}, "
                        f"TP: {st.session_state.transition_counts['TP']}, "
                        f"TB: {st.session_state.transition_counts['TB']}, "
                        f"TT: {st.session_state.transition_counts['TT']}")
            st.markdown(f"**Current Bet**: ${st.session_state.bet * st.session_state.base_bet:.2f} (Multiplier: {st.session_state.bet}x)")
            st.markdown(f"**Bankroll**: ${st.session_state.bankroll:.2f}")
            status = "Normal"
            if st.session_state.break_countdown > 0:
                status = f"Break ({st.session_state.break_countdown} left)"
            elif st.session_state.strategy == "Suchi Masterline":
                if st.session_state.in_force2:
                    status = "Force 2"
                elif st.session_state.masterline_step > 0:
                    status = f"Ladder Step {st.session_state.masterline_step + 1}"
                else:
                    status = "Base"
            elif st.session_state.strategy == "T3":
                status = f"T3 Level {st.session_state.t3_level}, Results: {st.session_state.t3_results}"
            st.markdown(f"**Status**: {status}")
        except Exception as e:
            st.error(f"Error rendering status: {e}")

def render_bead_plate():
    """Render the bead plate visualization of game results with responsive scrolling."""
    with st.expander("Bead Plate"):
        try:
            if not st.session_state.sequence:
                st.info("No results yet. Enter results to see the bead plate.")
                return

            rows = 6
            results = st.session_state.sequence
            num_results = len(results)
            cols = (num_results + rows - 1) // rows

            grid = [['' for _ in range(cols)] for _ in range(rows)]
            for i, result in enumerate(results):
                row = i % rows
                col = i // rows
                grid[row][col] = result

            html = """
            <style>
                .bead-plate-container {
                    background-color: #007BFF; /* Blue background */
                    padding: 10px;
                    border-radius: 8px;
                    overflow-x: auto; /* Enable horizontal scrolling */
                    max-width: 100%; /* Fit container to viewport */
                    -webkit-overflow-scrolling: touch; /* Smooth scrolling on mobile */
                }
                .bead-plate-table {
                    border-collapse: collapse;
                    margin: 0;
                    min-width: 100%; /* Ensure table can grow wide */
                }
                .bead-plate-table td {
                    width: 30px;
                    height: 30px;
                    min-width: 30px; /* Prevent cells from shrinking too much */
                    text-align: center;
                    vertical-align: middle;
                    border: 1px solid #ccc;
                    font-weight: bold;
                    font-size: 14px;
                }
                .player {
                    background-color: #3182ce;
                    color: white;
                    border-radius: 50%;
                }
                .banker {
                    background-color: #e53e3e;
                    color: white;
                    border-radius: 50%;
                }
                .tie {
                    background-color: #38a169;
                    color: white;
                    border-radius: 50%;
                }
                .empty {
                    background-color: #f7fafc;
                }
                /* Responsive adjustments */
                @media screen and (max-width: 600px) {
                    .bead-plate-table td {
                        width: 24px;
                        height: 24px;
                        min-width: 24px;
                        font-size: 12px;
                    }
                }
                @media screen and (max-width: 400px) {
                    .bead-plate-table td {
                        width: 20px;
                        height: 20px;
                        min-width: 20px;
                        font-size: 10px;
                    }
                }
            </style>
            <div class='bead-plate-container'>
                <table class='bead-plate-table'>
            """

            for row in range(rows):
                html += "<tr>"
                for col in range(cols):
                    result = grid[row][col]
                    if result == 'P':
                        html += "<td class='player'>P</td>"
                    elif result == 'B':
                        html += "<td class='banker'>B</td>"
                    elif result == 'T':
                        html += "<td class='tie'>T</td>"
                    else:
                        html += "<td class='empty'></td>"
                html += "</tr>"
            html += """
                </table>
            </div>
            """

            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering bead plate: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Baccarat Predictor and Tracker", layout="wide")
    st.title("Baccarat Predictor and Tracker")
    initialize_session_state()
    render_session_setup()
    render_result_input()
    render_bead_plate()
    render_prediction()
    render_status()

if __name__ == "__main__":
    main()
