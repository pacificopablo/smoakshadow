import streamlit as st
import uuid
from collections import defaultdict, Counter

def initialize_session_state():
    """Initialize session state with default values, including target profit settings."""
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
        st.session_state.transition_counts = {'PP': 0, 'PB': 0, 'PT': 0, 'BP': 0, 'BB': 0, 'BT': 0, 'TP': 0, 'TB': 0, 'TT': 0}
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.bet = 1
        st.session_state.bankroll = None
        st.session_state.initial_bankroll = None  # Track initial bankroll for profit calculation
        st.session_state.base_bet = None
        st.session_state.last_prediction = None
        st.session_state.strategy = "Flat Bet"
        st.session_state.masterline_step = 0
        st.session_state.in_force2 = False
        st.session_state.force2_failed = False
        st.session_state.break_countdown = 0
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.transitions = defaultdict(Counter)
        st.session_state.explanation = "No explanation available."
        st.session_state.session_started = False
        st.session_state.target_profit_mode = "None"  # "None", "Percentage", or "Units"
        st.session_state.target_profit_percent = 10.0  # Default 10%
        st.session_state.target_profit_units = 100.0  # Default $100
        st.session_state.target_reached = False  # Flag to pause session when target is met

def reset_session():
    """Reset session state to initial values, including target profit settings."""
    st.session_state.update({
        'sequence': [],
        'transition_counts': {'PP': 0, 'PB': 0, 'PT': 0, 'BP': 0, 'BB': 0, 'BT': 0, 'TP': 0, 'TB': 0, 'TT': 0},
        'session_id': str(uuid.uuid4()),
        'bet': 1,
        'bankroll': None,
        'initial_bankroll': None,
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
        'session_started': False,
        'target_profit_mode': "None",
        'target_profit_percent': 10.0,
        'target_profit_units': 100.0,
        'target_reached': False
    })

def place_result(result):
    """Append a game result, update transition counts, handle betting logic, and check target profit."""
    try:
        if st.session_state.target_reached:
            st.warning("Target profit reached! Session paused. Reset to continue.")
            return

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

        # Check for insufficient bankroll
        if st.session_state.bankroll < actual_bet and result in ("P", "B"):
            st.error("Insufficient bankroll to place the bet! Please reset the session or adjust settings.")
            return

        # Update bankroll based on strategy
        if result in ("P", "B") and st.session_state.last_prediction in ("P", "B"):
            commission = 0.95 if result == "B" else 1.0
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

        # Check if target profit is reached
        if st.session_state.initial_bankroll is not None:
            current_profit = st.session_state.bankroll - st.session_state.initial_bankroll
            if st.session_state.target_profit_mode == "Percentage":
                target_profit = st.session_state.initial_bankroll * (st.session_state.target_profit_percent / 100)
                if current_profit >= target_profit:
                    st.session_state.target_reached = True
                    st.success(f"Target profit of {st.session_state.target_profit_percent}% (${target_profit:.2f}) reached! Session paused.")
            elif st.session_state.target_profit_mode == "Units":
                if current_profit >= st.session_state.target_profit_units:
                    st.session_state.target_reached = True
                    st.success(f"Target profit of ${st.session_state.target_profit_units:.2f} reached! Session paused.")

        # Update prediction
        prediction, explanation = predict_next()
        st.session_state.last_prediction = prediction if prediction in ("P", "B") else None
        st.session_state.explanation = explanation
    except Exception as e:
        st.error(f"Error processing result: {e}")

def render_session_setup():
    """Render UI for starting, resetting a session, selecting strategy, and target profit settings."""
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
            target_profit_mode = st.selectbox(
                "Target Profit Mode",
                ["None", "Percentage", "Units"],
                index=["None", "Percentage", "Units"].index(st.session_state.target_profit_mode)
            )
            target_profit_percent = st.number_input(
                "Target Profit (% of Bankroll)",
                min_value=0.1,
                value=st.session_state.target_profit_percent,
                step=0.1,
                format="%.1f",
                disabled=target_profit_mode != "Percentage"
            )
            target_profit_units = st.number_input(
                "Target Profit (Units in $)",
                min_value=0.1,
                value=st.session_state.target_profit_units,
                step=0.1,
                format="%.2f",
                disabled=target_profit_mode != "Units"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Session", disabled=bankroll < base_bet):
                    if bankroll < base_bet:
                        st.error("Initial bankroll must be at least equal to the base bet.")
                        return
                    if target_profit_mode == "Units" and target_profit_units > bankroll:
                        st.error("Target profit in units cannot exceed initial bankroll.")
                        return
                    st.session_state.bankroll = bankroll
                    st.session_state.initial_bankroll = bankroll
                    st.session_state.base_bet = base_bet
                    st.session_state.strategy = strategy
                    st.session_state.target_profit_mode = target_profit_mode
                    st.session_state.target_profit_percent = target_profit_percent
                    st.session_state.target_profit_units = target_profit_units
                    st.session_state.bet = 1
                    st.session_state.t3_level = 1
                    st.session_state.t3_results = []
                    st.session_state.target_reached = False
                    st.session_state.session_started = True
                    st.success("Session started!")
            with col2:
                if st.session_state.session_started and st.button("Reset Session"):
                    reset_session()
                    st.success("Session reset!")
        except Exception as e:
            st.error(f"Error setting up session: {e}")

def render_status():
    """Render session status with hands played, streak status, betting info, and target profit progress."""
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
            if st.session_state.initial_bankroll is not None:
                profit = st.session_state.bankroll - st.session_state.initial_bankroll
                st.markdown(f"**Current Profit**: ${profit:.2f}")
            status = "Normal"
            if st.session_state.target_reached:
                status = "Target Profit Reached"
            elif st.session_state.break_countdown > 0:
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
            # Display target profit settings
            st.markdown(f"**Target Profit Mode**: {st.session_state.target_profit_mode}")
            if st.session_state.target_profit_mode == "Percentage" and st.session_state.initial_bankroll is not None:
                target_profit = st.session_state.initial_bankroll * (st.session_state.target_profit_percent / 100)
                st.markdown(f"**Target Profit**: {st.session_state.target_profit_percent}% (${target_profit:.2f})")
            elif st.session_state.target_profit_mode == "Units":
                st.markdown(f"**Target Profit**: ${st.session_state.target_profit_units:.2f}")
        except Exception as e:
            st.error(f"Error rendering status: {e}")

# The rest of the functions (handle_masterline, handle_t3, predict_next, render_result_input, render_prediction, render_bead_plate, main) remain unchanged.
