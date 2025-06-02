import streamlit as st

def initialize_session_state():
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 0.0
        st.session_state.base_bet = 0.0
        st.session_state.initial_bankroll = 0.0
        st.session_state.current_position = 0
        st.session_state.wins = 0
        st.session_state.losses = 0
        st.session_state.total_bets = 0
        st.session_state.bet_history = []
        st.session_state.t3_level = 1
        st.session_state.t3_results = []
        st.session_state.session_active = False
        st.session_state.sequence = ['P', 'B', 'P', 'P', 'B', 'B']
        st.session_state.stop_loss = 0.8  # Stop at 80% of initial bankroll
        st.session_state.win_limit = 1.5  # Stop at 150% of initial bankroll
        st.session_state.outcome = "-"

def start_session():
    bankroll = st.session_state.bankroll_input
    base_bet = st.session_state.base_bet_input
    if bankroll <= 0 or base_bet <= 0:
        st.error("Bankroll and base bet must be positive numbers.")
        return
    if base_bet > bankroll * 0.05:
        st.error("Base bet cannot exceed 5% of bankroll.")
        return
    st.session_state.bankroll = bankroll
    st.session_state.base_bet = base_bet
    st.session_state.initial_bankroll = bankroll
    st.session_state.current_position = 0
    st.session_state.wins = 0
    st.session_state.losses = 0
    st.session_state.total_bets = 0
    st.session_state.bet_history = []
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.session_active = True
    st.session_state.outcome = "-"
    st.info(f"Session Started! Bankroll: ${bankroll:.2f}, Base Bet: ${base_bet:.2f}")

def reset_session():
    initialize_session_state()
    st.session_state.bankroll_input = 0.0
    st.session_state.base_bet_input = 0.0
    st.info("Session Reset. Enter new bankroll and base bet to start.")

def undo_result():
    if not st.session_state.bet_history:
        st.warning("No results to undo.")
        return
    last_bet = st.session_state.bet_history.pop()
    result, bet_amount, bet_selection, bet_outcome, prev_position, prev_t3_level, prev_t3_results = last_bet
    st.session_state.current_position = prev_position
    st.session_state.t3_level = prev_t3_level
    st.session_state.t3_results = prev_t3_results[:]
    if bet_outcome == 'win':
        st.session_state.wins -= 1
        if bet_selection == 'B':
            st.session_state.bankroll -= bet_amount * 0.95  # Reverse Banker win (95% payout)
        else:
            st.session_state.bankroll -= bet_amount  # Reverse Player win
        st.session_state.total_bets -= 1
    elif bet_outcome == 'loss':
        st.session_state.losses -= 1
        st.session_state.bankroll += bet_amount  # Reverse loss
        st.session_state.total_bets -= 1
    st.session_state.outcome = "-"
    st.info(f"Undid last result: {result}")

def process_result(outcome):
    if st.session_state.bankroll <= 0:
        st.warning("Bankroll is depleted. Reset or exit.")
        st.session_state.session_active = False
        return
    if st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss:
        st.warning(f"Bankroll below {st.session_state.stop_loss*100:.0f}% of initial. Session ended.")
        st.session_state.session_active = False
        return
    if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
        st.info(f"Bankroll above {st.session_state.win_limit*100:.0f}% of initial. Session ended.")
        st.session_state.session_active = False
        return

    current_bet = st.session_state.sequence[st.session_state.current_position]
    bet_amount = st.session_state.base_bet * st.session_state.t3_level
    bet_selection = current_bet
    bet_outcome = None

    if bet_amount > st.session_state.bankroll:
        st.warning(f"Bet ${bet_amount:.2f} exceeds bankroll. Reset or exit.")
        st.session_state.session_active = False
        return

    st.session_state.outcome = outcome

    if outcome == 'T':
        st.session_state.bet_history.append((outcome, 0, bet_selection, 'tie', st.session_state.current_position, st.session_state.t3_level, st.session_state.t3_results[:]))
        st.info("Tie! Bet remains unchanged.")
    else:
        st.session_state.total_bets += 1
        if current_bet == outcome:
            st.session_state.wins += 1
            bet_outcome = 'win'
            if outcome == 'B':
                st.session_state.bankroll += bet_amount * 0.95  # Banker win with 5% commission
            else:
                st.session_state.bankroll += bet_amount  # Player win
            st.session_state.current_position = 0  # Reset to start of sequenceThe 
            if st.session_state.current_position == 0 and len(st.session_state.t3_results) == 0:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            st.session_state.t3_results.append('W')
            st.info(f"Win! +${bet_amount * (0.95 if outcome == 'B' else 1):.2f}")
        else:
            st.session_state.losses += 1
            bet_outcome = 'loss'
            st.session_state.bankroll -= bet_amount  # Deduct bet
            st.session_state.current_position = (st.session_state.current_position + 1) % len(st.session_state.sequence)
            st.session_state.t3_results.append('L')
            st.info(f"Loss! -${bet_amount:.2f}")

        if len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins > losses:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses > wins:
                st.session_state.t3_level += 1
            st.session_state.t3_results = []

        st.session_state.bet_history.append((outcome, bet_amount, bet_selection, bet_outcome, st.session_state.current_position, st.session_state.t3_level, st.session_state.t3_results[:]))

def generate_bead_plate():
    rows, cols = 6, 10
    html = """
    <style>
        .bead-plate {
            display: grid;
            grid-template-columns: repeat(10, 30px);
            gap: 2px;
            margin: 10px 0;
        }
        .bead-cell {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }
        .player { background-color: blue; }
        .banker { background-color: red; }
        .tie { background-color: green; }
        .empty { background-color: gray; }
    </style>
    <div class="bead-plate">
    """
    outcomes = [bet[0] for bet in st.session_state.bet_history]  # Get result from each bet
    grid = [['' for _ in range(cols)] for _ in range(rows)]
    for i, outcome in enumerate(outcomes):
        row = i % rows
        col = i // rows
        if col < cols:  # Ensure we don't exceed columns
            grid[row][col] = outcome
    for row in range(rows):
        for col in range(cols):
            outcome = grid[row][col]
            if outcome == 'P':
                html += '<div class="bead-cell player">P</div>'
            elif outcome == 'B':
                html += '<div class="bead-cell banker">B</div>'
            elif outcome == 'T':
                html += '<div class="bead-cell tie">T</div>'
            else:
                html += '<div class="bead-cell empty"></div>'
    html += "</div>"
    return html

def generate_prediction_display():
    if not st.session_state.session_active:
        return "<div>-</div>"
    bet = st.session_state.sequence[st.session_state.current_position]
    amount = st.session_state.base_bet * st.session_state.t3_level
    bet_name = "Player" if bet == 'P' else "Banker"
    color = "blue" if bet == 'P' else "red"
    html = f"""
    <style>
        .prediction {{
            background-color: {color};
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin: 10px 0;
        }}
    </style>
    <div class="prediction">{bet_name} (${amount:.2f})</div>
    """
    return html

def quit_game():
    win_rate = (st.session_state.wins / st.session_state.total_bets * 100) if st.session_state.total_bets > 0 else 0
    st.info(f"Game Over\nTotal Bets: {st.session_state.total_bets}\nWins: {st.session_state.wins}\nLosses: {st.session_state.losses}\nWin Rate: {win_rate:.2f}%\nFinal Bankroll: ${st.session_state.bankroll:.2f}")
    st.session_state.session_active = False
    initialize_session_state()

def main():
    st.title("Baccarat Predictor with T3")
    st.write("Sequence: P, B, P, P, B, B")

    initialize_session_state()

    st.session_state.bankroll_input = st.number_input("Enter Initial Bankroll ($):", min_value=0.0, step=10.0, value=st.session_state.bankroll_input if 'bankroll_input' in st.session_state else 0.0)
    st.session_state.base_bet_input = st.number_input("Enter Base Bet ($):", min_value=0.0, step=1.0, value=st.session_state.base_bet_input if 'base_bet_input' in st.session_state else 0.0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Session", type="primary"):
            start_session()
    with col2:
        if st.button("Reset Session"):
            reset_session()

    if st.session_state.session_active:
        st.markdown(f"**Current Bet:** {st.session_state.sequence[st.session_state.current_position]} (${st.session_state.base_bet * st.session_state.t3_level:.2f})")
        st.markdown(f"**Outcome:** {st.session_state.outcome}")
        
        col3, col4, col5 = st.columns(3)
        with col3:
            if st.button("Player"):
                process_result('P')
        with col4:
            if st.button("Banker"):
                process_result('B')
        with col5:
            if st.button("Tie"):
                process_result('T')

        if st.button("Undo Last Result"):
            undo_result()

    win_rate = (st.session_state.wins / st.session_state.total_bets * 100) if st.session_state.total_bets > 0 else 0
    st.markdown(f"**Stats:** Wins: {st.session_state.wins} | Losses: {st.session_state.losses} | Win Rate: {win_rate:.2f}%")
    st.markdown(f"**Bankroll:** ${st.session_state.bankroll:.2f}")
    st.markdown(f"**Base Bet:** ${st.session_state.base_bet:.2f}")
    st.markdown(f"**T3 Status:** Level {st.session_state.t3_level}, Results: {st.session_state.t3_results}")
    
    st.markdown("**Next Predicted Bet:**")
    st.markdown(generate_prediction_display(), unsafe_allow_html=True)
    
    st.markdown("**Bead Plate (Vertical):**")
    st.markdown(generate_bead_plate(), unsafe_allow_html=True)

    if st.button("Quit"):
        quit_game()

if __name__ == "__main__":
    main()
