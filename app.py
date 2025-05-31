import streamlit as st
import logging
import math

# Existing PATTERN_WEIGHTS and other functions remain unchanged
# ...

def money_management(bankroll, base_bet, strategy, t3_level=1, t3_results=None, bet_outcome=None, loss_streak=0, skip_betting=False):
    if t3_results is None:
        t3_results = []
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if bankroll < min_bet:
        logging.warning(f"Bankroll ({bankroll:.2f}) is less than minimum bet ({min_bet:.2f}).")
        return 0.0, t3_level, t3_results, loss_streak, skip_betting

    if strategy == "T3":
        if skip_betting:
            # Check if a win has occurred in the latest history to resume betting
            if 'latest_result' in st.session_state and st.session_state.latest_result == 'W':
                skip_betting = False
                t3_results = []  # Reset results to start fresh after resuming
                t3_level = max(1, t3_level - 1)  # Reduce level after resuming
            else:
                return 0.0, t3_level, t3_results, loss_streak, skip_betting

        if bet_outcome == 'win':
            t3_results.append('W')
            loss_streak = 0
            if not t3_results or len(t3_results) == 1:
                t3_level = max(1, t3_level - 1)
        elif bet_outcome == 'loss':
            t3_results.append('L')
            loss_streak += 1
            if loss_streak >= 3:
                skip_betting = True
                return 0.0, t3_level, t3_results, loss_streak, skip_betting

        if len(t3_results) == 3 and not skip_betting:
            wins = t3_results.count('W')
            losses = t3_results.count('L')
            if wins > losses:
                t3_level = max(1, t3_level - 1)
            elif losses > wins:
                t3_level += 1
            t3_results = []

        calculated_bet = base_bet * t3_level
    else:
        calculated_bet = base_bet
        loss_streak = 0
        skip_betting = False

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2), t3_level, t3_results, loss_streak, skip_betting

@st.cache_data
def calculate_bankroll(history, base_bet, strategy, initial_bankroll, ai_mode):
    bankroll = initial_bankroll
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    t3_level = 1
    t3_results = []
    loss_streak = 0
    skip_betting = False
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, pattern_insights = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie') or skip_betting:
            # Update latest result for skip logic
            if strategy == "T3":
                tracker_result = calculate_win_loss_tracker(current_rounds, base_bet, strategy, ai_mode)[-1]
                st.session_state.latest_result = tracker_result
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_size, t3_level, t3_results, loss_streak, skip_betting = money_management(
            current_bankroll, base_bet, strategy, t3_level, t3_results, loss_streak=loss_streak, skip_betting=skip_betting
        )
        if bet_size == 0.0:
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_sizes.append(bet_size)
        if actual_result == bet:
            if bet == 'Banker':
                win_amount = bet_size * 0.95
                current_bankroll += win_amount
            else:
                current_bankroll += bet_size
            if strategy == "T3":
                bet_size, t3_level, t3_results, loss_streak, skip_betting = money_management(
                    current_bankroll, base_bet, strategy, t3_level, t3_results, 'win', loss_streak, skip_betting
                )
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.pattern_performance)
            st.session_state.latest_result = 'W'
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            st.session_state.latest_result = 'T'
            continue
        else:
            current_bankroll -= bet_size
            if strategy == "T3":
                bet_size, t3_level, t3_results, loss_streak, skip_betting = money_management(
                    current_bankroll, base_bet, strategy, t3_level, t3_results, 'loss', loss_streak, skip_betting
                )
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.pattern_performance)
            st.session_state.latest_result = 'L'
        bankroll_progress.append(current_bankroll)
    return bankroll_progress, bet_sizes

def calculate_win_loss_tracker(history, base_bet, strategy, ai_mode):
    tracker = []
    t3_level = 1
    t3_results = []
    loss_streak = 0
    skip_betting = False
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, _, _, _, _ = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if actual_result == 'Tie':
            tracker.append('T')
        elif bet in (None, 'Pass') or skip_betting:
            tracker.append('S')
        elif actual_result == bet:
            tracker.append('W')
            if strategy == "T3":
                _, t3_level, t3_results, loss_streak, skip_betting = money_management(
                    st.session_state.initial_bankroll, base_bet, strategy, t3_level, t3_results, 'win', loss_streak, skip_betting
                )
        else:
            tracker.append('L')
            if strategy == "T3":
                _, t3_level, t3_results, loss_streak, skip_betting = money_management(
                    st.session_state.initial_bankroll, base_bet, strategy, t3_level, t3_results, 'loss', loss_streak, skip_betting
                )
    return tracker

def main():
    # ... (Previous main function code up to session state initialization)
    
    # Initialize additional session state for T3
    if 'latest_result' not in st.session_state:
        st.session_state.latest_result = None
    if 'loss_streak' not in st.session_state:
        st.session_state.loss_streak = 0
    if 'skip_betting' not in st.session_state:
        st.session_state.skip_betting = False

    # ... (Rest of the main function up to Prediction section)

    # Prediction
    with st.expander("Prediction", expanded=True):
        bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
        st.markdown("### Prediction")
        current_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.initial_bankroll, st.session_state.ai_mode)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
        recommended_bet_size, t3_level, t3_results, loss_streak, skip_betting = money_management(
            current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy,
            st.session_state.t3_level, st.session_state.t3_results, loss_streak=st.session_state.loss_streak, skip_betting=st.session_state.skip_betting
        )
        st.session_state.t3_level = t3_level
        st.session_state.t3_results = t3_results
        st.session_state.loss_streak = loss_streak
        st.session_state.skip_betting = skip_betting
        if current_bankroll < max(1.0, st.session_state.base_bet):
            st.warning("Insufficient bankroll to place a bet. Please increase your bankroll or reset the game.")
            bet = 'Pass'
            confidence = 0
            reason = "Bankroll too low to continue betting."
            emotional_tone = "Cautious"
        elif st.session_state.skip_betting:
            st.markdown("**No Bet**: Skipping due to three consecutive losses. Waiting for a win.")
        elif bet == 'Pass':
            st.markdown("**No Bet**: Insufficient confidence to place a bet.")
        else:
            st.markdown(f"**Bet**: {bet} | **Confidence**: {confidence}% | **Bet Size**: ${recommended_bet_size:.2f} | **Mood**: {emotional_tone}")
        st.markdown(f"**Reasoning**: {reason}")
        if pattern_insights:
            st.markdown("### Pattern Insights")
            st.markdown("Detected patterns influencing the prediction:")
            for insight in pattern_insights:
                st.markdown(f"- {insight}")

    # ... (Rest of the main function, including Pattern Performance, Bankroll Progress, and Reset sections)
    
    # Update Reset section to reset new session state variables
    with st.expander("Reset", expanded=False):
        if st.button("New Game"):
            st.session_state.confirm_reset = True
        if st.session_state.confirm_reset:
            cols = st.columns(2)
            with cols[0]:
                if st.button("Confirm Reset"):
                    final_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.initial_bankroll, st.session_state.ai_mode)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
                    st.session_state.history = []
                    st.session_state.initial_bankroll = max(1.0, final_bankroll)
                    st.session_state.base_bet = min(10.0, st.session_state.initial_bankroll)
                    st.session_state.money_management_strategy = "Flat Betting"
                    st.session_state.ai_mode = "Conservative"
                    st.session_state.selected_patterns = ["Bead Bin", "Win/Loss"]
                    st.session_state.t3_level = 1
                    st.session_state.t3_results = []
                    st.session_state.pattern_performance = {}
                    st.session_state.confirm_reset = False
                    st.session_state.latest_result = None
                    st.session_state.loss_streak = 0
                    st.session_state.skip_betting = False
                    st.rerun()
            with cols[1]:
                if st.button("Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()

# ... (Rest of the code remains unchanged)
