import streamlit as st
import logging
import math

# Existing PATTERN_WEIGHTS and other functions (e.g., normalize, pattern detection) remain unchanged
# ...

def money_management(bankroll, base_bet, strategy, t3_level=1, t3_results=None, bet_outcome=None, loss_streak=0, skip_betting=False, skip_hands_count=0):
    if t3_results is None:
        t3_results = []
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if bankroll < min_bet:
        logging.warning(f"Bankroll ({bankroll:.2f}) is less than minimum bet ({min_bet:.2f}).")
        return 0.0, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count

    if strategy == "T3":
        if skip_betting:
            # Resume betting if a win occurs or after 10 hands
            if 'latest_result' in st.session_state and st.session_state.latest_result == 'W' or skip_hands_count >= 10:
                skip_betting = False
                skip_hands_count = 0
                t3_results = []
                t3_level = max(1, t3_level - 1)  # Reduce level when resuming
                st.session_state.latest_result = None  # Clear result after resuming
            else:
                skip_hands_count += 1
                return 0.0, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count

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
                skip_hands_count = 1
                return 0.0, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count
        elif bet_outcome is None:  # Handle initial or skipped bets
            pass  # No change to T3 state

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
        skip_hands_count = 0

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2), t3_level, t3_results, loss_streak, skip_betting, skip_hands_count

@st.cache_data
def calculate_bankroll(history, base_bet, strategy, initial_bankroll, ai_mode, _t3_state=0):
    bankroll = initial_bankroll
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    t3_level = 1
    t3_results = []
    loss_streak = 0
    skip_betting = False
    skip_hands_count = 0
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, pattern_insights = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        # Update latest result for skip logic
        if strategy == "T3":
            tracker_result = calculate_win_loss_tracker(current_rounds, base_bet, strategy, ai_mode, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count)[-1]
            st.session_state.latest_result = tracker_result
        if bet in (None, 'Pass', 'Tie') or skip_betting:
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            if skip_betting:
                skip_hands_count += 1
            continue
        bet_size, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
            current_bankroll, base_bet, strategy, t3_level, t3_results, loss_streak=loss_streak, skip_betting=skip_betting, skip_hands_count=skip_hands_count
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
                bet_size, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                    current_bankroll, base_bet, strategy, t3_level, t3_results, 'win', loss_streak, skip_betting, skip_hands_count
                )
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.pattern_performance)
            st.session_state.latest_result = 'W'
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            st.session_state.latest_result = 'T'
            continue
        else:
            current_bankroll -= bet_size
            if strategy == "T3":
                bet_size, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                    current_bankroll, base_bet, strategy, t3_level, t3_results, 'loss', loss_streak, skip_betting, skip_hands_count
                )
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.pattern_performance)
            st.session_state.latest_result = 'L'
        bankroll_progress.append(current_bankroll)
    return bankroll_progress, bet_sizes

def calculate_win_loss_tracker(history, base_bet, strategy, ai_mode, t3_level=1, t3_results=None, loss_streak=0, skip_betting=False, skip_hands_count=0):
    if t3_results is None:
        t3_results = []
    tracker = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, _, _, _, _ = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if actual_result == 'Tie':
            tracker.append('T')
        elif bet in (None, 'Pass') or skip_betting:
            tracker.append('S')
            if skip_betting and strategy == "T3":
                skip_hands_count += 1
                if st.session_state.get('latest_result') == 'W' or skip_hands_count >= 10:
                    skip_betting = False
                    skip_hands_count = 0
                    t3_results = []
                    t3_level = max(1, t3_level - 1)
        elif actual_result == bet:
            tracker.append('W')
            if strategy == "T3":
                _, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                    st.session_state.initial_bankroll, base_bet, strategy, t3_level, t3_results, 'win', loss_streak, skip_betting, skip_hands_count
                )
        else:
            tracker.append('L')
            if strategy == "T3":
                _, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                    st.session_state.initial_bankroll, base_bet, strategy, t3_level, t3_results, 'loss', loss_streak, skip_betting, skip_hands_count
                )
    return tracker

def main():
    try:
        # ... (Previous main function code up to session state initialization)

        # Initialize additional session state for T3
        if 'latest_result' not in st.session_state:
            st.session_state.latest_result = None
        if 'loss_streak' not in st.session_state:
            st.session_state.loss_streak = 0
        if 'skip_betting' not in st.session_state:
            st.session_state.skip_betting = False
        if 'skip_hands_count' not in st.session_state:
            st.session_state.skip_hands_count = 0

        # ... (Rest of the main function up to Game Settings)

        # Game Settings
        with st.expander("Game Settings", expanded=False):
            cols = st.columns(4)
            with cols[0]:
                initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
            with cols[1]:
                base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
            with cols[2]:
                strategy_options = ["Flat Betting", "T3"]
                money_management_strategy = st.selectbox("Money Management Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
                st.markdown("*Flat Betting: Fixed bet size. T3: Adjusts bet level based on the last three bet outcomes (increase if more losses, decrease if more wins or first-step win). Skips betting after three consecutive losses until a win or 10 hands.*")
            with cols[3]:
                ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

            st.session_state.initial_bankroll = initial_bankroll
            st.session_state.base_bet = base_bet
            st.session_state.money_management_strategy = money_management_strategy
            st.session_state.ai_mode = ai_mode

            st.markdown(f"**Selected Strategy: {money_management_strategy}**")

        # ... (Input Game Results, Shoe Patterns, etc., remain unchanged)

        # Prediction
        with st.expander("Prediction", expanded=True):
            bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
            st.markdown("### Prediction")
            current_bankroll = calculate_bankroll(
                st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy,
                st.session_state.initial_bankroll, st.session_state.ai_mode,
                _t3_state=hash((st.session_state.t3_level, tuple(st.session_state.t3_results), st.session_state.loss_streak, st.session_state.skip_betting, st.session_state.skip_hands_count))
            )[0][-1] if st.session_state.history else st.session_state.initial_bankroll
            recommended_bet_size, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy,
                st.session_state.t3_level, st.session_state.t3_results,
                loss_streak=st.session_state.loss_streak, skip_betting=st.session_state.skip_betting, skip_hands_count=st.session_state.skip_hands_count
            )
            st.session_state.t3_level = t3_level
            st.session_state.t3_results = t3_results
            st.session_state.loss_streak = loss_streak
            st.session_state.skip_betting = skip_betting
            st.session_state.skip_hands_count = skip_hands_count
            if current_bankroll < max(1.0, st.session_state.base_bet):
                st.warning("Insufficient bankroll to place a bet. Please increase your bankroll or reset the game.")
                bet = 'Pass'
                confidence = 0
                reason = "Bankroll too low to continue betting."
                emotional_tone = "Cautious"
            elif st.session_state.skip_betting:
                skip_reason = f"Skipping betting after three consecutive losses. Waiting for a win or {10 - st.session_state.skip_hands_count} more hands. (Hands skipped: {st.session_state.skip_hands_count})"
                st.markdown(f"**No Bet**: {skip_reason}")
                if st.session_state.money_management_strategy == "T3":
                    st.markdown(f"**T3 Level**: {st.session_state.t3_level} | **Recent Outcomes**: {''.join(st.session_state.t3_results)}")
            elif bet == 'Pass':
                st.markdown("**No Bet**: Insufficient confidence to place a bet.")
                if st.session_state.money_management_strategy == "T3":
                    st.markdown(f"**T3 Level**: {st.session_state.t3_level} | **Recent Outcomes**: {''.join(st.session_state.t3_results)}")
            else:
                st.markdown(f"**Bet**: {bet} | **Confidence**: {confidence}% | **Bet Size**: ${recommended_bet_size:.2f} | **Mood**: {emotional_tone}")
                if st.session_state.money_management_strategy == "T3":
                    st.markdown(f"**T3 Level**: {st.session_state.t3_level} | **Recent Outcomes**: {''.join(st.session_state.t3_results)}")
            st.markdown(f"**Reasoning**: {reason}")
            if pattern_insights:
                st.markdown("### Pattern Insights")
                st.markdown("Detected patterns influencing the prediction:")
                for insight in pattern_insights:
                    st.markdown(f"- {insight}")

        # ... (Pattern Performance, Bankroll Progress remain unchanged)

        # Reset
        with st.expander("Reset", expanded=False):
            if st.button("New Game"):
                st.session_state.confirm_reset = True
            if st.session_state.confirm_reset:
                cols = st.columns(2)
                with cols[0]:
                    if st.button("Confirm Reset"):
                        final_bankroll = calculate_bankroll(
                            st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy,
                            st.session_state.initial_bankroll, st.session_state.ai_mode,
                            _t3_state=hash((st.session_state.t3_level, tuple(st.session_state.t3_results), st.session_state.loss_streak, st.session_state.skip_betting, st.session_state.skip_hands_count))
                        )[0][-1] if st.session_state.history else st.session_state.initial_bankroll
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
                        st.session_state.skip_hands_count = 0
                        st.rerun()
                with cols[1]:
                    if st.button("Cancel"):
                        st.session_state.confirm_reset = False
                        st.rerun()

    except (KeyError, ValueError, IndexError) as e:
        logging.error(f"Error in main: {str(e)}")
        st.error(f"Error occurred: {str(e)}. Please try refreshing the page or resetting the game.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        st.error(f"Unexpected error: {str(e)}. Contact support if this persists.")

# ... (Rest of the code remains unchanged)
