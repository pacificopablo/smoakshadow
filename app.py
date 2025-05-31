import streamlit as st
import logging
import plotly.graph_objects as go
import math
import hashlib

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Existing PATTERN_WEIGHTS and other functions (e.g., normalize, pattern detection) remain unchanged
# ...

def money_management(bankroll, base_bet, strategy, t3_level=1, t3_results=None, bet_outcome=None, loss_streak=0, skip_betting=False, skip_hands_count=0):
    logging.debug(f"money_management called: bankroll={bankroll}, strategy={strategy}, t3_level={t3_level}, bet_outcome={bet_outcome}, loss_streak={loss_streak}, skip_betting={skip_betting}, skip_hands_count={skip_hands_count}")
    if t3_results is None:
        t3_results = []
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if bankroll < min_bet:
        logging.warning(f"Bankroll ({bankroll:.2f}) is less than minimum bet ({min_bet:.2f}).")
        return 0.0, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count

    if strategy == "T3":
        if skip_betting:
            latest_result = st.session_state.get('latest_result')
            logging.debug(f"Skipping: latest_result={latest_result}, skip_hands_count={skip_hands_count}")
            if latest_result == 'W' or skip_hands_count >= 10:
                skip_betting = False
                skip_hands_count = 0
                t3_results = []
                t3_level = max(1, t3_level - 1)
                st.session_state['latest_result'] = None
                logging.info("Resuming betting after win or timeout")
            else:
                skip_hands_count += 1
                return 0.0, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count

        if bet_outcome == 'win':
            t3_results.append('W')
            loss_streak = 0
            if not t3_results or len(t3_results) == 1:
                t3_level = max(1, t3_level - 1)
            logging.debug(f"Win: t3_level={t3_level}, t3_results={t3_results}")
        elif bet_outcome == 'loss':
            t3_results.append('L')
            loss_streak += 1
            if loss_streak >= 3:
                skip_betting = True
                skip_hands_count = 1
                logging.info("Skipping betting after three consecutive losses")
                return 0.0, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count
        elif bet_outcome is None:
            pass

        if len(t3_results) == 3 and not skip_betting:
            wins = t3_results.count('W')
            losses = t3_results.count('L')
            if wins > losses:
                t3_level = max(1, t3_level - 1)
            elif losses > wins:
                t3_level += 1
            t3_results = []
            logging.debug(f"T3 adjustment: t3_level={t3_level}, t3_results={t3_results}")

        calculated_bet = base_bet * t3_level
    else:
        calculated_bet = base_bet
        loss_streak = 0
        skip_betting = False
        skip_hands_count = 0

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    logging.debug(f"Bet size calculated: {bet_size}")
    return round(bet_size, 2), t3_level, t3_results, loss_streak, skip_betting, skip_hands_count

@st.cache_data
def calculate_bankroll(history, base_bet, strategy, initial_bankroll, ai_mode):
    logging.debug(f"calculate_bankroll called: history_len={len(history)}, strategy={strategy}")
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
        try:
            bet, confidence, _, _, pattern_insights = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        except Exception as e:
            logging.error(f"Error in advanced_bet_selection: {str(e)}")
            bet = 'Pass'
            confidence = 0
            pattern_insights = []
        actual_result = history[i]
        if strategy == "T3":
            try:
                tracker_result = calculate_win_loss_tracker(current_rounds, base_bet, strategy, ai_mode, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count)[-1]
                st.session_state['latest_result'] = tracker_result
            except Exception as e:
                logging.error(f"Error in calculate_win_loss_tracker: {str(e)}")
                st.session_state['latest_result'] = None
        if bet in (None, 'Pass', 'Tie') or skip_betting:
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            if skip_betting:
                skip_hands_count += 1
            continue
        try:
            bet_size, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                current_bankroll, base_bet, strategy, t3_level, t3_results, loss_streak=loss_streak, skip_betting=skip_betting, skip_hands_count=skip_hands_count
            )
        except Exception as e:
            logging.error(f"Error in money_management: {str(e)}")
            bet_size = 0.0
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
                try:
                    bet_size, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                        current_bankroll, base_bet, strategy, t3_level, t3_results, 'win', loss_streak, skip_betting, skip_hands_count
                    )
                except Exception as e:
                    logging.error(f"Error in money_management (win): {str(e)}")
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.pattern_performance)
            st.session_state['latest_result'] = 'W'
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            st.session_state['latest_result'] = 'T'
            continue
        else:
            current_bankroll -= bet_size
            if strategy == "T3":
                try:
                    bet_size, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                        current_bankroll, base_bet, strategy, t3_level, t3_results, 'loss', loss_streak, skip_betting, skip_hands_count
                    )
                except Exception as e:
                    logging.error(f"Error in money_management (loss): {str(e)}")
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.pattern_performance)
            st.session_state['latest_result'] = 'L'
        bankroll_progress.append(current_bankroll)
    return bankroll_progress, bet_sizes

def calculate_win_loss_tracker(history, base_bet, strategy, ai_mode, t3_level=1, t3_results=None, loss_streak=0, skip_betting=False, skip_hands_count=0):
    if t3_results is None:
        t3_results = []
    tracker = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        try:
            bet, _, _, _, _ = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        except Exception as e:
            logging.error(f"Error in advanced_bet_selection (tracker): {str(e)}")
            bet = 'Pass'
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
                try:
                    _, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                        st.session_state.initial_bankroll, base_bet, strategy, t3_level, t3_results, 'win', loss_streak, skip_betting, skip_hands_count
                    )
                except Exception as e:
                    logging.error(f"Error in money_management (tracker win): {str(e)}")
        else:
            tracker.append('L')
            if strategy == "T3":
                try:
                    _, t3_level, t3_results, loss_streak, skip_betting, skip_hands_count = money_management(
                        st.session_state.initial_bankroll, base_bet, strategy, t3_level, t3_results, 'loss', loss_streak, skip_betting, skip_hands_count
                    )
                except Exception as e:
                    logging.error(f"Error in money_management (tracker loss): {str(e)}")
    return tracker

def main():
    try:
        st.set_page_config(page_title="Mang Baccarat Predictor", page_icon="🎲", layout="wide")
        st.title("Mang Baccarat Predictor")
        logging.info("App initialized")

        # Initialize session state
        session_defaults = {
            'history': [],
            'initial_bankroll': 1000.0,
            'base_bet': 10.0,
            'money_management_strategy': "Flat Betting",
            'ai_mode': "Conservative",
            'selected_patterns': ["Bead Bin", "Win/Loss"],
            't3_level': 1,
            't3_results': [],
            'pattern_performance': {},
            'confirm_reset': False,
            'latest_result': None,
            'loss_streak': 0,
            'skip_betting': False,
            'skip_hands_count': 0,
            'screen_width': 1024
        }
        for key, value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        logging.debug("Session state initialized")

        # Simplified JavaScript
        st.markdown("""
            <script>
            function updateScreenWidth() {
                const width = window.innerWidth || 1024;
                const input = document.getElementById('screen-width-input');
                if (input) input.value = width;
            }
            function autoScrollPatterns() {
                const ids = ['bead-bin-scroll', 'big-road-scroll', 'big-eye-scroll', 'cockroach-scroll', 'win-loss-scroll'];
                ids.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.scrollLeft = el.scrollWidth;
                });
            }
            document.addEventListener('keydown', function(event) {
                const buttons = {
                    'p': 'player_button',
                    'P': 'player_button',
                    'b': 'banker_button',
                    'B': 'banker_button',
                    't': 'tie_button',
                    'T': 'tie_button',
                    'u': 'undo_button',
                    'U': 'undo_button'
                };
                const buttonId = buttons[event.key];
                if (buttonId) {
                    const btn = document.getElementById(buttonId);
                    if (btn && !btn.disabled) btn.click();
                }
            });
            try {
                updateScreenWidth();
                autoScrollPatterns();
                window.onresize = updateScreenWidth;
            } catch (e) {
                console.error('JavaScript error:', e);
            }
            </script>
            <input type="hidden" id="screen-width-input">
        """, unsafe_allow_html=True)
        logging.debug("JavaScript injected")

        screen_width_input = st.text_input("Screen Width", key="screen_width_input", value=str(st.session_state.screen_width), disabled=True)
        try:
            st.session_state.screen_width = int(screen_width_input) if screen_width_input.isdigit() else 1024
        except ValueError:
            st.session_state.screen_width = 1024

        # CSS
        st.markdown("""
            <style>
            .pattern-scroll {
                overflow-x: auto;
                white-space: nowrap;
                max-width: 100%;
                padding: 8px;
                border: 1px solid #e1e1e1;
                background-color: #f9f9f9;
            }
            .pattern-scroll::-webkit-scrollbar {
                height: 6px;
            }
            .pattern-scroll::-webkit-scrollbar-thumb {
                background-color: #888;
                border-radius: 3px;
            }
            .stButton > button {
                width: 100%;
                padding: 6px;
                margin: 4px 0;
            }
            .stNumberInput, .stSelectbox {
                width: 100%;
            }
            .pattern-circle {
                width: 20px;
                height: 20px;
                display: inline-block;
                margin: 2px;
                border-radius: 50%;
            }
            .display-circle {
                width: 20px;
                height: 20px;
                display: inline-block;
                margin: 2px;
            }
            @media (max-width: 768px) {
                .pattern-circle, .display-circle {
                    width: 14px;
                    height: 14px;
                }
                .stButton > button {
                    font-size: 0.85rem;
                    padding: 5px;
                }
                h1 { font-size: 1.8rem; }
                h3 { font-size: 1.2rem; }
                p, div, span { font-size: 0.9rem; }
            }
            </style>
        """, unsafe_allow_html=True)

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
            logging.debug("Game settings updated")

        # Pattern Weights
        with st.expander("Pattern Weights", expanded=False):
            for pattern, weight in PATTERN_WEIGHTS.items():
                PATTERN_WEIGHTS[pattern] = st.slider(f"{pattern.capitalize()} Weight", 0.0, 2.0, weight, step=0.1)
            if st.button("Reset to Default Weights"):
                PATTERN_WEIGHTS.update({
                    'streak': 1.2, 'alternating': 1.0, 'zigzag': 0.8, 'trend': 0.9,
                    'big_road': 0.7, 'big_eye': 0.6, 'cockroach': 0.5,
                    'choppy': 0.8, 'double': 0.7, 'markov': 0.9
                })
                st.rerun()

        # Input Game Results
        with st.expander("Input Game Results", expanded=True):
            cols = st.columns(4)
            with cols[0]:
                if st.button("Player", key="player_button"):
                    st.session_state.history.append("Player")
                    st.rerun()
            with cols[1]:
                if st.button("Banker", key="banker_button"):
                    st.session_state.history.append("Banker")
                    st.rerun()
            with cols[2]:
                if st.button("Tie", key="tie_button"):
                    st.session_state.history.append("Tie")
                    st.rerun()
            with cols[3]:
                if st.button("Undo", key="undo_button", disabled=len(st.session_state.history) == 0):
                    st.session_state.history.pop()
                    if st.session_state.money_management_strategy == "T3":
                        st.session_state.t3_results = []
                        st.session_state.t3_level = 1
                        st.session_state.loss_streak = 0
                        st.session_state.skip_betting = False
                        st.session_state.skip_hands_count = 0
                    st.rerun()

        # Shoe Patterns
        with st.expander("Shoe Patterns", expanded=False):
            try:
                pattern_container = st.container()
                with pattern_container:
                    pattern_options = ["Bead Bin", "Big Road", "Big Eye", "Cockroach", "Win/Loss"]
                    selected_patterns = st.multiselect(
                        "Select Patterns to Display",
                        pattern_options,
                        default=st.session_state.selected_patterns,
                        key="pattern_select"
                    )
                    st.session_state.selected_patterns = selected_patterns

                    max_display_cols = min(10 if st.session_state.screen_width < 768 else 14, len(st.session_state.history))

                    if "Bead Bin" in selected_patterns:
                        st.markdown("### Bead Bin")
                        sequence = [r for r in st.session_state.history][-84:]
                        sequence = ['P' if result == 'Player' else 'B' if result == 'Banker' else 'T' for result in sequence]
                        grid = [['' for _ in range(max_display_cols)] for _ in range(6)]
                        for i, result in enumerate(sequence):
                            if result in ['P', 'B', 'T']:
                                col = i // 6
                                row = i % 6
                                if col < max_display_cols:
                                    color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                                    grid[row][col] = f'<div class="pattern-circle" style="background-color: {color}; border-radius: 50%; border: 1px solid #ffffff;"></div>'
                        st.markdown('<div id="bead-bin-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                        for row in grid:
                            st.markdown(' '.join(row), unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        if not st.session_state.history:
                            st.markdown("No results yet. Enter results below.")

                    if "Big Road" in selected_patterns:
                        st.markdown("### Big Road")
                        big_road_grid, num_cols = build_big_road(st.session_state.history)
                        if num_cols > 0:
                            display_cols = min(num_cols, max_display_cols)
                            st.markdown('<div id="big-road-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                            for row in range(6):
                                row_display = []
                                for col in range(display_cols):
                                    outcome = big_road_grid[row][col]
                                    if outcome == 'P':
                                        row_display.append(f'<div class="pattern-circle" style="background-color: #3182ce; border-radius: 50%; border: 1px solid #ffffff;"></div>')
                                    elif outcome == 'B':
                                        row_display.append(f'<div class="pattern-circle" style="background-color: #e53e3e; border-radius: 50%; border: 1px solid #ffffff;"></div>')
                                    elif outcome == 'T':
                                        row_display.append(f'<div class="pattern-circle" style="border: 2px solid #38a169; border-radius: 50%;"></div>')
                                    else:
                                        row_display.append(f'<div class="display-circle"></div>')
                                st.markdown(''.join(row_display), unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown("No Big Road data.")

                    # Similar logic for Big Eye, Cockroach, Win/Loss (omitted for brevity but unchanged)
                    logging.debug("Shoe Patterns rendered")
            except Exception as e:
                logging.error(f"Error rendering Shoe Patterns: {str(e)}")
                st.warning("Unable to display patterns. Please try again or reset the game.")

        # Prediction
        with st.expander("Prediction", expanded=True):
            try:
                bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
                current_bankroll = calculate_bankroll(
                    st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy,
                    st.session_state.initial_bankroll, st.session_state.ai_mode
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
                st.markdown("### Prediction")
                if current_bankroll < max(1.0, st.session_state.base_bet):
                    st.warning("Insufficient bankroll to place a bet. Please increase your bankroll or reset the game.")
                    bet = 'Pass'
                    confidence = 0
                    reason = "Bankroll too low to continue betting."
                    emotional_tone = "Cautious"
                elif st.session_state.skip_betting:
                    skip_reason = f"Skipping betting after three consecutive losses. Waiting for a win or {10 - st.session_state.skip_hands_count} more hands. (Hands skipped: {st.session_state.skip_hands_count})"
                    st.markdown(f"**No Bet**: {skip_reason}")
                elif bet == 'Pass':
                    st.markdown("**No Bet**: Insufficient confidence to place a bet.")
                else:
                    st.markdown(f"**Bet**: {bet} | **Confidence**: {confidence}% | **Bet Size**: ${recommended_bet_size:.2f} | **Mood**: {emotional_tone}")
                if st.session_state.money_management_strategy == "T3":
                    st.markdown(f"**T3 Level**: {st.session_state.t3_level} | **Recent Outcomes**: {''.join(st.session_state.t3_results)} | **Loss Streak**: {st.session_state.loss_streak}")
                st.markdown(f"**Reasoning**: {reason}")
                if pattern_insights:
                    st.markdown("### Pattern Insights")
                    st.markdown("Detected patterns influencing the prediction:")
                    for insight in pattern_insights:
                        st.markdown(f"- {insight}")
                logging.debug("Prediction section rendered")
            except Exception as e:
                logging.error(f"Error in Prediction section: {str(e)}")
                st.warning("Unable to generate prediction. Please try again or reset the game.")

        # Pattern Performance
        with st.expander("Pattern Performance", expanded=False):
            try:
                st.markdown("### Pattern Contributions and Performance")
                for insight in pattern_insights:
                    pattern = insight.split(':')[0].lower()
                    weight = PATTERN_WEIGHTS.get(pattern, 1.0)
                    perf = st.session_state.pattern_performance.get(pattern, {'correct': 0, 'total': 0})
                    accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
                    st.markdown(f"- {insight} (Weight: {weight:.2f}, Accuracy: {accuracy:.2%})")
            except Exception as e:
                logging.error(f"Error in Pattern Performance: {str(e)}")
                st.warning("Unable to display pattern performance.")

        # Bankroll Progress
        with st.expander("Bankroll Progress", expanded=True):
            try:
                bankroll_progress, bet_sizes = calculate_bankroll(
                    st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy,
                    st.session_state.initial_bankroll, st.session_state.ai_mode
                )
                if bankroll_progress:
                    st.markdown("### Bankroll Progress")
                    total_hands = len(bankroll_progress)
                    for i in range(total_hands):
                        hand_number = total_hands - i
                        val = bankroll_progress[total_hands - i - 1]
                        bet_size = bet_sizes[total_hands - i - 1]
                        bet_display = f"Bet ${bet_size:.2f}" if bet_size > 0 else "No Bet"
                        st.markdown(f"Hand {hand_number}: ${val:.2f} | {bet_display}")
                    st.markdown(f"**Current Bankroll**: ${bankroll_progress[-1]:.2f}")

                    st.markdown("### Bankroll Progression Chart")
                    labels = [f"Hand {i+1}" for i in range(len(bankroll_progress))]
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=labels,
                            y=bankroll_progress,
                            mode='lines+markers',
                            name='Bankroll',
                            line=dict(color='#38a169', width=2),
                            marker=dict(size=6)
                        )
                    )
                    fig.update_layout(
                        title=dict(text="Bankroll Over Time", x=0.5, xanchor='center'),
                        xaxis_title="Hand",
                        yaxis_title="Bankroll ($)",
                        xaxis=dict(tickangle=45),
                        yaxis=dict(autorange=True),
                        template="plotly_white",
                        height=400,
                        margin=dict(l=40, r=40, t=50, b=100)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"**Current Bankroll**: ${st.session_state.initial_bankroll:.2f}")
                    st.markdown("No bankroll history yet. Enter results below.")
                logging.debug("Bankroll Progress rendered")
            except Exception as e:
                logging.error(f"Error in Bankroll Progress: {str(e)}")
                st.warning("Unable to display bankroll progress.")

        # Reset
        with st.expander("Reset", expanded=False):
            if st.button("New Game"):
                st.session_state.confirm_reset = True
            if st.session_state.confirm_reset:
                cols = st.columns(2)
                with cols[0]:
                    if st.button("Confirm Reset"):
                        try:
                            final_bankroll = calculate_bankroll(
                                st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy,
                                st.session_state.initial_bankroll, st.session_state.ai_mode
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
                            logging.info("Game reset successfully")
                            st.rerun()
                        except Exception as e:
                            logging.error(f"Error during reset: {str(e)}")
                            st.error("Failed to reset game. Please try again.")
                with cols[1]:
                    if st.button("Cancel"):
                        st.session_state.confirm_reset = False
                        st.rerun()

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        st.error(f"Critical error: {str(e)}. Please refresh the page, clear cache (streamlit cache clear), or restart the app.")
        st.markdown("### Debug Info")
        st.write("If the issue persists, please share the error logs from your terminal.")
        st.write(f"Error: {str(e)}")
        st.write("Try resetting the game or checking your Python environment.")

if __name__ == "__main__":
    main()
