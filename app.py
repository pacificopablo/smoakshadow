import streamlit as st
import logging
import plotly.graph_objects as go
import math
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Normalize input
def normalize(s):
    s = s.strip().lower()
    if s in ('banker', 'b'):
        return 'Banker'
    if s in ('player', 'p'):
        return 'Player'
    if s in ('tie', 't'):
        return 'Tie'
    return None

# Build Big Road grid
def build_big_road(s):
    max_rows = 6
    max_cols = 50
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = 0
    row = 0
    last_outcome = None

    for result in s:
        mapped = 'P' if result == 'Player' else 'B' if result == 'Banker' else 'T'
        if mapped == 'T':
            if col < max_cols and row < max_rows and grid[row][col] == '':
                grid[row][col] = 'T'
            continue
        if col >= max_cols:
            break
        if last_outcome is None or (mapped == last_outcome and row < max_rows - 1):
            grid[row][col] = mapped
            row += 1
        else:
            col += 1
            row = 0
            if col < max_cols:
                grid[row][col] = mapped
                row += 1
        last_outcome = mapped if mapped != 'T' else last_outcome
    return grid, col + 1

# Build Big Eye Boy grid
def build_big_eye_boy(big_road_grid, num_cols):
    max_rows = 6
    max_cols = 50
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = 0
    row = 0

    for c in range(3, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        third_last = [big_road_grid[r][c - 3] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        third_non_empty = next((i for i, x in enumerate(third_last) if x in ['P', 'B']), None)
        if last_non_empty is not None and third_non_empty is not None:
            if last_col[last_non_empty] == third_last[third_non_empty]:
                grid[row][col] = 'R'
            else:
                grid[row][col] = 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, col + 1 if row > 0 else col

# Build Cockroach Pig grid
def build_cockroach_pig(big_road_grid, num_cols):
    max_rows = 6
    max_cols = 50
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = 0
    row = 0

    for c in range(4, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        fourth_last = [big_road_grid[r][c - 4] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        fourth_non_empty = next((i for i, x in enumerate(fourth_last) if x in ['P', 'B']), None)
        if last_non_empty is not None and fourth_non_empty is not None:
            if last_col[last_non_empty] == fourth_last[fourth_non_empty]:
                grid[row][col] = 'R'
            else:
                grid[row][col] = 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, col + 1 if row > 0 else col

# Build Small Road grid
def build_small_road(big_road_grid, num_cols):
    max_rows = 6
    max_cols = 50
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = 0
    row = 0

    for c in range(4, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        third_last = [big_road_grid[r][c - 3] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        third_non_empty = next((i for i, x in enumerate(third_last) if x in ['P', 'B']), None)
        if last_non_empty is not None and third_non_empty is not None:
            if last_col[last_non_empty] == third_last[third_non_empty]:
                grid[row][col] = 'R'
            else:
                grid[row][col] = 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, col + 1 if row > 0 else col

# Advanced bet selection with PBPPBB sequence and reset on win
def advanced_bet_selection(history: List[str], mode: str = 'Conservative') -> Tuple[str, float, str, str, List[str]]:
    CONFIG = {
        'max_recent_count': 40,
        'half_life': 20,
        'min_confidence': {'Conservative': 65, 'Aggressive': 45},
        'tie_confidence_threshold': 85,
        'tie_ratio_threshold': 0.2,
        'pattern_weights': {
            'sequence': 40,  # Weight for PBPPBB sequence
            'frequency': {'banker': 0.9, 'player': 1.0, 'tie': 0.6, 'tie_boost': 25},
        },
        'entropy_threshold': 1.5,
        'entropy_reduction': 0.7,
        'late_shoe_threshold': 60,
        'late_shoe_penalty': 10,
    }

    def calculate_entropy(freq: Dict[str, int], total: int) -> float:
        if total == 0:
            return 0
        return -sum((count / total) * math.log2(count / total) for count in freq.values() if count > 0)

    recent = history[-CONFIG['max_recent_count']:] if len(history) >= CONFIG['max_recent_count'] else history
    if not recent:
        return 'Pass', 0, "No results yet. Waiting for shoe to develop.", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    shoe_position = len(history)

    # Initialize or update sequence position
    if 'sequence_position' not in st.session_state:
        st.session_state.sequence_position = 0
    sequence = ['Player', 'Banker', 'Player', 'Player', 'Banker', 'Banker']  # PBPPBB

    # Check if previous bet won to reset sequence
    if history and 'last_bet' in st.session_state and 'last_result' in st.session_state:
        if st.session_state.last_bet == st.session_state.last_result:
            st.session_state.sequence_position = 0
            reason_parts.append("Previous bet won; resetting PBPPBB sequence to start.")
            pattern_insights.append("Sequence Reset: Starting PBPPBB")
        elif st.session_state.last_result == 'Tie':
            reason_parts.append("Previous result was Tie; maintaining sequence position.")
        else:
            st.session_state.sequence_position = (st.session_state.sequence_position + 1) % len(sequence)
            reason_parts.append(f"Advancing to position {st.session_state.sequence_position + 1} in PBPPBB sequence.")
            pattern_insights.append(f"Sequence Position: {st.session_state.sequence_position + 1}")

    # Select bet based on current sequence position
    bet_choice = sequence[st.session_state.sequence_position]
    scores[bet_choice] += CONFIG['pattern_weights']['sequence']
    reason_parts.append(f"Betting {bet_choice} based on position {st.session_state.sequence_position + 1} in PBPPBB sequence.")
    pattern_insights.append(f"PBPPBB Sequence: Betting {bet_choice}")

    # Store current bet for next iteration
    st.session_state.last_bet = bet_choice
    if history:
        st.session_state.last_result = history[-1]

    # Entropy check
    freq = defaultdict(int, {'Banker': 0, 'Player': 0, 'Tie': 0})
    for r in recent:
        freq[r] += 1
    total = len(recent)
    entropy = calculate_entropy(freq, total)
    if entropy > CONFIG['entropy_threshold']:
        for key in scores:
            scores[key] *= CONFIG['entropy_reduction']
        reason_parts.append("High randomness detected; lowering confidence.")
        pattern_insights.append("Randomness: High entropy")
        emotional_tone = "Cautious"

    # Frequency analysis
    if total > 0:
        banker_ratio = freq['Banker'] / total
        player_ratio = freq['Player'] / total
        tie_ratio = freq['Tie'] / total
        scores['Banker'] += (banker_ratio * CONFIG['pattern_weights']['frequency']['banker']) * CONFIG['pattern_weights']['frequency']['tie_boost']
        scores['Player'] += (player_ratio * CONFIG['pattern_weights']['frequency']['player']) * CONFIG['pattern_weights']['frequency']['tie_boost']
        scores['Tie'] += (tie_ratio * CONFIG['pattern_weights']['frequency']['tie']) * CONFIG['pattern_weights']['frequency']['tie_boost'] if tie_ratio > CONFIG['tie_ratio_threshold'] else 0
        reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
        pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    confidence = min(round(max(scores.values(), default=0) * 1.3), 95)

    confidence_threshold = CONFIG['min_confidence'][mode]
    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}% < {confidence_threshold}%). Passing.")
    elif mode == 'Conservative' and confidence < 75:
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence; proceeding cautiously.")

    if bet_choice == 'Tie' and (confidence < CONFIG['tie_confidence_threshold'] or freq['Tie'] / total < CONFIG['tie_ratio_threshold']):
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice] * 1.3), 95)
        reason_parts.append("Tie bet too risky; switching to safer option.")
        emotional_tone = "Cautious"

    if shoe_position > CONFIG['late_shoe_threshold']:
        confidence = max(confidence - CONFIG['late_shoe_penalty'], 40)
        reason_parts.append("Late in shoe; increasing caution.")
        emotional_tone = "Cautious"

    reason = " ".join(reason_parts)
    return bet_choice, confidence, reason, emotional_tone, pattern_insights

# Money management with 1-3-2-1 progression
def money_management(bankroll, base_bet, strategy, bet_outcome=None):
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if bankroll < min_bet:
        logging.warning(f"Bankroll ({bankroll:.2f}) is less than minimum bet ({min_bet:.2f}).")
        return 0.0

    if strategy == "1-3-2-1":
        progression = [1, 3, 2, 1]  # BLACKBOXAI progression
        if bet_outcome == 'win':
            st.session_state.progression_count += 1
            if st.session_state.progression_count >= len(progression):
                st.session_state.progression_count = 0
            st.session_state.progression_level = progression[st.session_state.progression_count]
        elif bet_outcome == 'loss':
            st.session_state.progression_count = 0
            st.session_state.progression_level = progression[0]
        calculated_bet = base_bet * st.session_state.progression_level
    elif strategy == "T3":
        if bet_outcome == 'win':
            if not st.session_state.t3_results:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            st.session_state.t3_results.append('W')
        elif bet_outcome == 'loss':
            st.session_state.t3_results.append('L')
        if len(st.session_state.t3_results) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins > losses:
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            elif losses > wins:
                st.session_state.t3_level += 1
            st.session_state.t3_results = []
        calculated_bet = base_bet * st.session_state.t3_level
    else:  # Flat Betting
        calculated_bet = base_bet

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2)

# Bankroll calculation with BLACKBOXAI limits
def calculate_bankroll(history, base_bet, strategy):
    bankroll = st.session_state.initial_bankroll
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.progression_level = 1
    st.session_state.progression_count = 0
    main_fund = bankroll * 0.6  # 60% main betting (BLACKBOXAI)
    recovery_fund = bankroll * 0.3  # 30% recovery
    accel_fund = bankroll * 0.1  # 10% acceleration
    profit_target = bankroll * 1.3  # 30% profit
    loss_limit = bankroll * 0.8  # 20% loss

    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, _ = advanced_bet_selection(current_rounds[:-1], st.session_state.ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', '')
        actual_result = history[i]
        
        # Use recovery fund after loss, acceleration fund during streak
        use_fund = main_fund
        if st.session_state.get('last_bet_outcome') == 'loss' and recovery_fund > 0:
            use_fund = min(recovery_fund, main_fund)
        elif st.session_state.get('win_streak', 0) >= 2 and accel_fund > 0:
            use_fund = min(accel_fund, main_fund)

        if bet in (None, 'Pass', 'Tie') or current_bankroll < base_bet:
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue

        bet_size = money_management(use_fund, base_bet, strategy)
        if bet_size == 0.0:
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_sizes.append(bet_size)

        if actual_result == bet:
            if bet == 'Banker':
                win_amount = bet_size  # Assuming commission-free (BLACKBOXAI)
                current_bankroll += win_amount
                main_fund += win_amount
                accel_fund += win_amount * 0.1
                st.session_state.win_streak = st.session_state.get('win_streak', 0) + 1
                st.session_state.last_bet_outcome = 'win'
            else:
                current_bankroll += bet_size
                main_fund += bet_size
                accel_fund += bet_size * 0.1
                st.session_state.win_streak = st.session_state.get('win_streak', 0) + 1
                st.session_state.last_bet_outcome = 'win'
            if strategy in ["T3", "1-3-2-1"]:
                money_management(use_fund, base_bet, strategy, bet_outcome='win')
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
            main_fund -= bet_size
            recovery_fund -= bet_size * 0.3
            st.session_state.win_streak = 0
            st.session_state.last_bet_outcome = 'loss'
            if strategy in ["T3", "1-3-2-1"]:
                money_management(use_fund, base_bet, strategy, bet_outcome='loss')

        bankroll_progress.append(current_bankroll)

        # Enforce BLACKBOXAI profit/loss limits
        if current_bankroll >= profit_target or current_bankroll <= loss_limit:
            break

    return bankroll_progress, bet_sizes

# Win/Loss tracker
def calculate_win_loss_tracker(history: List[str], base_bet: float, strategy: str, ai_mode: str) -> List[str]:
    tracker = []
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    st.session_state.progression_level = 1
    st.session_state.progression_count = 0
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, _, _, _, _ = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', '')
        actual_result = history[i]
        if actual_result == 'Tie':
            tracker.append('T')
        elif bet in (None, 'Pass'):
            tracker.append('S')
        elif actual_result == bet:
            tracker.append('W')
            if strategy in ["T3", "1-3-2-1"]:
                money_management(st.session_state.initial_bankroll, base_bet, strategy, bet_outcome='win')
        else:
            tracker.append('L')
            if strategy in ["T3", "1-3-2-1"]:
                money_management(st.session_state.initial_bankroll, base_bet, strategy, bet_outcome='loss')
    return tracker

def main():
    try:
        st.set_page_config(page_title="Mang Baccarat Predictor", page_icon="🎲", layout="wide")
        st.title("Mang Baccarat Predictor")

        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'initial_bankroll' not in st.session_state:
            st.session_state.initial_bankroll = 1000.0
        if 'base_bet' not in st.session_state:
            st.session_state.base_bet = 10.0
        if 'money_management_strategy' not in st.session_state:
            st.session_state.money_management_strategy = "1-3-2-1"
        if 'ai_mode' not in st.session_state:
            st.session_state.ai_mode = "Conservative"
        if 'selected_patterns' not in st.session_state:
            st.session_state.selected_patterns = ["Bead Bin", "Win/Loss", "Triple Repeat"]
        if 't3_level' not in st.session_state:
            st.session_state.t3_level = 1
        if 't3_results' not in st.session_state:
            st.session_state.t3_results = []
        if 'progression_level' not in st.session_state:
            st.session_state.progression_level = 1
        if 'progression_count' not in st.session_state:
            st.session_state.progression_count = 0
        if 'win_streak' not in st.session_state:
            st.session_state.win_streak = 0
        if 'last_bet_outcome' not in st.session_state:
            st.session_state.last_bet_outcome = None
        if 'screen_width' not in st.session_state:
            st.session_state.screen_width = 1024
        if 'sequence_position' not in st.session_state:
            st.session_state.sequence_position = 0
        if 'last_bet' not in st.session_state:
            st.session_state.last_bet = None
        if 'last_result' not in st.session_state:
            st.session_state.last_result = None

        st.markdown("""
            <script>
            function updateScreenWidth() {
                const width = window.innerWidth;
                document.getElementById('screen-width-input').value = width;
            }
            window.onload = updateScreenWidth;
            window.onresize = updateScreenWidth;
            </script>
            <input type="hidden" id="screen-width-input">
        """, unsafe_allow_html=True)

        screen_width_input = st.text_input("Screen Width", key="screen_width_input", value=str(st.session_state.screen_width), disabled=True)
        try:
            st.session_state.screen_width = int(screen_width_input) if screen_width_input.isdigit() else 1024
        except ValueError:
            st.session_state.screen_width = 1024

        st.markdown("""
            <style>
            .pattern-scroll {
                overflow-x: auto;
                white-space: nowrap;
                max-width: 100%;
                padding: 10px;
                border: 1px solid #e1e1e1;
                background-color: #f9f9f9;
            }
            .pattern-scroll::-webkit-scrollbar {
                height: 8px;
            }
            .pattern-scroll::-webkit-scrollbar-thumb {
                background-color: #888;
                border-radius: 4px;
            }
            .stButton > button {
                width: 100%;
                padding: 8px;
                margin: 5px 0;
            }
            .stNumberInput, .stSelectbox {
                width: 100% !important;
            }
            .stExpander {
                margin-bottom: 10px;
            }
            h1 {
                font-size: 2.5rem;
                text-align: center;
            }
            h3 {
                font-size: 1.5rem;
            }
            p, div, span {
                font-size: 1rem;
            }
            .pattern-circle {
                width: 22px;
                height: 22px;
                display: inline-block;
                margin: 2px;
            }
            .display-circle {
                width: 22px;
                height: 22px;
                display: inline-block;
                margin: 2px;
            }
            @media (min-width: 769px) {
                .stButton > button, .stNumberInput, .stSelectbox {
                    max-width: 200px;
                }
            }
            @media (max-width: 768px) {
                h1 {
                    font-size: 1.8rem;
                }
                h3 {
                    font-size: 1.2rem;
                }
                p, div, span {
                    font-size: 0.9rem;
                }
                .pattern-circle, .display-circle {
                    width: 16px !important;
                    height: 16px !important;
                }
                .stButton > button {
                    font-size: 0.9rem;
                    padding: 6px;
                }
                .stNumberInput input, .stSelectbox div {
                    font-size: 0.9rem;
                }
                .st-emotion-cache-1dj3wfg {
                    flex-wrap: wrap;
                }
            }
            </style>
            <script>
            function autoScrollPatterns() {
                const containers = [
                    'bead-bin-scroll',
                    'big-road-scroll',
                    'big-eye-scroll',
                    'cockroach-scroll',
                    'win-loss-scroll',
                    'triple-repeat-scroll'
                ];
                containers.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) {
                        element.scrollLeft = element.scrollWidth;
                    }
                });
            }
            window.onload = autoScrollPatterns;
            </script>
        """, unsafe_allow_html=True)

        with st.expander("Game Settings", expanded=False):
            st.markdown("**BLACKBOXAI Strategy**: Use commission-free Baccarat (e.g., EZ Baccarat) for lower house edge. Divide bankroll into 50 units: 60% main betting, 30% recovery, 10% acceleration. Stop at 30% profit or 20% loss. Take 30-minute breaks after big wins/losses.")
            cols = st.columns(4)
            with cols[0]:
                initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
            with cols[1]:
                base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
            with cols[2]:
                strategy_options = ["Flat Betting", "T3", "1-3-2-1"]
                money_management_strategy = st.selectbox("Money Management Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
                st.markdown("*Flat: Fixed bets. T3: Adjusts based on last three outcomes. 1-3-2-1: BLACKBOXAI progression (1, 3, 2, 1 units).*")
            with cols[3]:
                ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

            st.session_state.initial_bankroll = initial_bankroll
            st.session_state.base_bet = base_bet
            st.session_state.money_management_strategy = money_management_strategy
            st.session_state.ai_mode = ai_mode

            st.markdown(f"**Selected Strategy: {money_management_strategy}**")

        with st.expander("Input Game Results", expanded=True):
            cols = st.columns(4)
            with cols[0]:
                if st.button("Player"):
                    st.session_state.history.append("Player")
                    st.rerun()
            with cols[1]:
                if st.button("Banker"):
                    st.session_state.history.append("Banker")
                    st.rerun()
            with cols[2]:
                if st.button("Tie"):
                    st.session_state.history.append("Tie")
                    st.rerun()
            with cols[3]:
                undo_clicked = st.button("Undo", disabled=len(st.session_state.history) == 0)
                if undo_clicked and len(st.session_state.history) == 0:
                    st.warning("No results to undo!")
                elif undo_clicked:
                    st.session_state.history.pop()
                    if st.session_state.money_management_strategy in ["T3", "1-3-2-1"]:
                        st.session_state.t3_results = []
                        st.session_state.t3_level = 1
                        st.session_state.progression_count = 0
                        st.session_state.progression_level = 1
                    st.rerun()

        with st.expander("Shoe Patterns", expanded=False):
            pattern_options = ["Bead Bin", "Big Road", "Big Eye", "Cockroach", "Win/Loss", "Triple Repeat"]
            selected_patterns = st.multiselect(
                "Select Patterns to Display",
                pattern_options,
                default=st.session_state.selected_patterns,
                key="pattern_select"
            )
            st.session_state.selected_patterns = selected_patterns

            max_display_cols = 10 if st.session_state.screen_width < 768 else 14

            if "Bead Bin" in st.session_state.selected_patterns:
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

            if "Big Road" in st.session_state.selected_patterns:
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

            if "Big Eye" in st.session_state.selected_patterns:
                st.markdown("### Big Eye Boy")
                st.markdown("<p style='font-size: 12px; color: #666666;'>Red (🎴): Repeat Pattern, Blue (🎵): Break Pattern</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = build_big_road(st.session_state.history)
                big_eye_grid, big_eye_cols = build_big_eye_boy(big_road_grid, num_cols)
                if big_eye_cols > 0:
                    display_cols = min(big_eye_cols, max_display_cols)
                    st.markdown('<div id="big-eye-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = big_eye_grid[row][col]
                            if outcome == 'R':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #e53e3e; border-radius: 50%; border: 1px solid #000000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #3182ce; border-radius: 50%; border: 1px solid #000000;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("No recent Big Eye data.")

            if "Cockroach" in st.session_state.selected_patterns:
                st.markdown("### Cockroach Pig")
                st.markdown("<p style='font-size: 12px; color: #666666;'>Red (🎴): Repeat Pattern, Blue (🎵): Break Pattern</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = build_big_road(st.session_state.history)
                cockroach_grid, cockroach_cols = build_cockroach_pig(big_road_grid, num_cols)
                if cockroach_cols > 0:
                    display_cols = min(cockroach_cols, max_display_cols)
                    st.markdown('<div id="cockroach-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = cockroach_grid[row][col]
                            if outcome == 'R':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #e53e3e; border-radius: 50%; border: 1px solid #000000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #3182ce; border-radius: 50%; border: 1px solid #000000;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("No recent Cockroach data.")

            if "Triple Repeat" in st.session_state.selected_patterns:
                st.markdown("### Triple Repeat (BLACKBOXAI PBPPBB)")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🎢: PBPPBB pattern</p>", unsafe_allow_html=True)
                sequence = st.session_state.history[-12:]
                row_display = []
                for i in range(len(sequence) - 5):
                    chunk = sequence[i:i+6]
                    if len(chunk) == 6 and chunk == ['Player', 'Banker', 'Player', 'Player', 'Banker', 'Banker']:
                        row_display.append(f'<div class="pattern-circle" style="background-color: #38a169;"></div>')
                    else:
                        row_display.append(f'<div class="display-circle"></div>')
                st.markdown('<div id="triple-repeat-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if "Win/Loss" in st.session_state.selected_patterns:
                st.markdown("### Win/Loss")
                st.markdown("<p style='font-size: 12px; color: #666666;'>Green (🎢): Win, Red (🎴): Loss, Gray (⬛): Skip or Tie</p>", unsafe_allow_html=True)
                tracker = calculate_win_loss_tracker(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.ai_mode)[-max_display_cols:]
                row_display = []
                for result in tracker:
                    if result in ['W', 'L', 'S', 'T']:
                        color = '#38a169' if result == 'W' else '#e53e3e' if result == 'L' else '#A0AEC0'
                        row_display.append(f'<div class="pattern-circle" style="background-color: {color}; border-radius: 50%; border: 1px solid #000000;"></div>')
                    else:
                        row_display.append(f'<div class="display-circle"></div>')
                st.markdown('<div id="win-loss-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if not st.session_state.history:
                    st.markdown("No results yet. Enter results below.")

        with st.expander("Prediction", expanded=True):
            bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
            st.markdown("### Prediction")
            current_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
            recommended_bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy)
            if current_bankroll < max(1.0, st.session_state.base_bet):
                st.warning("Insufficient bankroll to place a bet. Please increase your bankroll or reset the game.")
                bet = 'Pass'
                confidence = 0
                reason = "Bankroll too low to continue betting."
                emotional_tone = "Cautious"
            if bet == 'Pass':
                st.markdown("**No Bet**: Insufficient confidence or bankroll to place a bet.")
            else:
                st.markdown(f"**Bet**: {bet} | **Confidence**: {confidence}% | **Bet Size**: ${recommended_bet_size:.2f} | **Mood**: {emotional_tone}")
            st.markdown(f"**Reasoning**: {reason}")
            if pattern_insights:
                st.markdown("### Pattern Insights")
                st.markdown("Detected patterns influencing the prediction:")
                for insight in pattern_insights:
                    st.markdown(f"- {insight}")

        with st.expander("Bankroll Progress", expanded=True):
            bankroll_progress, bet_sizes = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)
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

        with st.expander("Reset", expanded=False):
            if st.button("New Game"):
                final_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
                st.session_state.history = []
                st.session_state.initial_bankroll = max(1.0, final_bankroll)
                st.session_state.base_bet = min(10.0, st.session_state.initial_bankroll)
                st.session_state.money_management_strategy = "1-3-2-1"
                st.session_state.ai_mode = "Conservative"
                st.session_state.selected_patterns = ["Bead Bin", "Win/Loss", "Triple Repeat"]
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.session_state.progression_count = 0
                st.session_state.progression_level = 1
                st.session_state.win_streak = 0
                st.session_state.last_bet_outcome = None
                st.session_state.sequence_position = 0
                st.session_state.last_bet = None
                st.session_state.last_result = None
                st.rerun()

    except (KeyError, ValueError, IndexError) as e:
        logging.error(f"Error in main: {str(e)}")
        st.error(f"Error occurred: {str(e)}. Please try refreshing the page or resetting the game.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        st.error(f"Unexpected error: {str(e)}. Contact support if this persists.")

if __name__ == "__main__":
    main()
