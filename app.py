
import streamlit as st
import time
import logging
import plotly.graph_objects as go
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize pattern weights
if 'PATTERN_WEIGHTS' not in st.session_state:
    st.session_state.PATTERN_WEIGHTS = {
        'streak': 1.2, 'alternating': 1.0, 'trend': 0.9,
        'big_road': 0.7, 'big_eye': 0.6, 'cockroach': 0.5
    }

@st.cache_data
def adjust_pattern_weights(performance_history):
    if not performance_history:
        return
    min_predictions = 5
    min_weight, max_weight = 0.5, 2.0
    default_weights = {
        'streak': 1.2, 'alternating': 1.0, 'trend': 0.9,
        'big_road': 0.7, 'big_eye': 0.6, 'cockroach': 0.5
    }
    start_time = time.time()
    for pattern in st.session_state.PATTERN_WEIGHTS:
        if pattern in performance_history and performance_history[pattern]['total'] >= min_predictions:
            accuracy = performance_history[pattern]['correct'] / performance_history[pattern]['total']
            current_weight = st.session_state.PATTERN_WEIGHTS[pattern]
            new_weight = current_weight * (1.1 if accuracy > 0.6 else 0.9 if accuracy < 0.4 else 1.0)
            st.session_state.PATTERN_WEIGHTS[pattern] = round(max(min_weight, min(new_weight, max_weight)), 2)
        else:
            st.session_state.PATTERN_WEIGHTS[pattern] = default_weights[pattern]
    logging.info(f"Pattern weights adjustment took {time.time() - start_time:.2f} seconds")

# Pattern detection functions
def normalize(s):
    s = s.strip().lower()
    return 'Banker' if s == 'b' else 'Player' if s == 'p' else 'Tie' if s == 't' else None

@st.cache_data
def detect_streaks(_s, window=20):
    s = list(_s)
    streaks = []
    if not s:
        return streaks
    recent = s[-window:] if len(s) >= window else s
    current_outcome = recent[-1]
    current_count = 1
    for i in range(len(recent) - 2, -1, -1):
        if recent[i] == current_outcome:
            current_count += 1
        else:
            streaks.append((current_outcome, current_count))
            current_outcome = recent[i]
            current_count = 1
    streaks.append((current_outcome, current_count))
    return streaks[::-1]

def is_alternating(s, min_length=4):
    if len(s) < min_length:
        return False
    return all(s[i] != s[i + 1] for i in range(len(s) - 1))

def tie_streak(s):
    if not s:
        return 0
    return sum(1 for i in range(len(s)-1, -1, -1) if s[i] == 'Tie' and i >= len(s)-2)

def recent_trend(s, window=10):
    recent = s[-window:] if len(s) >= window else s
    if not recent:
        return None, 0
    freq = frequency_count(recent)
    total = len(recent)
    if total == 0:
        return None, 0
    banker_ratio = freq['Banker'] / total
    player_ratio = freq['Player'] / total
    if banker_ratio > player_ratio + 0.2:
        return 'Banker', min(banker_ratio * 50, 80)
    elif player_ratio > banker_ratio + 0.2:
        return 'Player', min(player_ratio * 50, 80)
    return None, 0

def frequency_count(s):
    return {'Banker': s.count('Banker'), 'Player': s.count('Player'), 'Tie': s.count('Tie')}

@st.cache_data
def build_big_road(_s):
    s = list(_s)
    max_rows, max_cols = 6, 6
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = row = 0
    last_outcome = None
    for result in s[-18:]:
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
    return grid, min(col + 1, max_cols)

@st.cache_data
def build_big_eye_boy(_big_road_grid, num_cols):
    big_road_grid = [list(row) for row in _big_road_grid]
    max_rows, max_cols = 6, 6
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = row = 0
    for c in range(3, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        third_last = [big_road_grid[r][c - 3] for r in range(max_rows) if c >= 3]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        third_non_empty = next((i for i, x in enumerate(third_last) if x in ['P', 'B']), None) if third_last else None
        if last_non_empty is not None and third_non_empty is not None:
            grid[row][col] = 'R' if last_col[last_non_empty] == third_last[third_non_empty] else 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, min(col + 1, max_cols)

@st.cache_data
def build_cockroach_pig(_big_road_grid, num_cols):
    big_road_grid = [list(row) for row in _big_road_grid]
    max_rows, max_cols = 6, 6
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = row = 0
    for c in range(4, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        fourth_last = [big_road_grid[r][c - 4] for r in range(max_rows) if c >= 4]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        fourth_non_empty = next((i for i, x in enumerate(fourth_last) if x in ['P', 'B']), None) if fourth_last else None
        if last_non_empty is not None and fourth_non_empty is not None:
            grid[row][col] = 'R' if last_col[last_non_empty] == fourth_last[fourth_non_empty] else 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, min(col + 1, max_cols)

@st.cache_data
def cache_roads(_history):
    history = list(_history)
    cache = {}
    big_road_grid, num_cols = build_big_road(tuple(history))
    cache['big_road'] = (big_road_grid, num_cols)
    cache['big_eye'] = build_big_eye_boy(big_road_grid, num_cols)
    cache['cockroach'] = build_cockroach_pig(big_road_grid, num_cols)
    return cache

def shoe_position_factor(shoe_position):
    return 1.2 if shoe_position < 20 else 1.0 if shoe_position < 40 else 0.8

def dynamic_confidence_threshold(bankroll, initial_bankroll, entropy, mode):
    base_threshold = 65 if mode == 'Conservative' else 45
    if bankroll / initial_bankroll < 0.5 or entropy > 1.4:
        base_threshold += 5
    return min(base_threshold, 90)

def score_streaks(recent, scores, reason_parts, pattern_insights, pattern_count, mode, position_factor):
    streaks = detect_streaks(tuple(recent), window=10)
    if streaks:
        streak_value, streak_length = streaks[-1]
        if streak_length >= 3 and streak_value != "Tie":
            streak_score = min(25 + (streak_length - 3) * 8, 50) * position_factor * st.session_state.PATTERN_WEIGHTS['streak']
            scores[streak_value] += streak_score
            reason_parts.append(f"Streak of {streak_length} {streak_value}.")
            pattern_insights.append(f"Streak: {streak_length} {streak_value}")
            pattern_count += 1
            if streak_length >= 5 and mode == 'Aggressive':
                contrarian_bet = 'Player' if streak_value == 'Banker' else 'Banker'
                scores[contrarian_bet] += 15 * position_factor
                reason_parts.append("Long streak; possible break.")
                pattern_insights.append("Possible streak break")
    return pattern_count

def score_alternating(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor):
    if len(recent) >= 4 and is_alternating(recent[-4:]):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[alternate_bet] += 30 * position_factor * st.session_state.PATTERN_WEIGHTS['alternating']
        reason_parts.append("Alternating pattern in last 4 hands.")
        pattern_insights.append("Ping Pong: Alternating")
        pattern_count += 1
    return pattern_count

def score_trend(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor, shoe_position):
    trend_bet, trend_score = recent_trend(recent)
    if trend_bet:
        scores[trend_bet] += min(trend_score * 0.8 * position_factor * st.session_state.PATTERN_WEIGHTS['trend'], 30)
        reason_parts.append(f"Trend favors {trend_bet}.")
        pattern_insights.append(f"Trend: {trend_bet}")
        pattern_count += 1
    return pattern_count

def score_big_road(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor):
    big_road_grid, num_cols = road_cache['big_road']
    if num_cols > 0:
        last_col = [big_road_grid[row][num_cols - 1] for row in range(6)]
        col_length = sum(1 for x in last_col if x in ['P', 'B'])
        if col_length >= 3:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            col_score = (25 if col_length == 3 else 35) * position_factor * st.session_state.PATTERN_WEIGHTS['big_road']
            scores[bet_side] += col_score
            reason_parts.append(f"Big Road: {col_length} {bet_side}.")
            pattern_insights.append(f"Big Road: {col_length} {bet_side}")
            pattern_count += 1
    return pattern_count

def score_big_eye(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor):
    big_eye_grid, big_eye_cols = road_cache['big_eye']
    big_road_grid, num_cols = road_cache['big_road']
    if big_eye_cols > 1:
        last_signal = big_eye_grid[0][big_eye_cols - 1] if big_eye_grid[0][big_eye_cols - 1] in ['R', 'B'] else None
        if last_signal == 'R':
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            scores[last_side] += 15 * position_factor * st.session_state.PATTERN_WEIGHTS['big_eye']
            reason_parts.append("Big Eye Boy: Repeat pattern.")
            pattern_insights.append("Big Eye: Repeat")
            pattern_count += 1
        elif last_signal == 'B':
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            scores[opposite_side] += 10 * position_factor * st.session_state.PATTERN_WEIGHTS['big_eye']
            reason_parts.append("Big Eye Boy: Break pattern.")
            pattern_insights.append("Big Eye: Break")
            pattern_count += 1
    return pattern_count

def score_cockroach(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor):
    cockroach_grid, cockroach_cols = road_cache['cockroach']
    big_road_grid, num_cols = road_cache['big_road']
    if cockroach_cols > 1:
        last_signal = cockroach_grid[0][cockroach_cols - 1] if cockroach_grid[0][cockroach_cols - 1] in ['R', 'B'] else None
        if last_signal == 'R':
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            scores[last_side] += 12 * position_factor * st.session_state.PATTERN_WEIGHTS['cockroach']
            reason_parts.append("Cockroach Pig: Repeat pattern.")
            pattern_insights.append("Cockroach: Repeat")
            pattern_count += 1
        elif last_signal == 'B':
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            scores[opposite_side] += 10 * position_factor * st.session_state.PATTERN_WEIGHTS['cockroach']
            reason_parts.append("Cockroach Pig: Break pattern.")
            pattern_insights.append("Cockroach: Break")
            pattern_count += 1
    return pattern_count

def update_pattern_performance(history, pattern_insights, bet, actual_result, performance_tracker):
    for insight in set(pattern_insights):
        pattern = insight.split(':')[0].lower()
        if pattern not in performance_tracker:
            performance_tracker[pattern] = {'correct': 0, 'total': 0}
        performance_tracker[pattern]['total'] += 1
        if bet == actual_result:
            performance_tracker[pattern]['correct'] += 1
    active_patterns = {insight.split(':')[0].lower() for insight in pattern_insights}
    for pattern in list(performance_tracker.keys()):
        if pattern not in active_patterns:
            del performance_tracker[pattern]

@st.cache_data
def advanced_bet_selection(_history, mode='Conservative'):
    start_time = time.time()
    max_recent = 20
    history = list(_history)
    recent = history[-max_recent:] if len(history) > max_recent else history
    if not recent:
        return 'Pass', 0, "Waiting for data.", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    pattern_count = 0
    shoe_position = len(history)
    position_factor = shoe_position_factor(shoe_position)
    current_bankroll = st.session_state.get('current_bankroll', 1000.0)
    road_cache = cache_roads(tuple(history))

    freq = frequency_count(recent)
    total = len(recent)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values() if count > 0) if total > 0 else 0

    if tie_streak(recent) >= 2:
        return 'Pass', 0, "Recent Ties; unstable.", "Cautious", []

    # Compute win/loss tracker
    tracker = []
    base_bet = st.session_state.get('base_bet', 10.0)
    strategy = st.session_state.get('money_management_strategy', 'Flat Betting')
    for i in range(1, len(recent)):
        prev_rounds = recent[:i]
        prev_bet, _ = advanced_bet_selection(tuple(prev_rounds), mode)
        actual_result = recent[i]
        if prev_bet in ('Pass', 'Tie', None) or actual_result == 'Tie':
            tracker.append('S')
        elif prev_bet == actual_result:
            tracker.append('W')
        else:
            tracker.append('L')
    if len(tracker) >= 3 and all(t == 'L' for t in tracker[-3:]):
        return 'Pass', 0, "Three losses; skip until win.", "Cautious", []

    pattern_count = score_streaks(recent, scores, reason_parts, pattern_insights, pattern_count, mode, position_factor)
    pattern_count = score_alternating(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor)
    pattern_count = score_trend(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor, shoe_position)
    pattern_count = score_big_road(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor)
    pattern_count = score_big_eye(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor)
    pattern_count = score_cockroach(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor)

    if total > 0:
        banker_ratio = freq['Banker'] / total
        player_ratio = freq['Player'] / total
        tie_ratio = freq['Tie'] / total
        scores['Banker'] += banker_ratio * 15
        scores['Player'] += player_ratio * 15
        scores['Tie'] += tie_ratio * 10 if tie_ratio > 0.25 else 0
        reason_parts.append(f"B:{freq['Banker']} P:{freq['Player']} T:{freq['Tie']}")
        pattern_insights.append(f"Frequency: B:{freq['Banker']} P:{freq['Player']}")

    if entropy > 1.4:
        for key in scores:
            scores[key] *= 0.6
        reason_parts.append("High randomness.")
        emotional_tone = "Cautious"

    if pattern_count >= 2:
        max_score = max(scores['Banker'], scores['Player'])
        if max_score > 0:
            max_bet = 'Banker' if scores['Banker'] > scores['Player'] else 'Player'
            scores[max_bet] += 10
            reason_parts.append(f"Patterns align: {max_bet}.")
            pattern_insights.append(f"Patterns: {pattern_count}")
        else:
            for key in scores:
                scores[key] = max(0, scores[key] - 10)
            reason_parts.append("Conflicting patterns.")
            emotional_tone = "Skeptical"

    bet_choice = max(scores, key=scores.get)
    confidence = min(round(max(scores.values(), default=0) * 2), 95)
    confidence_threshold = dynamic_confidence_threshold(current_bankroll, st.session_state.get('initial_bankroll', 1000.0), entropy, mode)

    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Low confidence ({confidence}%).")
    elif mode == 'Conservative' and confidence < 70:
        emotional_tone = "Cautious"

    if bet_choice == 'Tie' and (confidence < 80 or freq['Tie'] / total < 0.2):
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(confidence * 0.9), 95)
        reason_parts.append("Tie too risky.")
        emotional_tone = "Cautious"

    reason = ' '.join(reason_parts)
    logging.info(f"advanced_bet_selection took {time.time() - start_time:.2f} seconds")
    return bet_choice, confidence, reason, emotional_tone, pattern_insights

@st.cache_data
def money_management(bankroll, base_bet, strategy):
    min_bet = max(1.0, base_bet)
    max_bet = bankroll
    if bankroll < min_bet:
        return 0.0
    if strategy == 'T3':
        calculated_bet = base_bet * st.session_state.get('t3_level', 1)
    else:
        calculated_bet = base_bet
    return round(max(min_bet, min(calculated_bet, max_bet)), 2)

@st.cache_data
def calculate_bankroll(_history, base_bet, strategy, max_window=20):
    start_time = time.time()
    history = list(_history)
    history = history[-max_window:] if len(history) > max_window else history
    bankroll = st.session_state.get('initial_bankroll', 1000.0)
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, pattern_insights = advanced_bet_selection(tuple(current_rounds[:-1]), st.session_state.get('ai_mode', 'Conservative')) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if bet in ('Pass', 'Tie', None):
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_size = money_management(current_bankroll, base_bet, strategy)
        if bet_size == 0:
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_sizes.append(bet_size)
        if actual_result == bet:
            current_bankroll += bet_size * (0.95 if bet == 'Banker' else 1.0)
            if strategy == 'T3':
                st.session_state.t3_results = st.session_state.get('t3_results', []) + ['W']
                if len(st.session_state.t3_results) == 3:
                    wins = st.session_state.t3_results.count('W')
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1) if wins > 1 else st.session_state.t3_level + 1
                    st.session_state.t3_results = []
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.get('pattern_performance', {}))
        elif actual_result == 'Tie':
            continue
        else:
            current_bankroll -= bet_size
            if strategy == 'T3':
                st.session_state.t3_results = st.session_state.get('t3_results', []) + ['L']
                if len(st.session_state.t3_results) == 3:
                    losses = st.session_state.t3_results.count('L')
                    st.session_state.t3_level = st.session_state.t3_level + 1 if losses > 1 else max(1, st.session_state.t3_level - 1)
                    st.session_state.t3_results = []
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.get('pattern_performance', {}))
        bankroll_progress.append(current_bankroll)
    st.session_state.current_bankroll = current_bankroll
    logging.info(f"Bankroll calc took {time.time() - start_time:.2f} seconds")
    return bankroll_progress, bet_sizes

def add_result(result):
    st.session_state.history.append(result)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]

def undo_result():
    if st.session_state.history:
        st.session_state.history.pop()
    if st.session_state.get('money_management_strategy') == 'T3':
        st.session_state.t3_results = []
        st.session_state.t3_level = 1

def update_settings(initial_bankroll, base_bet, strategy, ai_mode):
    st.session_state.initial_bankroll = initial_bankroll
    st.session_state.base_bet = base_bet
    st.session_state.money_management_strategy = strategy
    st.session_state.ai_mode = ai_mode
    st.session_state.current_bankroll = initial_bankroll
    if strategy == 'T3':
        st.session_state.t3_level = 1
        st.session_state.t3_results = []

def main():
    start_time = time.time()
    try:
        st.set_page_config(page_title="Mang Baccarat", page_icon="🎲", layout="wide")
        st.title("Mang Baccarat Predictor")

        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'initial_bankroll' not in st.session_state:
            st.session_state.initial_bankroll = 1000.0
        if 'current_bankroll' not in st.session_state:
            st.session_state.current_bankroll = st.session_state.initial_bankroll
        if 'base_bet' not in st.session_state:
            st.session_state.base_bet = 10.0
        if 'money_management_strategy' not in st.session_state:
            st.session_state.money_management_strategy = 'Flat Betting'
        if 'ai_mode' not in st.session_state:
            st.session_state.ai_mode = 'Conservative'
        if 'selected_patterns' not in st.session_state:
            st.session_state.selected_patterns = ['Win/Loss']
        if 't3_level' not in st.session_state:
            st.session_state.t3_level = 1
        if 't3_results' not in st.session_state:
            st.session_state.t3_results = []
        if 'pattern_performance' not in st.session_state:
            st.session_state.pattern_performance = {}

        if len(st.session_state.history) >= 10 or (len(st.session_state.history) >= 3 and calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)[1][-3:] == ['L', 'L', 'L']):
            adjust_pattern_weights(st.session_state.pattern_performance)

        # CSS for minimal styling
        st.markdown("""
        <style>
        .pattern-scroll { overflow-x: auto; white-space: nowrap; padding: 5px; }
        .stButton > button { width: 100%; padding: 6px; margin: 3px 0; }
        .pattern-circle { width: 16px; height: 16px; display: inline-block; margin: 1px; }
        @media (max-width: 768px) {
            .pattern-circle { width: 12px; height: 12px; }
            .stButton > button { font-size: 0.8rem; padding: 4px; }
        }
        </style>
        """, unsafe_allow_html=True)

        # Game Settings with Form
        with st.form("settings_form"):
            st.markdown("### Settings")
            cols = st.columns(4)
            with cols[0]:
                initial_bankroll = st.number_input("Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
            with cols[1]:
                base_bet = st.number_input("Base Bet", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
            with cols[2]:
                strategy = st.selectbox("Strategy", ["Flat Betting", "T3"], index=["Flat Betting", "T3"].index(st.session_state.money_management_strategy))
            with cols[3]:
                ai_mode = st.selectbox("Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))
            if st.form_submit_button("Apply"):
                update_settings(initial_bankroll, base_bet, strategy, ai_mode)

        # Input Results
        with st.container():
            cols = st.columns(4)
            with cols[0]:
                st.button("Player", on_click=add_result, args=('Player',))
            with cols[1]:
                st.button("Banker", on_click=add_result, args=('Banker',))
            with cols[2]:
                st.button("Tie", on_click=add_result, args=('Tie',))
            with cols[3]:
                st.button("Undo", on_click=undo_result, disabled=len(st.session_state.history) == 0)

        # Patterns
        with st.expander("Patterns", expanded=False):
            selected_patterns = st.selectbox("Pattern", ["Win/Loss", "Big Road"], index=["Win/Loss", "Big Road"].index(st.session_state.selected_patterns[0] if st.session_state.selected_patterns else "Win/Loss"))
            st.session_state.selected_patterns = [selected_patterns]
            max_display_cols = 6

            if selected_patterns == "Big Road":
                big_road_grid, num_cols = build_big_road(tuple(st.session_state.history))
                if num_cols > 0:
                    st.markdown('<div class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(min(num_cols, max_display_cols)):
                            outcome = big_road_grid[row][col]
                            row_display.append('P' if outcome == 'P' else 'B' if outcome == 'B' else 'T' if outcome == 'T' else '.')
                        st.write(''.join(row_display))
                    st.markdown('</div>', unsafe_allow_html=True)

            if selected_patterns == "Win/Loss":
                st.write("Win/Loss: W=Win, L=Loss, S=Skip/Tie")
                bankroll_progress, bet_sizes = calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)
                row_display = []
                for i, bet_size in enumerate(bet_sizes[-max_display_cols:]):
                    if i >= len(bankroll_progress) - 1:
                        continue
                    if bet_size == 0:
                        row_display.append('S')
                    elif bankroll_progress[i + 1] > bankroll_progress[i]:
                        row_display.append('W')
                    else:
                        row_display.append('L')
                st.write(' '.join(row_display))

        # Prediction
        with st.container():
            bet, confidence, reason, _, _ = advanced_bet_selection(tuple(st.session_state.history), st.session_state.ai_mode)
            current_bankroll = st.session_state.current_bankroll
            bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy)
            if bet == 'Pass':
                st.markdown(f"**No Bet**: {reason}")
            else:
                st.markdown(f"**Bet**: {bet} | **Confidence**: {confidence}% | **Size**: ${bet_size:.2f} | **Reason**: {reason}")

        # Bankroll Progress
        with st.container():
            bankroll_progress, _ = calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)
            bankroll_progress = bankroll_progress[-20:] if len(bankroll_progress) > 20 else bankroll_progress
            if bankroll_progress:
                st.markdown(f"**Bankroll**: ${bankroll_progress[-1]:.2f}")
                fig = go.Figure(go.Scatter(x=list(range(len(bankroll_progress))), y=bankroll_progress, mode='lines', line=dict(color='#38a169')))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # Reset
        with st.container():
            if st.button("Reset"):
                st.session_state.history = []
                st.session_state.current_bankroll = st.session_state.initial_bankroll
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.session_state.pattern_performance = {}
                st.rerun()

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"Error: {str(e)}")
    logging.info(f"Main took {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
