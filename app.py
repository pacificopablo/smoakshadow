
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
        'streak': 1.2, 'alternating': 1.0, 'zigzag': 0.8, 'trend': 0.9,
        'big_road': 0.7, 'big_eye': 0.6, 'cockroach': 0.5, 'choppy': 0.8,
        'double': 0.7, 'markov': 0.9
    }

# Function to adjust pattern weights
@st.cache_data
def adjust_pattern_weights(performance_history):
    if not performance_history:
        return
    performance = performance_history
    min_predictions = 10
    min_weight, max_weight = 0.5, 2.0
    default_weights = {
        'streak': 1.2, 'alternating': 1.0, 'zigzag': 0.8, 'trend': 0.9,
        'big_road': 0.7, 'big_eye': 0.6, 'cockroach': 0.5, 'choppy': 0.8,
        'double': 0.7, 'markov': 0.9
    }
    start_time = time.time()
    for pattern in st.session_state.PATTERN_WEIGHTS:
        if pattern in performance and performance[pattern]['total'] >= min_predictions:
            accuracy = performance[pattern]['correct'] / performance[pattern]['total']
            current_weight = st.session_state.PATTERN_WEIGHTS[pattern]
            if accuracy > 0.6:
                new_weight = current_weight * 1.1
            elif accuracy < 0.4:
                new_weight = current_weight * 0.9
            else:
                new_weight = current_weight
            new_weight = max(min_weight, min(new_weight, max_weight))
            st.session_state.PATTERN_WEIGHTS[pattern] = round(new_weight, 2)
        else:
            st.session_state.PATTERN_WEIGHTS[pattern] = default_weights[pattern]
    logging.info(f"Pattern weights adjustment took {time.time() - start_time:.2f} seconds")

# Pattern detection functions
def normalize(s):
    s = s.strip().lower()
    return 'Banker' if s in ('banker', 'b') else 'Player' if s in ('player', 'p') else 'Tie' if s in ('tie', 't') else None

@st.cache_data
def detect_streaks(s, window=32):
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

def is_zigzag(s):
    if len(s) < 3:
        return False
    return any(s[i] == s[i + 2] and s[i] != s[i + 1] for i in range(len(s) - 2))

def is_choppy(s, min_length=6):
    if len(s) < min_length:
        return False
    streak_lengths = []
    current_streak = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current_streak += 1
        else:
            streak_lengths.append(current_streak)
            current_streak = 1
    streak_lengths.append(current_streak)
    avg_streak = sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0
    return len(streak_lengths) >= 4 and 1 <= avg_streak <= 2

def is_double_pattern(s, min_length=6):
    if len(s) < min_length:
        return False, None
    for i in range(len(s) - 3):
        if s[i] == s[i+1] and s[i+2] == s[i+3] and s[i] != s[i+2]:
            return True, s[i+2]
    return False, None

def tie_streak(s):
    if not s:
        return 0
    return sum(1 for i in range(len(s)-1, -1, -1) if s[i] == 'Tie')

def recent_trend(s, window=12):
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

def calculate_transition_probs(s, window=20):
    recent = s[-window:] if len(s) >= window else s
    transitions = {('B', 'B'): 0, ('B', 'P'): 0, ('P', 'B'): 0, ('P', 'P'): 0}
    total_from = {'B': 0, 'P': 0}
    for i in range(len(recent) - 1):
        if recent[i] in ['Banker', 'Player'] and recent[i+1] in ['Banker', 'Player']:
            from_outcome = 'B' if recent[i] == 'Banker' else 'P'
            to_outcome = 'B' if recent[i+1] == 'Banker' else 'P'
            transitions[(from_outcome, to_outcome)] += 1
            total_from[from_outcome] += 1
    probs = {(f, t): transitions[(f, t)] / total_from[f] if total_from[f] > 0 else 0 for f in ['B', 'P'] for t in ['B', 'P']}
    return probs

@st.cache_data
def build_big_road(_s):
    s = list(_s)  # Ensure input is a list
    max_rows, max_cols = 6, 32
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = row = 0
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

@st.cache_data
def build_big_eye_boy(_big_road_grid, num_cols):
    big_road_grid = [list(row) for row in _big_road_grid]  # Ensure mutable copy
    max_rows, max_cols = 6, 32
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = row = 0
    for c in range(3, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        third_last = [big_road_grid[r][c - 3] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        third_non_empty = next((i for i, x in enumerate(third_last) if x in ['P', 'B']), None)
        if last_non_empty is not None and third_non_empty is not None:
            grid[row][col] = 'R' if last_col[last_non_empty] == third_last[third_non_empty] else 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, col + 1 if row > 0 else col

@st.cache_data
def build_cockroach_pig(_big_road_grid, num_cols):
    big_road_grid = [list(row) for row in _big_road_grid]  # Ensure mutable copy
    max_rows, max_cols = 6, 32
    grid = [['' for _ in range(max_cols)] for _ in range(max_rows)]
    col = row = 0
    for c in range(4, num_cols):
        if col >= max_cols:
            break
        last_col = [big_road_grid[r][c - 1] for r in range(max_rows)]
        fourth_last = [big_road_grid[r][c - 4] for r in range(max_rows)]
        last_non_empty = next((i for i, x in enumerate(last_col) if x in ['P', 'B']), None)
        fourth_non_empty = next((i for i, x in enumerate(fourth_last) if x in ['P', 'B']), None)
        if last_non_empty is not None and fourth_non_empty is not None:
            grid[row][col] = 'R' if last_col[last_non_empty] == fourth_last[fourth_non_empty] else 'B'
            row += 1
            if row >= max_rows:
                col += 1
                row = 0
        else:
            col += 1
            row = 0
    return grid, col + 1 if row > 0 else col

@st.cache_data
def cache_roads(_history):
    history = list(_history)  # Ensure input is a list
    cache = {}
    big_road_grid, num_cols = build_big_road(tuple(history))
    cache['big_road'] = (big_road_grid, num_cols)
    cache['big_eye'] = build_big_eye_boy(big_road_grid, num_cols)
    cache['cockroach'] = build_cockroach_pig(big_road_grid, num_cols)
    return cache

def shoe_position_factor(shoe_position):
    if shoe_position < 20:
        return 1.2
    elif shoe_position < 40:
        return 1.0
    elif shoe_position < 60:
        return 0.8
    else:
        return 0.6

def dynamic_confidence_threshold(bankroll, initial_bankroll, entropy, mode):
    base_threshold = 65 if mode == 'Conservative' else 45
    bankroll_ratio = bankroll / initial_bankroll
    if bankroll_ratio < 0.5:
        base_threshold += 5
    if entropy > 1.5:
        base_threshold += 5
    return min(base_threshold, 90)

def score_streaks(recent, scores, reason_parts, pattern_insights, pattern_count, mode, position_factor):
    streaks = detect_streaks(tuple(recent), window=20)
    if streaks:
        streak_value, streak_length = streaks[-1]
        if streak_length >= 3 and streak_value != "Tie":
            streak_score = min(25 + (streak_length - 3) * 8, 50) * position_factor * st.session_state.PATTERN_WEIGHTS['streak']
            if streak_length >= 6:
                streak_score += 10
                pattern_insights.append(f"Dragon Tail: {streak_length} {streak_value}")
            scores[streak_value] += streak_score
            reason_parts.append(f"Streak of {streak_length} {streak_value} wins detected.")
            pattern_insights.append(f"Streak: {streak_length} {streak_value}")
            pattern_count += 1
            if streak_length >= 5 and mode == 'Aggressive':
                contrarian_bet = 'Player' if streak_value == 'Banker' else 'Banker'
                scores[contrarian_bet] += 20 * position_factor * st.session_state.PATTERN_WEIGHTS['streak']
                reason_parts.append(f"Long streak ({streak_length}); considering break in Aggressive mode.")
                pattern_insights.append("Possible streak break")
        long_streaks = sum(1 for _, length in streaks if length >= 3)
        if long_streaks >= 3:
            scores[streak_value] += 15 * position_factor * st.session_state.PATTERN_WEIGHTS['streak']
            reason_parts.append(f"Frequent streaks detected ({long_streaks} in last 20 hands).")
            pattern_insights.append(f"Frequent Streaks: {long_streaks} detected")
            pattern_count += 1
    return pattern_count

def score_alternating(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor):
    if len(recent) >= 6 and is_alternating(recent[-6:], min_length=6):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[alternate_bet] += 35 * position_factor * st.session_state.PATTERN_WEIGHTS['alternating']
        reason_parts.append("Strong alternating pattern (Ping Pong) in last 6 hands.")
        pattern_insights.append("Ping Pong: Alternating P/B")
        pattern_count += 1
    return pattern_count

def score_zigzag(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor):
    if is_zigzag(recent[-8:]):
        last = recent[-1]
        zigzag_bet = 'Player' if last == 'Banker' else 'Banker'
        zigzag_score = (30 if len(recent) < 30 else 20) * position_factor * st.session_state.PATTERN_WEIGHTS['zigzag']
        scores[zigzag_bet] += zigzag_score
        reason_parts.append("Zigzag pattern (P-B-P or B-P-B) detected in last 8 hands.")
        pattern_insights.append("Zigzag: P-B-P/B-P-B")
        pattern_count += 1
    return pattern_count

def score_choppy(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor):
    if is_choppy(recent[-8:]):
        last = recent[-1]
        choppy_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[choppy_bet] += 25 * position_factor * st.session_state.PATTERN_WEIGHTS['choppy']
        reason_parts.append("Choppy pattern detected in last 8 hands (short streaks).")
        pattern_insights.append("Choppy: Short alternating streaks")
        pattern_count += 1
    return pattern_count

def score_double(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor):
    double_detected, double_bet = is_double_pattern(recent[-8:])
    if double_detected:
        scores[double_bet] += 20 * position_factor * st.session_state.PATTERN_WEIGHTS['double']
        reason_parts.append(f"Double pattern (e.g., BB or PP) detected; betting {double_bet}.")
        pattern_insights.append(f"Double: {double_bet} expected")
        pattern_count += 1
    return pattern_count

def score_trend(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor, shoe_position):
    trend_bet, trend_score = recent_trend(recent, window=12)
    if trend_bet:
        trend_weight = trend_score * (1 if shoe_position < 20 else 0.8) * position_factor * st.session_state.PATTERN_WEIGHTS['trend']
        scores[trend_bet] += min(trend_weight, 35)
        reason_parts.append(f"Recent trend favors {trend_bet} in last 12 hands.")
        pattern_insights.append(f"Trend: {trend_bet} dominance")
        pattern_count += 1
    return pattern_count

def score_big_road(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor):
    big_road_grid, num_cols = road_cache['big_road']
    if num_cols > 0:
        last_col = [big_road_grid[row][num_cols - 1] for row in range(6)]
        col_length = sum(1 for x in last_col if x in ['P', 'B'])
        if col_length >= 3:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            col_score = (25 if col_length == 3 else 35 if col_length == 4 else 45) * position_factor * st.session_state.PATTERN_WEIGHTS['big_road']
            scores[bet_side] += col_score
            reason_parts.append(f"Big Road column of {col_length} {bet_side}.")
            pattern_insights.append(f"Big Road: {col_length} {bet_side}")
            pattern_count += 1
    return pattern_count

def score_big_eye(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor):
    big_eye_grid, big_eye_cols = road_cache['big_eye']
    big_road_grid, num_cols = road_cache['big_road']
    if big_eye_cols > 1:
        last_two_cols = [[big_eye_grid[row][c] for row in range(6)] for c in range(big_eye_cols - 2, big_eye_cols)]
        last_signals = [next((x for x in col if x in ['R', 'B']), None) for col in last_two_cols]
        if all(s == 'R' for s in last_signals if s):
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            scores[last_side] += 20 * position_factor * st.session_state.PATTERN_WEIGHTS['big_eye']
            reason_parts.append("Big Eye Boy shows consistent repeat pattern.")
            pattern_insights.append("Big Eye Boy: Consistent repeat")
            pattern_count += 1
        elif all(s == 'B' for s in last_signals if s):
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            scores[opposite_side] += 15 * position_factor * st.session_state.PATTERN_WEIGHTS['big_eye']
            reason_parts.append("Big Eye Boy shows consistent break pattern.")
            pattern_insights.append("Big Eye Boy: Consistent break")
            pattern_count += 1
    return pattern_count

def score_cockroach(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor):
    cockroach_grid, cockroach_cols = road_cache['cockroach']
    big_road_grid, num_cols = road_cache['big_road']  # Fixed reference
    if cockroach_cols > 1:
        last_two_cols = [[cockroach_grid[row][c] for row in range(6)] for c in range(cockroach_cols - 2, cockroach_cols)]
        last_signals = [next((x for x in col if x in ['R', 'B']), None) for col in last_two_cols]
        if all(s == 'R' for s in last_signals if s):
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            scores[last_side] += 15 * position_factor * st.session_state.PATTERN_WEIGHTS['cockroach']
            reason_parts.append("Cockroach Pig shows consistent repeat pattern.")
            pattern_insights.append("Cockroach Pig: Consistent repeat")
            pattern_count += 1
        elif all(s == 'B' for s in last_signals if s):
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            scores[opposite_side] += 12 * position_factor * st.session_state.PATTERN_WEIGHTS['cockroach']
            reason_parts.append("Cockroach Pig shows consistent break pattern.")
            pattern_insights.append("Cockroach Pig: Consistent break")
            pattern_count += 1
    return pattern_count

def score_markov(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor):
    if recent:
        last_outcome = 'B' if recent[-1] == 'Banker' else 'P' if recent[-1] == 'Player' else None
        if last_outcome:
            trans_probs = calculate_transition_probs(recent)
            next_banker_prob = trans_probs[(last_outcome, 'B')]
            next_player_prob = trans_probs[(last_outcome, 'P')]
            if next_banker_prob > next_player_prob + 0.1:
                scores['Banker'] += 20 * next_banker_prob * position_factor * st.session_state.PATTERN_WEIGHTS['markov']
                reason_parts.append(f"Markov transition favors Banker ({next_banker_prob:.2%} vs {next_player_prob:.2%}).")
                pattern_insights.append(f"Markov: Banker {next_banker_prob:.2%}")
                pattern_count += 1
            elif next_player_prob > next_banker_prob + 0.1:
                scores['Player'] += 20 * next_player_prob * position_factor * st.session_state.PATTERN_WEIGHTS['markov']
                reason_parts.append(f"Markov transition favors Player ({next_player_prob:.2%} vs {next_banker_prob:.2%}).")
                pattern_insights.append(f"Markov: Player {next_player_prob:.2%}")
                pattern_count += 1
    return pattern_count

def update_pattern_performance(history, pattern_insights, bet, actual_result, performance_tracker):
    for insight in set(pattern_insights):
        pattern = insight.split(':')[0].lower()
        if pattern_count not in performance_tracker:
            performance_tracker[pattern] = {'correct': 0, 'total': 0}
        performance_tracker[pattern]['total'] += 1
        if bet == actual_result:
            performance_tracker[pattern]['correct'] += 1

@st.cache_data
def advanced_bet_selection(history, mode='Conservative'):
    start_time = time.time()
    max_recent = 32
    recent = history[-max_recent:] if len(history) >= max_recent else history
    if not recent:
        return 'Pass', 0, "No results yet. Waiting for shoe to develop.", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    pattern_count = 0
    shoe_position = len(history)
    position_factor = shoe_position_factor(shoe_position)
    current_bankroll = st.session_state.get('initial_bankroll', 1000.0)
    road_cache = cache_roads(tuple(history))  # Convert to tuple for caching

    freq = frequency_count(recent)
    total = len(recent)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values() if count > 0) if total > 0 else 0

    if tie_streak(recent) >= 2:
        return 'Pass', 0, f"Recent streak of {tie_streak(recent)} Ties; too unstable to bet.", "Cautious", []

    tracker = calculate_win_loss_tracker(recent, st.session_state.get('base_bet', 10.0), st.session_state.get('money_management_strategy', 'Flat Betting'), mode)
    if len(tracker) >= 3 and tracker[-3:] == ['L', 'L', 'L']:
        return 'Pass', 0, "Three consecutive losses detected; skipping until a win.", "Cautious", []

    pattern_count = score_streaks(recent, scores, reason_parts, pattern_insights, pattern_count, mode, position_factor)
    pattern_count = score_alternating(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor)
    pattern_count = score_zigzag(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor)
    pattern_count = score_choppy(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor)
    pattern_count = score_double(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor)
    pattern_count = score_trend(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor, shoe_position)
    pattern_count = score_big_road(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor)
    pattern_count = score_big_eye(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor)
    pattern_count = score_cockroach(recent, scores, reason_parts, pattern_insights, pattern_count, road_cache, position_factor)
    pattern_count = score_markov(recent, scores, reason_parts, pattern_insights, pattern_count, position_factor)

    recent_wins = recent[-6:] if len(recent) >= 6 else recent
    for i, result in enumerate(recent_wins):
        if result in ['Banker', 'Player']:
            weight = 0.5 ** ((len(recent_wins) - i - 1) / 20)
            scores[result] += 15 * weight
    reason_parts.append("Weighted recent momentum applied.")

    if total > 0:
        banker_ratio = freq['Banker'] / total
        player_ratio = freq['Player'] / total
        tie_ratio = freq['Tie'] / total
        scores['Banker'] += (banker_ratio * 0.9) * 25
        scores['Player'] += (player_ratio * 1.0) * 25
        scores['Tie'] += (tie_ratio * 0.6) * 25 if tie_ratio > 0.25 else 0
        reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
        pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    if entropy > 1.5:
        for key in scores:
            scores[key] *= 0.7
        reason_parts.append("High randomness detected; lowering pattern confidence.")
        pattern_insights.append("Randomness: High entropy")
        emotional_tone = "Cautious"

    if pattern_count >= 3:
        max_score = max(scores['Banker'], scores['Player'])
        if max_score > 0:
            coherence_bonus = 15 if pattern_count == 3 else 20
            max_bet = 'Banker' if scores['Banker'] > scores['Player'] else 'Player'
            scores[max_bet] += coherence_bonus
            reason_parts.append(f"Multiple patterns align on {max_bet} (+{coherence_bonus} bonus).")
            pattern_insights.append(f"Coherence: {pattern_count} patterns align")
        else:
            confidence_penalty = 15
            for key in scores:
                scores[key] = max(0, scores[key] - confidence_penalty)
            reason_parts.append("Conflicting patterns detected; reducing confidence.")
            emotional_tone = "Skeptical"

    bet_choice = max(scores, key=scores.get)
    confidence = min(round(max(scores.values(), default=0) * 1.3), 95)
    confidence_threshold = dynamic_confidence_threshold(current_bankroll, st.session_state.get('initial_bankroll', 1000.0), entropy, mode)

    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}% < {confidence_threshold}%). Passing.")
    elif mode == 'Conservative' and confidence < 75:
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence; proceeding cautiously.")

    if bet_choice == 'Tie' and (confidence < 85 or freq['Tie'] / total < 0.2):
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice] * 1.3), 95)
        reason_parts.append("Tie bet too risky; switching to safer option.")
        emotional_tone = "Cautious"

    if shoe_position > 60:
        confidence = max(confidence - 10, 40)
        reason_parts.append("Late in shoe; increasing caution.")
        emotional_tone = "Cautious"

    reason = " ".join(reason_parts)
    logging.info(f"advanced_bet_selection took {time.time() - start_time:.2f} seconds")
    return bet_choice, confidence, reason, emotional_tone, pattern_insights

@st.cache_data
def money_management(bankroll, base_bet, strategy, bet_outcome=None):
    min_bet = max(1.0, base_bet)
    max_bet = bankroll
    if bankroll < min_bet:
        logging.warning(f"Bankroll ({bankroll:.2f}) < min bet ({min_bet:.2f}).")
        return 0.0
    if strategy == "T3":
        if bet_outcome == 'win':
            if not st.session_state.get('t3_results', []):
                st.session_state.t3_level = max(1, st.session_state.get('t3_level', 1) - 1)
            st.session_state.t3_results = st.session_state.get('t3_results', []) + ['W']
        elif bet_outcome == 'loss':
            st.session_state.t3_results = st.session_state.get('t3_results', []) + ['L']
        if len(st.session_state.get('t3_results', [])) == 3:
            wins = st.session_state.t3_results.count('W')
            losses = st.session_state.t3_results.count('L')
            if wins > losses:
                st.session_state.t3_level = max(1, st.session_state.get('t3_level', 1) - 1)
            elif losses > wins:
                st.session_state.t3_level = st.session_state.get('t3_level', 1) + 1
            st.session_state.t3_results = []
        calculated_bet = base_bet * st.session_state.get('t3_level', 1)
    else:
        calculated_bet = base_bet
    bet_size = round(calculated_bet / base_bet) * base_bet
    return round(max(min_bet, min(bet_size, max_bet)), 2)

@st.cache_data
def calculate_bankroll(_history, base_bet, strategy, max_window=32):
    start_time = time.time()
    history = list(_history)
    history = history[-max_window:] if len(history) > max_window else history
    bankroll = st.session_state.initial_bankroll
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
            if bet == 'Banker':
                current_bankroll += bet_size * 0.95
            else:
                current_bankroll += bet_size
            if strategy == "T3":
                money_management(current_bankroll, base_bet, strategy, 'win')
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.get('pattern_performance', {}))
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
            if strategy == "T3":
                money_management(current_bankroll, base_bet, strategy, 'loss')
            update_pattern_performance(current_rounds, pattern_insights, bet, actual_result, st.session_state.get('pattern_performance', {}))
        bankroll_progress.append(current_bankroll)
    logging.info(f"calculate_bankroll took {time.time() - start_time:.2f} seconds")
    return bankroll_progress, bet_sizes

@st.cache_data
def calculate_win_loss_tracker(_history, base_bet, strategy, ai_mode, max_hands=20):
    start_time = time.time()
    history = list(_history)
    history = history[-max_hands:] if len(history) > max_hands else history
    tracker = []
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, _, _, _, _ = advanced_bet_selection(tuple(current_rounds[:-1]), ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if actual_result == 'Tie':
            tracker.append('T')
        elif bet in ('Pass', None):
            tracker.append('S')
        elif actual_result == bet:
            tracker.append('W')
            if strategy == 'T3':
                money_management(st.session_state.get('initial_bankroll', 1000.0), base_bet, strategy, 'win')
        else:
            tracker.append('L')
            if strategy == 'T3':
                money_management(st.session_state.get('initial_bankroll', 1000.0), base_bet, strategy, 'loss')
    logging.info(f"calculate_win_loss_tracker took {time.time() - start_time:.2f} seconds")
    return tracker

def add_result(result):
    st.session_state.history.append(result)

def undo_result():
    if st.session_state.history:
        st.session_state.history.pop()
        if st.session_state.get('money_management_strategy') == "T3":
            st.session_state.t3_results = []
            st.session_state.t3_level = 1

def main():
    start_time = time.time()
    try:
        st.set_page_config(page_title="Mang Baccarat Predictor", page_icon="🎲", layout="wide")
        st.title("Mang Baccarat Predictor")

        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'initial_bankroll' not in st.session_state:
            st.session_state.initial_bankroll = 1000.0
        if 'base_bet' not in st.session_state:
            st.session_state.base_bet = 10.0
        if 'money_management_strategy' not in st.session_state:
            st.session_state.money_management_strategy = "Flat Betting"
        if 'ai_mode' not in st.session_state:
            st.session_state.ai_mode = "Conservative"
        if 'selected_patterns' not in st.session_state:
            st.session_state.selected_patterns = ["Win/Loss"]
        if 't3_level' not in st.session_state:
            st.session_state.t3_level = 1
        if 't3_results' not in st.session_state:
            st.session_state.t3_results = []
        if 'screen_width' not in st.session_state:
            st.session_state.screen_width = 1024
        if 'pattern_performance' not in st.session_state:
            st.session_state.pattern_performance = {}

        # Adjust weights less frequently
        if len(st.session_state.history) % 10 == 0 or (len(st.session_state.history) >= 3 and calculate_win_loss_tracker(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.ai_mode)[-3:] == ['L', 'L', 'L']):
            adjust_pattern_weights(st.session_state.pattern_performance)

        # JavaScript for screen width
        st.markdown("""
            <script>
            function updateScreenWidth() {
                document.getElementById('screen-width-input').value = window.innerWidth;
            }
            window.onload = updateScreenWidth;
            window.onresize = updateScreenWidth;
            </script>
            <input type="hidden" id="screen-width-input">
        """, unsafe_allow_html=True)

        screen_width_input = st.text_input("Screen Width", key="screen_width_input", value=str(st.session_state.screen_width), disabled=True)
        st.session_state.screen_width = int(screen_width_input) if screen_width_input.isdigit() else 1024

        # CSS for styling
        st.markdown("""
            <style>
            .pattern-scroll { overflow-x: auto; white-space: nowrap; max-width: 100%; padding: 10px; border: 1px solid #e1e1e1; background-color: #f9f9f9; }
            .pattern-scroll::-webkit-scrollbar { height: 8px; }
            .pattern-scroll::-webkit-scrollbar-thumb { background-color: #888; border-radius: 4px; }
            .stButton > button { width: 100%; padding: 8px; margin: 5px 0; }
            .stNumberInput, .stSelectbox { width: 100% !important; }
            .pattern-circle { width: 20px; height: 20px; display: inline-block; margin: 2px; }
            .display-circle { width: 20px; height: 20px; display: inline-block; margin: 2px; }
            @media (max-width: 768px) {
                .pattern-circle, .display-circle { width: 16px !important; height: 16px !important; }
                .stButton > button { font-size: 0.9rem; padding: 6px; }
            }
            </style>
        """, unsafe_allow_html=True)

        # Game Settings
        with st.expander("Game Settings", expanded=False):
            cols = st.columns(4)
            with cols[0]:
                initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
            with cols[1]:
                base_bet = st.number_input("Base Bet", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
            with cols[2]:
                strategy_options = ["Flat Betting", "T3"]
                money_management_strategy = st.selectbox("Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
            with cols[3]:
                ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

            st.session_state.initial_bankroll = initial_bankroll
            st.session_state.base_bet = base_bet
            st.session_state.money_management_strategy = money_management_strategy
            st.session_state.ai_mode = ai_mode

        # Pattern Weights
        with st.expander("Pattern Weights", expanded=False):
            st.markdown("**Pattern Weights (Auto-Adjusted)**")
            for pattern, weight in st.session_state.PATTERN_WEIGHTS.items():
                perf = st.session_state.pattern_performance.get(pattern, {'correct': 0, 'total': 0})
                accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
                st.markdown(f"{pattern.capitalize()}: {weight:.2f} (Accuracy: {accuracy:.2%})")
            if st.button("Reset Weights"):
                st.session_state.PATTERN_WEIGHTS = {
                    'streak': 1.2, 'alternating': 1.0, 'zigzag': 0.8, 'trend': 0.9,
                    'big_road': 0.7, 'big_eye': 0.6, 'cockroach': 0.5, 'choppy': 0.8,
                    'double': 0.7, 'markov': 0.9
                }
                st.rerun()

        # Input Game Results
        with st.expander("Input Game Results", expanded=True):
            cols = st.columns(4)
            with cols[0]:
                st.button("Player", on_click=add_result, args=("Player",))
            with cols[1]:
                st.button("Banker", on_click=add_result, args=("Banker",))
            with cols[2]:
                st.button("Tie", on_click=add_result, args=("Tie",))
            with cols[3]:
                st.button("Undo", on_click=undo_result, disabled=len(st.session_state.history) == 0)

        # Shoe Patterns
        with st.expander("Shoe Patterns", expanded=False):
            pattern_options = ["Bead Bin", "Big Road", "Big Eye", "Cockroach", "Win/Loss"]
            selected_patterns = st.multiselect("Select Patterns", pattern_options, default=st.session_state.selected_patterns, key="pattern_select")
            st.session_state.selected_patterns = selected_patterns
            max_display_cols = 8 if st.session_state.screen_width < 768 else 12

            if "Bead Bin" in selected_patterns:
                st.markdown("### Bead Bin")
                sequence = st.session_state.history[-72:]
                sequence = ['P' if r == 'Player' else 'B' if r == 'Banker' else 'T' for r in sequence]
                grid = [['' for _ in range(max_display_cols)] for _ in range(6)]
                for i, result in enumerate(sequence):
                    if result in ['P', 'B', 'T']:
                        col = i // 6
                        row = i % 6
                        if col < max_display_cols:
                            color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                            grid[row][col] = f'<div class="pattern-circle" style="background-color: {color}; border-radius: 50%; border: 1px solid #fff;"></div>'
                st.markdown('<div class="pattern-scroll">', unsafe_allow_html=True)
                for row in grid:
                    st.markdown(' '.join(row), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if "Big Road" in selected_patterns:
                st.markdown("### Big Road")
                big_road_grid, num_cols = build_big_road(tuple(st.session_state.history))
                if num_cols > 0:
                    display_cols = min(num_cols, max_display_cols)
                    st.markdown('<div class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = big_road_grid[row][col]
                            if outcome == 'P':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #3182ce; border-radius: 50%; border: 1px solid #fff;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #e53e3e; border-radius: 50%; border: 1px solid #fff;"></div>')
                            elif outcome == 'T':
                                row_display.append(f'<div class="pattern-circle" style="border: 2px solid #38a169; border-radius: 50%;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Big Eye" in selected_patterns:
                st.markdown("### Big Eye Boy")
                st.markdown("<p style='font-size: 12px; color: #666;'>Red (🔴): Repeat, Blue (🔵): Break</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = build_big_road(tuple(st.session_state.history))
                big_eye_grid, big_eye_cols = build_big_eye_boy(big_road_grid, num_cols)
                if big_eye_cols > 0:
                    display_cols = min(big_eye_cols, max_display_cols)
                    st.markdown('<div class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = big_eye_grid[row][col]
                            if outcome == 'R':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #e53e3e; border-radius: 50%; border: 1px solid #000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #3182ce; border-radius: 50%; border: 1px solid #000;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Cockroach" in selected_patterns:
                st.markdown("### Cockroach Pig")
                st.markdown("<p style='font-size: 12px; color: #666;'>Red (🔴): Repeat, Blue (🔵): Break</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = build_big_road(tuple(st.session_state.history))
                cockroach_grid, cockroach_cols = build_cockroach_pig(big_road_grid, num_cols)
                if cockroach_cols > 0:
                    display_cols = min(cockroach_cols, max_display_cols)
                    st.markdown('<div class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = cockroach_grid[row][col]
                            if outcome == 'R':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #e53e3e; border-radius: 50%; border: 1px solid #000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #3182ce; border-radius: 50%; border: 1px solid #000;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Win/Loss" in selected_patterns:
                st.markdown("### Win/Loss")
                st.markdown("<p style='font-size: 12px; color: #666;'>Green (🟢): Win, Red (🔴): Loss, Gray (⬜): Skip/Tie</p>", unsafe_allow_html=True)
                tracker = calculate_win_loss_tracker(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.ai_mode)[-max_display_cols:]
                row_display = []
                for result in tracker:
                    color = '#38a169' if result == 'W' else '#e53e3e' if result == 'L' else '#A0AEC0'
                    row_display.append(f'<div class="pattern-circle" style="background-color: {color}; border-radius: 50%; border: 1px solid #000;"></div>')
                st.markdown('<div class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Prediction
        with st.expander("Prediction", expanded=True):
            bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(tuple(st.session_state.history), st.session_state.ai_mode)
            st.markdown("### Prediction")
            current_bankroll = calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
            recommended_bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy)
            if current_bankroll < max(1.0, st.session_state.base_bet):
                st.warning("Insufficient bankroll to bet. Increase bankroll or reset.")
                bet = 'Pass'
                confidence = 0
                reason = "Bankroll too low."
                emotional_tone = "Cautious"
            if bet == 'Pass':
                st.markdown("**No Bet**: Insufficient confidence, bankroll, or three consecutive losses.")
            else:
                st.markdown(f"**Bet**: {bet} | **Confidence**: {confidence}% | **Bet Size**: ${recommended_bet_size:.2f} | **Mood**: {emotional_tone}")
            st.markdown(f"**Reasoning**: {reason}")
            if pattern_insights:
                st.markdown("### Pattern Insights")
                for insight in pattern_insights:
                    st.markdown(f"- {insight}")

        # Pattern Performance
        with st.expander("Pattern Performance", expanded=False):
            st.markdown("### Pattern Contributions")
            for insight in pattern_insights:
                pattern = insight.split(':')[0].lower()
                weight = st.session_state.PATTERN_WEIGHTS.get(pattern, 1.0)
                perf = st.session_state.pattern_performance.get(pattern, {'correct': 0, 'total': 0})
                accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0
                st.markdown(f"- {insight} (Weight: {weight:.2f}, Accuracy: {accuracy:.2%})")

        # Bankroll Progress
        with st.expander("Bankroll Progress", expanded=True):
            bankroll_progress, bet_sizes = calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)
            bankroll_progress = bankroll_progress[-50:] if len(bankroll_progress) > 50 else bankroll_progress
            bet_sizes = bet_sizes[-50:] if len(bet_sizes) > 50 else bet_sizes
            if bankroll_progress:
                st.markdown("### Bankroll Progress")
                for i, (val, bet_size) in enumerate(zip(bankroll_progress[::-1], bet_sizes[::-1])):
                    hand = len(bankroll_progress) - i
                    bet_display = f"Bet ${bet_size:.2f}" if bet_size > 0 else "No Bet"
                    st.markdown(f"Hand {hand}: ${val:.2f} | {bet_display}")
                st.markdown(f"**Current Bankroll**: ${bankroll_progress[-1]:.2f}")

                st.markdown("### Bankroll Chart")
                labels = [f"Hand {i+1}" for i in range(len(bankroll_progress))]
                fig = go.Figure(go.Scatter(x=labels, y=bankroll_progress, mode='lines+markers', name='Bankroll', line=dict(color='#38a169'), marker=dict(size=6)))
                fig.update_layout(title="Bankroll Over Time", xaxis_title="Hand", yaxis_title="Bankroll ($)", xaxis=dict(tickangle=45), yaxis=dict(autorange=True), height=400, margin=dict(l=40, r=40, t=50, b=100))
                st.plotly_chart(fig, use_container_width=True)

        # Reset
        with st.expander("Reset", expanded=False):
            if st.button("New Game"):
                final_bankroll = calculate_bankroll(tuple(st.session_state.history), st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
                st.session_state.history = []
                st.session_state.initial_bankroll = max(1.0, final_bankroll)
                st.session_state.base_bet = min(10.0, st.session_state.initial_bankroll)
                st.session_state.money_management_strategy = "Flat Betting"
                st.session_state.ai_mode = "Conservative"
                st.session_state.selected_patterns = ["Win/Loss"]
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.session_state.pattern_performance = {}
                st.session_state.PATTERN_WEIGHTS = {
                    'streak': 1.2, 'alternating': 1.0, 'zigzag': 0.8, 'trend': 0.9,
                    'big_road': 0.7, 'big_eye': 0.6, 'cockroach': 0.5, 'choppy': 0.8,
                    'double': 0.7, 'markov': 0.9
                }
                st.rerun()

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        st.error(f"Error: {str(e)}. Try refreshing or resetting.")
    logging.info(f"Main loop took {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
