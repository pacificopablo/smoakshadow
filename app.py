
import streamlit as st
import logging
import plotly.graph_objects as go
import math

# Set up basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Normalize input
def normalize(s):
    s = s.strip().lower()
    if s == 'banker' or s == 'b':
        return 'Banker'
    if s == 'player' or s == 'p':
        return 'Player'
    if s == 'tie' or s == 't':
        return 'Tie'
    return None

def detect_streak(s):
    if not s:
        return None, 0
    last = s[-1]
    count = 1
    for i in range(len(s) - 2, -1, -1):
        if s[i] == last:
            count += 1
        else:
            break
    return last, count

def is_alternating(s, min_length=4):
    if len(s) < min_length:
        return False
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            return False
    return True

def is_zigzag(s):
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i + 2] and s[i] != s[i + 1]:
            return True
    return False

def is_choppy(s, window=10, min_alternations=6):
    if len(s) < window:
        return False
    recent = s[-window:]
    alternations = sum(1 for i in range(len(recent) - 1) if recent[i] != recent[i + 1])
    return alternations >= min_alternations

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
    count = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in s:
        if r in count:
            count[r] += 1
    return count

def build_big_road(s):
    if not s:
        return [['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)]], 0
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

def build_big_eye_boy(big_road_grid, num_cols):
    if num_cols < 3:
        return [['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)]], 0
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

def build_cockroach_pig(big_road_grid, num_cols):
    if num_cols < 4:
        return [['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)], ['' for _ in range(50)]], 0
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

def advanced_bet_selection(s, mode='Conservative'):
    max_recent_count = 40
    recent = s[-max_recent_count:] if len(s) >= max_recent_count else s
    if len(recent) < 5:
        return 'Pass', 0, "Not enough history to make a confident prediction.", "Cautious", [], {}, []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    pattern_scores = {'Banker': {}, 'Player': {}, 'Tie': {}}  # Track pattern contributions
    reason_parts = []
    pattern_insights = []
    pattern_keys = []  # Store pattern names for accuracy tracking
    emotional_tone = "Neutral"
    pattern_count = 0
    shoe_position = len(s)
    max_patterns = 8  # Streak, BigRoad, BigEye, Cockroach, Alternating, Zigzag, Trend, Choppy

    def decay_weight(index, total_length, half_life=20):
        return 0.5 ** ((total_length - index - 1) / half_life)

    def get_pattern_accuracy(pattern):
        return st.session_state.pattern_accuracy[pattern]['correct'] / st.session_state.pattern_accuracy[pattern]['total'] if st.session_state.pattern_accuracy[pattern]['total'] > 0 else 1

    # Grid caching
    history_key = ''.join(s[-max_recent_count:]) if s else ''
    if 'grid_cache' not in st.session_state or st.session_state.grid_cache.get('history_key') != history_key:
        st.session_state.grid_cache = {
            'history_key': history_key,
            'big_road': build_big_road(recent),
            'big_eye': None,
            'cockroach': None
        }
        st.session_state.grid_cache['big_eye'] = build_big_eye_boy(*st.session_state.grid_cache['big_road'])
        st.session_state.grid_cache['cockroach'] = build_cockroach_pig(*st.session_state.grid_cache['big_road'])

    # Shoe position factor
    shoe_position_factor = 1.2 if shoe_position < 20 else 0.7 if shoe_position > 50 else 1.0

    # Streak detection
    streak_value, streak_length = detect_streak(recent)
    if streak_length >= 3 and streak_value != "Tie":
        streak_score = min(25 + (streak_length - 3) * 8, 50)
        if streak_length >= 6:
            streak_score += 10
            pattern_insights.append(f"Dragon Tail: {streak_length} {streak_value}")
            emotional_tone = "Confident"
        accuracy = get_pattern_accuracy('Streak')
        scores[streak_value] += streak_score * accuracy * shoe_position_factor
        pattern_scores[streak_value]['Streak'] = streak_score * accuracy
        reason_parts.append(f"Streak of {streak_length} {streak_value} wins detected (weighted by {accuracy:.2f}).")
        pattern_insights.append(f"Streak: {streak_length} {streak_value}")
        pattern_keys.append('Streak')
        pattern_count += 1
        if streak_length >= 5 and mode == 'Aggressive':
            contrarian_bet = 'Player' if streak_value == 'Banker' else 'Banker'
            contrarian_score = 20 * get_pattern_accuracy('Streak')
            scores[contrarian_bet] += contrarian_score
            pattern_scores[contrarian_bet]['Streak'] = contrarian_score
            reason_parts.append(f"Long streak ({streak_length}); considering break in Aggressive mode.")
            pattern_insights.append("Possible streak break")
            emotional_tone = "Skeptical"

    # Alternating pattern
    if len(recent) >= 6 and is_alternating(recent[-6:], min_length=6):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        alternate_score = 35 * get_pattern_accuracy('Alternating') * shoe_position_factor
        scores[alternate_bet] += alternate_score
        pattern_scores[alternate_bet]['Alternating'] = alternate_score
        reason_parts.append(f"Strong alternating pattern (Ping Pong) in last 6 hands (weighted by {get_pattern_accuracy('Alternating'):.2f}).")
        pattern_insights.append("Ping Pong: Alternating P/B")
        pattern_keys.append('Alternating')
        pattern_count += 1
        emotional_tone = "Excited"

    # Zigzag pattern
    if is_zigzag(recent[-8:]):
        last = recent[-1]
        zigzag_bet = 'Player' if last == 'Banker' else 'Banker'
        zigzag_score = (30 if shoe_position < 30 else 20) * get_pattern_accuracy('Zigzag') * shoe_position_factor
        scores[zigzag_bet] += zigzag_score
        pattern_scores[zigzag_bet]['Zigzag'] = zigzag_score
        reason_parts.append(f"Zigzag pattern (P-B-P or B-P-B) detected in last 8 hands (weighted by {get_pattern_accuracy('Zigzag'):.2f}).")
        pattern_insights.append("Zigzag: P-B-P/B-P-B")
        pattern_keys.append('Zigzag')
        pattern_count += 1
        emotional_tone = "Curious"

    # Choppy pattern
    if is_choppy(recent, window=10):
        last = recent[-1]
        choppy_bet = 'Player' if last == 'Banker' else 'Banker'
        choppy_score = 20 * get_pattern_accuracy('Choppy') * shoe_position_factor
        scores[choppy_bet] += choppy_score
        pattern_scores[choppy_bet]['Choppy'] = choppy_score
        reason_parts.append(f"Choppy pattern detected in last 10 hands (weighted by {get_pattern_accuracy('Choppy'):.2f}).")
        pattern_insights.append("Choppy: Frequent alternations")
        pattern_keys.append('Choppy')
        pattern_count += 1
        emotional_tone = "Observant"

    # Recent trend
    trend_bet, trend_score = recent_trend(recent, window=12)
    if trend_bet:
        trend_weight = trend_score * (1.2 if shoe_position < 20 else 0.8) * get_pattern_accuracy('Trend')
        scores[trend_bet] += min(trend_weight, 35) * shoe_position_factor
        pattern_scores[trend_bet]['Trend'] = min(trend_weight, 35)
        reason_parts.append(f"Recent trend favors {trend_bet} in last 12 hands (weighted by {get_pattern_accuracy('Trend'):.2f}).")
        pattern_insights.append(f"Trend: {trend_bet} dominance")
        pattern_keys.append('Trend')
        pattern_count += 1
        emotional_tone = "Hopeful"

    # Big Road
    big_road_grid, num_cols = st.session_state.grid_cache['big_road']
    if num_cols > 0:
        last_col = [big_road_grid[row][num_cols - 1] for row in range(6)]
        col_length = sum(1 for x in last_col if x in ['P', 'B'])
        if col_length >= 3:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            col_score = (25 if col_length == 3 else 35 if col_length == 4 else 45) * get_pattern_accuracy('BigRoad') * shoe_position_factor
            scores[bet_side] += col_score
            pattern_scores[bet_side]['Big Road'] = col_score
            reason_parts.append(f"Big Road column of {col_length} {bet_side} (weighted by {get_pattern_accuracy('BigRoad'):.2f}).")
            pattern_insights.append(f"Big Road: {col_length} {bet_side}")
            pattern_keys.append('BigRoad')
            pattern_count += 1

    # Big Eye Boy
    big_eye_grid, big_eye_cols = st.session_state.grid_cache['big_eye']
    if big_eye_cols > 1:
        last_two_cols = [[big_eye_grid[row][c] for row in range(6)] for c in range(big_eye_cols - 2, big_eye_cols)]
        last_signals = [next((x for x in col if x in ['R', 'B']), None) for col in last_two_cols]
        if all(s == 'R' for s in last_signals if s):
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            big_eye_score = 20 * get_pattern_accuracy('BigEye') * shoe_position_factor
            scores[last_side] += big_eye_score
            pattern_scores[last_side]['Big Eye Boy'] = big_eye_score
            reason_parts.append(f"Big Eye Boy shows consistent repeat pattern (weighted by {get_pattern_accuracy('BigEye'):.2f}).")
            pattern_insights.append("Big Eye Boy: Consistent repeat")
            pattern_keys.append('BigEye')
            pattern_count += 1
        elif all(s == 'B' for s in last_signals if s):
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            big_eye_score = 15 * get_pattern_accuracy('BigEye') * shoe_position_factor
            scores[opposite_side] += big_eye_score
            pattern_scores[opposite_side]['Big Eye Boy'] = big_eye_score
            reason_parts.append(f"Big Eye Boy shows consistent break pattern (weighted by {get_pattern_accuracy('BigEye'):.2f}).")
            pattern_insights.append("Big Eye Boy: Consistent break")
            pattern_keys.append('BigEye')
            pattern_count += 1

    # Cockroach Pig
    cockroach_grid, cockroach_cols = st.session_state.grid_cache['cockroach']
    if cockroach_cols > 1:
        last_two_cols = [[cockroach_grid[row][c] for row in range(6)] for c in range(cockroach_cols - 2, cockroach_cols)]
        last_signals = [next((x for x in col if x in ['R', 'B']), None) for col in last_two_cols]
        if all(s == 'R' for s in last_signals if s):
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            cockroach_score = 15 * get_pattern_accuracy('Cockroach') * shoe_position_factor
            scores[last_side] += cockroach_score
            pattern_scores[last_side]['Cockroach Pig'] = cockroach_score
            reason_parts.append(f"Cockroach Pig shows consistent repeat pattern (weighted by {get_pattern_accuracy('Cockroach'):.2f}).")
            pattern_insights.append("Cockroach Pig: Consistent repeat")
            pattern_keys.append('Cockroach')
            pattern_count += 1
        elif all(s == 'B' for s in last_signals if s):
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            cockroach_score = 12 * get_pattern_accuracy('Cockroach') * shoe_position_factor
            scores[opposite_side] += cockroach_score
            pattern_scores[opposite_side]['Cockroach Pig'] = cockroach_score
            reason_parts.append(f"Cockroach Pig shows consistent break pattern (weighted by {get_pattern_accuracy('Cockroach'):.2f}).")
            pattern_insights.append("Cockroach Pig: Consistent break")
            pattern_keys.append('Cockroach')
            pattern_count += 1

    # Entropy calculation
    freq = frequency_count(recent)
    total = len(recent)
    if total > 0:
        entropy = -sum((count / total) * math.log2(count / total) for count in freq.values() if count > 0)
        if entropy > 1.8:
            for key in scores:
                scores[key] *= 0.85
            reason_parts.append("High randomness detected; lowering pattern confidence.")
            pattern_insights.append("Randomness: High entropy")
            emotional_tone = "Cautious"

    # Recent momentum
    recent_wins = recent[-6:] if len(recent) >= 6 else recent
    for i, result in enumerate(recent_wins):
        if result in ['Banker', 'Player']:
            weight = decay_weight(i, len(recent_wins))
            scores[result] += 15 * weight
            pattern_scores[result]['Momentum'] = pattern_scores[result].get('Momentum', 0) + 15 * weight
    reason_parts.append("Weighted recent momentum applied.")

    # Long-term frequency
    if total > 0:
        banker_ratio = freq['Banker'] / total
        player_ratio = freq['Player'] / total
        tie_ratio = freq['Tie'] / total
        scores['Banker'] += (banker_ratio * 0.9) * 25
        scores['Player'] += (player_ratio * 1.0) * 25
        scores['Tie'] += (tie_ratio * 0.6) * 25 if tie_ratio > 0.25 else 0
        pattern_scores['Banker']['Frequency'] = (banker_ratio * 0.9) * 25
        pattern_scores['Player']['Frequency'] = (player_ratio * 1.0) * 25
        pattern_scores['Tie']['Frequency'] = (tie_ratio * 0.6) * 25 if tie_ratio > 0.25 else 0
        reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
        pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    # Pattern coherence
    if pattern_count >= 3:
        top_score = max(scores.values(), default=0)
        second_score = max([s for s in scores.values() if s != top_score], default=0)
        if top_score > 0:
            coherence_bonus = 15 if pattern_count == 3 else 20
            max_bet = max(scores, key=scores.get)
            scores[max_bet] += coherence_bonus
            pattern_scores[max_bet]['Coherence'] = coherence_bonus
            reason_parts.append(f"Multiple patterns align on {max_bet} (+{coherence_bonus} bonus).")
            pattern_insights.append(f"Coherence: {pattern_count} patterns align")
        elif top_score - second_score < 10:
            confidence_penalty = 15
            for key in scores:
                scores[key] = max(0, scores[key] - confidence_penalty)
            reason_parts.append("Conflicting patterns detected; reducing confidence.")
            emotional_tone = "Skeptical"

    # Determine bet choice and confidence
    bet_choice = max(scores, key=scores.get)
    top_score = max(scores.values(), default=0)
    second_score = max([s for s in scores.values() if s != top_score], default=0)
    confidence = min(round(100 * (top_score - second_score) / (top_score + 1e-10) * pattern_count / max_patterns), 95)
    if entropy > 1.8:
        confidence = max(confidence - 20, 20)
        reason_parts.append("High entropy reduces confidence.")

    # Mode-specific adjustments
    confidence_threshold = 65 if mode == 'Conservative' else 45
    if shoe_position > 50:
        confidence_threshold += 10
        reason_parts.append("Late in shoe; increasing caution.")
        emotional_tone = "Cautious"

    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}% < {confidence_threshold}%). Passing.")
    elif mode == 'Conservative' and confidence < 75:
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence; proceeding cautiously.")

    # Restrict Tie bets
    if bet_choice == 'Tie':
        recent_ties = sum(1 for r in recent[-10:] if r == 'Tie')
        if recent_ties < 3 or freq['Tie'] / total < 0.25:
            scores['Tie'] = 0
            bet_choice = max(scores, key=scores.get)
            top_score = max(scores.values(), default=0)
            second_score = max([s for s in scores.values() if s != top_score], default=0)
            confidence = min(round(100 * (top_score - second_score) / (top_score + 1e-10) * pattern_count / max_patterns), 95)
            reason_parts.append("Tie bet rejected; insufficient recent Ties or frequency.")
            emotional_tone = "Cautious"

    reason = " ".join(reason_parts)
    return bet_choice, confidence, reason, emotional_tone, pattern_insights, pattern_scores, pattern_keys

def money_management(bankroll, base_bet, strategy, bet_outcome=None):
    if bankroll <= 0:
        return 0.0
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if bankroll < min_bet:
        logging.warning(f"Bankroll ({bankroll:.2f}) is less than minimum bet ({min_bet:.2f}).")
        return 0.0

    if strategy == "T3":
        if bet_outcome == 'win':
            if not st.session_state.t3_results:  # First step in T3 cycle
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
            st.session_state.t3_results.append('W')
        elif bet_outcome == 'loss':
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

        calculated_bet = base_bet * st.session_state.t3_level
    else:
        calculated_bet = base_bet

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2)

def calculate_bankroll(history, base_bet, strategy):
    bankroll = st.session_state.initial_bankroll
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    # Use temporary T3 state to avoid modifying global state
    t3_level = 1
    t3_results = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, _, _, _ = advanced_bet_selection(current_rounds[:-1], st.session_state.ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [], {}, [])
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie'):
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        # Pass temporary T3 state
        bet_size = money_management(current_bankroll, base_bet, strategy)
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
                if not t3_results:
                    t3_level = max(1, t3_level - 1)
                t3_results.append('W')
                if len(t3_results) == 3:
                    wins = t3_results.count('W')
                    losses = t3_results.count('L')
                    if wins > losses:
                        t3_level = max(1, t3_level - 1)
                    elif losses > wins:
                        t3_level += 1
                    t3_results = []
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
            if strategy == "T3":
                t3_results.append('L')
                if len(t3_results) == 3:
                    wins = t3_results.count('W')
                    losses = t3_results.count('L')
                    if wins > losses:
                        t3_level = max(1, t3_level - 1)
                    elif losses > wins:
                        t3_level += 1
                    t3_results = []
        bankroll_progress.append(current_bankroll)
    return bankroll_progress, bet_sizes

def calculate_win_loss_tracker(history, base_bet, strategy, ai_mode):
    tracker = []
    # Use temporary T3 state
    t3_level = 1
    t3_results = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, _, _, _, patterns_insights, _, pattern_keys = advanced_bet_selection(current_rounds[:-1], ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [], {}, [])
        actual_result = history[i]
        if bet not in (None, 'Pass'):
            for pattern in pattern_keys:
                if pattern in st.session_state.pattern_accuracy:
                    st.session_state.pattern_accuracy[pattern]['total'] += 1
                    if actual_result == bet:
                        st.session_state.pattern_accuracy[pattern]['correct'] += 1
        if actual_result == 'Tie':
            tracker.append('T')
        elif bet in (None, 'Pass'):
            tracker.append('S')
        elif actual_result == bet:
            tracker.append('W')
            if strategy == "T3":
                if not t3_results:
                    t3_level = max(1, t3_level - 1)
                t3_results.append('W')
                if len(t3_results) == 3:
                    wins = t3_results.count('W')
                    losses = t3_results.count('L')
                    if wins > losses:
                        t3_level = max(1, t3_level - 1)
                    elif losses > wins:
                        t3_level += 1
                    t3_results = []
        else:
            tracker.append('L')
            if strategy == "T3":
                t3_results.append('L')
                if len(t3_results) == 3:
                    wins = t3_results.count('W')
                    losses = t3_results.count('L')
                    if wins > losses:
                        t3_level = max(1, t3_level - 1)
                    elif losses > wins:
                        t3_level += 1
                    t3_results = []
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
            st.session_state.money_management_strategy = "Flat Betting"
        if 'ai_mode' not in st.session_state:
            st.session_state.ai_mode = "Conservative"
        if 'selected_patterns' not in st.session_state:
            st.session_state.selected_patterns = ["Bead Bin", "Win/Loss"]
        if 't3_level' not in st.session_state:
            st.session_state.t3_level = 1
        if 't3_results' not in st.session_state:
            st.session_state.t3_results = []
        if 'screen_width' not in st.session_state:
            st.session_state.screen_width = 1024
        if 'pattern_accuracy' not in st.session_state:
            st.session_state.pattern_accuracy = {
                'Streak': {'correct': 0, 'total': 0},
                'BigRoad': {'correct': 0, 'total': 0},
                'BigEye': {'correct': 0, 'total': 0},
                'Cockroach': {'correct': 0, 'total': 0},
                'Alternating': {'correct': 0, 'total': 0},
                'Zigzag': {'correct': 0, 'total': 0},
                'Trend': {'correct': 0, 'total': 0},
                'Choppy': {'correct': 0, 'total': 0}
            }
        if 'prediction_accuracy' not in st.session_state:
            st.session_state.prediction_accuracy = {
                'Banker': {'correct': 0, 'total': 0},
                'Player': {'correct': 0, 'total': 0},
                'Tie': {'correct': 0, 'total': 0}
            }
        if 'grid_cache' not in st.session_state:
            st.session_state.grid_cache = {}

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
                font-size: 2.25rem;
                text-align: center;
            }
            h3 {
                font-size: 1.5rem;
            }
            p, div, span {
                font-size: 18px;
            }
            .pattern-circle {
                width: 22px;
                height: 24px;
                display: inline-block;
                margin: 2px;
            }
            .display-circle {
                width: 22px;
                height: 24px;
                display: inline-block;
                margin: 2px;
            }
            @media (min-width: 769px) {
                .stButton > button, .stNumberInput, .stSelectbox {
                    width: 200px;
                }
            }
            @media (max-width: 768px) {
                h1 {
                    font-size: 1.8rem;
                }
                h3 {
                    font-size: 1.2rem;
                }
                p, div, .span {
                    font-size: 16px;
                }
                .pattern-circle, .display-circle {
                    width: 16px !important;
                    height: 16px !important;
                }
                .stButton > button {
                    font-size: 14px;
                    padding: 6px;
                }
                .stNumberInput input, .stSelectbox div {
                    font-size: 14px;
                }
                .st-emotion-cache-1dj3wfg {
                    flex-wrap: wrap;
                }
            }
            @media (max-width: 500px) {
                .pattern-scroll {
                    max-width: 100%;
                }
                .pattern-circle, .display-circle {
                    width: 12px !important;
                    height: 12px !important;
                }
            }
            </script>
            <style>
                function autoScrollPatterns() {
                    const containers = ['bead-bin-scroll', 'big-road-scroll', 'big-eye-scroll', 'cockroach-scroll', 'win-loss-scroll'];
                    containers.forEach(id => {
                        const element = document.getElementById(id);
                        if (element) {
                            element.scrollLeft = element.scrollWidth;
                        }
                        );
                    return
                }
                window.onload = autoScrollPatterns;
            }
            </script>
        """, unsafe_allow_html=True)

        with st.expander("Game Settings", expanded=True):
            cols = st.columns(4]
            with cols[0]:
                initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
            with cols[1]:
                base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=0.2, format="%.2f")
            with cols[2]:
                strategy_options = ["Flat Betting", "T3"]
                money_management_strategy = st.selectbox("Money Management Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
                st.markdown("<p style='font-size: 14px; font-family: Times New Roman, Times, serif; color: #666666; margin-top: 8px;'>*Flat Betting: Fixed bet size.<br>T3: Adjusts bet level based on outcomes of last three bets (up if more losses, down if more wins or first-step win).</p>*</p>", unsafe_allow_html=True)

            with cols[3]:
                ai_mode = st.selectbox("Choose Your Style", ["Conservative", "Balanced", "Aggressive"], index=["Conservative", "Balanced", "Aggressive"].index(st.session_state.ai_mode))

            st.session_state.initial_bankroll = initial_bankroll
            st.session_state.base_bet = base_bet
            st.session_state.money_management_strategy = money_management_strategy
            st.session_state.ai_mode = = ai_mode

            st.markdown(f"**Selected Strategy**: {money_management_strategy}")

        with st.expander("Input Game Results", expanded=True):
            cols = st.columns[4]
            with cols[0]:
                if st.button("Add Player"):
                    st.session_state['history'].append("Player")
                    st.rerun()
            with cols[1]:
                if st.button("Add Banker"):
                    st.session_state['history'].append("Banker")
                    st.rerun()
            with cols[2]:
                if st.button("Add Tie"):
                    st.session_state['history'].append("Tie")
                    st.rerun()
            with cols[3]:
                undo_clicked = st.button("Undo", disabled=len(st.session_state.history) == 0)
                if undo_clicked and len(st.session_state.history) == 0:
                    st.warning("No results to undo!")
                elif undo_clicked:
                    st.session_state.history.pop()
                    if st.session_state.money_management_strategy == "T3":
                        st.session_state.t3_results = []
                        st.session_state['t3_level'] = 1
                    st.rerun()

        with st.expander("Shoe Patterns", expanded=False):
            pattern_options = ["Bead Bin", "Big Road", "BigEye Boy", "Cockroach Pig", "Win/Loss"]
            selected_patterns = st.multiselect(
                "Select Patterns to Display",
                pattern_options,
                default=st.session_state.selected_patterns,
                key="pattern_key"
            )
            st.session_state.selected_patterns = selected_patterns

            max_display_cols = 6 if st.session_state.screen_width <= 500 else 10 if st.session_state.screen_width < 768 else 8

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
                            color = '#3182ce' if result == 'P' else '#000000' if result == 'B' else '#660066'
                            grid[row][col] = f'<div class="pattern-circle" style="background-color: {color}; border-radius: 50%; border: 2px solid #333; color: #fff; text-align: center; line-height: 30px; font-size: bold;">{result}</div>'
                st.markdown('<div id="bead-bin-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                for row in grid:
                    st.markdown(' '.join(row), unsafe_allow_html=True)
                st.markdown('</div>')
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
                            outcome = big_road_grid[row][col]:
                                if outcome == 'P':
                                    row_display.append(f'<div class="pattern-circle" style="background-color: blue; border-radius: 50%; border: 2px solid #333;"></div>')
                                else:
                                    outcome == 'B':
                                    row_display.append(f'<div class="pattern-circle" style="background-color: #000000; border-radius: 50%; border: 2px solid #333;"></div>')
                                else:
                                    outcome == 'T':
                                        row_display.append(f'<div class="pattern-circle" style="border: 2px solid #660066; border-radius: 50%;"></div>')
                                    row_display.append('')
                                else:
                                    row_display.append(f'<div class="display-circle"></div>')
                            st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>')
                else:
                    st.markdown("No valid Big Road data available.")

            if "BigEye Boy" in st.session_state.selected_patterns:
                st.markdown("### Big Eye Boy")
                st.markdown("<p style='font-size: 14px; color: #666666;'>Red (🔴): Repeat Pattern, Blue (🔵): Break Pattern</p>", unsafe_allow_html=True)
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
                                row_display.append(f'<div class="pattern-circle" style="background-color: #ff0000; border-radius: 50%; border: 2px solid #333;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #0000ff; border-radius: 50%; border: 2px solid #333;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>')
                else:
                    st.markdown("No recent Big Eye data.")

            if "Cockroach Pig" in st.session_state.selected_patterns:
                st.markdown("### Cockroach Pig")
                st.markdown("<p style='font-size: 14px; color: #666666;'>Red (🔴): Repeat Pattern, Blue (🔵): Break Pattern</p>", unsafe_allow_html=True)
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
                                row_display.append(f'<div class="pattern-circle" style="background-color: #ff0000; border-radius: 50%; border: 2px solid #333;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #0000ff; border-radius: 50%; border: 2px solid #333;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>')
                else:
                    st.markdown("No recent Cockroach data.")

            if "Win/Loss" in st.session_state.selected_patterns:
                st.markdown("### Win/Loss")
                st.markdown("<p style='font-size: 14px; color: #666666;'>Green (🟢): Win, Red (🔴): Loss, Gray (⬜): Skip or Tie</p>", unsafe_allow_html=True)
                tracker = calculate_win_loss_tracker(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.ai_mode)[-max_display_cols:]
                row_display = []
                for result in tracker:
                    if result in ['W', 'L', 'S', 'T']:
                        color = '#00cc00' if result == 'W' else '#ff0000' if result == 'L' else '#cccccc'
                        row_display.append(f'<div class="pattern-circle" style="background-color: {color}; border-radius: 50%; border: 2px solid #333;"></div>')
                    else:
                        row_display.append(f'<div class="display-circle"></div>')
                st.markdown('<div id="win-loss-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>')
                if not st.session_state.history:
                    st.markdown("No results yet. Enter results below.")

        with st.expander("Prediction", expanded=True):
            bet, confidence, reason, emotional_tone, pattern_insights, pattern_scores, _ = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
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
            if pattern_scores:
                st.markdown("### Pattern Score Breakdown")
                for bet_type in ['Banker', 'Player', 'Tie']:
                    if pattern_scores[bet_type]:
                        st.markdown(f"**{bet_type}**:")
                        for pattern, score in pattern_scores[bet_type].items():
                            st.markdown(f"- {pattern}: +{score:.1f}")

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
                        line=dict(color='#00cc00', width=2),
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
                st.session_state.money_management_strategy = "Flat Betting"
                st.session_state.ai_mode = "Conservative"
                st.session_state.selected_patterns = ["Bead Bin", "Win/Loss"]
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.session_state.pattern_accuracy = {
                    'Streak': {'correct': 0, 'total': 0},
                    'BigRoad': {'correct': 0, 'total': 0},
                    'BigEye': {'correct': 0, 'total': 0},
                    'Cockroach': {'correct': 0, 'total': 0},
                    'Alternating': {'correct': 0, 'total': 0},
                    'Zigzag': {'correct': 0, 'total': 0},
                    'Trend': {'correct': 0, 'total': 0},
                    'Choppy': {'correct': 0, 'total': 0}
                }
                st.session_state.prediction_accuracy = {
                    'Banker': {'correct': 0, 'total': 0},
                    'Player': {'correct': 0, 'total': 0},
                    'Tie': {'correct': 0, 'total': 0}
                }
                st.session_state.grid_cache = {}
                st.rerun()

    except (KeyError, ValueError, IndexError) as e:
        logging.error(f"Error in main: {str(e)}")
        st.error(f"Error occurred: {str(e)}. Please try refreshing the page or resetting the game.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        st.error(f"Unexpected error: {str(e)}. Contact support if this persists.")

if __name__ == "__main__":
    main()
