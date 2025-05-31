import streamlit as st
import logging
import plotly.graph_objects as go
import math
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

# Set up logging
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

# Existing helper functions (unchanged)
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

# Improved advanced_bet_selection with new patterns
def advanced_bet_selection(history: List[str], mode: str = 'Conservative') -> Tuple[str, float, str, str, List[str]]:
    CONFIG = {
        'max_recent_count': 40,
        'half_life': 20,
        'min_confidence': {'Conservative': 65, 'Aggressive': 45},
        'tie_confidence_threshold': 85,
        'tie_ratio_threshold': 0.2,
        'pattern_weights': {
            'streak': {'base': 25, 'long_bonus': 10, 'per_length': 8},
            'alternating': 35,
            'zigzag': {'early': 30, 'late': 20},
            'double_repeat': 30,
            'triple_repeat': 28,
            'chop': 32,
            'mirrored_pair': 27,
            'small_road': {'repeat': 25, 'break': 20},
            'trend': {'base': 35, 'early_multiplier': 1.0, 'late_multiplier': 0.8},
            'big_road': {'length_3': 25, 'length_4': 35, 'length_5_plus': 45},
            'big_eye': {'repeat': 20, 'break': 15},
            'cockroach': {'repeat': 15, 'break': 12},
            'recent_momentum': 15,
            'frequency': {'banker': 0.9, 'player': 1.0, 'tie': 0.6, 'tie_boost': 25},
            'coherence': {'3_patterns': 15, '4_plus_patterns': 20},
        },
        'entropy_threshold': 1.5,
        'entropy_reduction': 0.7,
        'late_shoe_threshold': 60,
        'late_shoe_penalty': 10,
    }

    def detect_streak(s: List[str]) -> Tuple[Optional[str], int]:
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

    def is_alternating(s: List[str], min_length: int = 4) -> bool:
        if len(s) < min_length:
            return False
        return all(s[i] != s[i + 1] for i in range(len(s) - 1))

    def is_zigzag(s: List[str], min_length: int = 3) -> bool:
        if len(s) < min_length:
            return False
        return any(s[i] == s[i + 2] and s[i] != s[i + 1] for i in range(len(s) - 2))

    def is_double_repeat(s: List[str], min_length: int = 4) -> bool:
        if len(s) < min_length or min_length % 2 != 0:
            return False
        for i in range(0, len(s) - 1, 2):
            if i + 1 >= len(s) or s[i] != s[i + 1] or s[i] == 'Tie':
                return False
            if i + 2 < len(s) and s[i] == s[i + 2]:
                return False
        return True

    def is_triple_repeat(s: List[str], min_length: int = 4) -> bool:
        if len(s) < min_length:
            return False
        if s[-4:-1] == ['Banker', 'Banker', 'Banker'] and s[-1] == 'Player':
            return True
        if s[-4:-1] == ['Player', 'Player', 'Player'] and s[-1] == 'Banker':
            return True
        return False

    def is_chop(s: List[str], min_length: int = 4) -> bool:
        if len(s) < min_length:
            return False
        pattern = s[-min_length:]
        for i in range(len(pattern) - 1):
            if pattern[i] == pattern[i + 1] or pattern[i] == 'Tie' or pattern[i + 1] == 'Tie':
                return False
        return True

    def is_mirrored_pair(s: List[str], min_length: int = 6) -> bool:
        if len(s) < min_length:
            return False
        pattern = s[-min_length:]
        if len(pattern) >= 6:
            return (pattern[-6:-4] == ['Banker', 'Banker'] and pattern[-4:-2] == ['Player', 'Player'] and pattern[-2:] == ['Banker', 'Banker']) or \
                   (pattern[-6:-4] == ['Player', 'Player'] and pattern[-4:-2] == ['Banker', 'Banker'] and pattern[-2:] == ['Player', 'Player'])
        return False

    def recent_trend(s: List[str], window: int = 12) -> Tuple[Optional[str], float]:
        recent = s[-window:] if len(s) >= window else s
        if not recent:
            return None, 0
        freq = defaultdict(int, {'Banker': 0, 'Player': 0, 'Tie': 0})
        for r in recent:
            freq[r] += 1
        total = len(recent)
        banker_ratio = freq['Banker'] / total
        player_ratio = freq['Player'] / total
        if banker_ratio > player_ratio + 0.2:
            return 'Banker', min(banker_ratio * 50, 80)
        if player_ratio > banker_ratio + 0.2:
            return 'Player', min(player_ratio * 50, 80)
        return None, 0

    def calculate_entropy(freq: Dict[str, int], total: int) -> float:
        if total == 0:
            return 0
        return -sum((count / total) * math.log2(count / total) for count in freq.values() if count > 0)

    def decay_weight(index: int, total_length: int, half_life: float = 20) -> float:
        return 0.5 ** ((total_length - index - 1) / half_life)

    recent = history[-CONFIG['max_recent_count']:] if len(history) >= CONFIG['max_recent_count'] else history
    if not recent:
        return 'Pass', 0, "No results yet. Waiting for shoe to develop.", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    shoe_position = len(history)
    pattern_count = 0

    # Pattern Analysis
    streak_value, streak_length = detect_streak(recent)
    if streak_length >= 3 and streak_value != "Tie":
        streak_score = min(CONFIG['pattern_weights']['streak']['base'] + (streak_length - 3) * CONFIG['pattern_weights']['streak']['per_length'], 50)
        if streak_length >= 6:
            streak_score += CONFIG['pattern_weights']['streak']['long_bonus']
            pattern_insights.append(f"Dragon Tail: {streak_length} {streak_value}")
            emotional_tone = "Confident"
        scores[streak_value] += streak_score
        reason_parts.append(f"Streak of {streak_length} {streak_value} wins detected.")
        pattern_insights.append(f"Streak: {streak_length} {streak_value}")
        pattern_count += 1
        if streak_length >= 5 and mode == 'Aggressive':
            contrarian_bet = 'Player' if streak_value == 'Banker' else 'Banker'
            scores[contrarian_bet] += 20
            reason_parts.append(f"Long streak ({streak_length}); considering break in Aggressive mode.")
            pattern_insights.append("Possible streak break")
            emotional_tone = "Skeptical"

    if len(recent) >= 6 and is_alternating(recent[-6:], min_length=6):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[alternate_bet] += CONFIG['pattern_weights']['alternating']
        reason_parts.append("Strong alternating pattern (Ping Pong) in last 6 hands.")
        pattern_insights.append("Ping Pong: Alternating P/B")
        pattern_count += 1
        emotional_tone = "Excited"

    if is_zigzag(recent[-8:]):
        last = recent[-1]
        zigzag_bet = 'Player' if last == 'Banker' else 'Banker'
        zigzag_score = CONFIG['pattern_weights']['zigzag']['early'] if shoe_position < 30 else CONFIG['pattern_weights']['zigzag']['late']
        scores[zigzag_bet] += zigzag_score
        reason_parts.append("Zigzag pattern (P-B-P or B-P-B) detected in last 8 hands.")
        pattern_insights.append("Zigzag: P-B-P/B-P-B")
        pattern_count += 1
        emotional_tone = "Curious"

    if len(recent) >= 4 and is_double_repeat(recent[-4:], min_length=4):
        last_two = recent[-2:]
        double_bet = 'Banker' if last_two == ['Player', 'Player'] else 'Player' if last_two == ['Banker', 'Banker'] else None
        if double_bet:
            scores[double_bet] += CONFIG['pattern_weights']['double_repeat']
            reason_parts.append("Double repeat pattern (BBPP or PPBB) detected in last 4 hands.")
            pattern_insights.append("Double Repeat: BBPP or PPBB")
            pattern_count += 1
            emotional_tone = "Intrigued"

    if len(recent) >= 4 and is_triple_repeat(recent[-4:], min_length=4):
        last = recent[-1]
        triple_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[triple_bet] += CONFIG['pattern_weights']['triple_repeat']
        reason_parts.append("Triple repeat pattern (BBBP or PPPB) detected in last 4 hands.")
        pattern_insights.append("Triple Repeat: BBBP or PPPB")
        pattern_count += 1
        emotional_tone = "Intrigued"

    if len(recent) >= 4 and is_chop(recent[-4:], min_length=4):
        last = recent[-1]
        chop_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[chop_bet] += CONFIG['pattern_weights']['chop']
        reason_parts.append("Chop pattern (single alternation) detected in last 4 hands.")
        pattern_insights.append("Chop: BPBP or PBPB")
        pattern_count += 1
        emotional_tone = "Alert"

    if len(recent) >= 6 and is_mirrored_pair(recent[-6:], min_length=6):
        last_two = recent[-2:]
        mirror_bet = 'Banker' if last_two == ['Banker', 'Banker'] else 'Player'
        scores[mirror_bet] += CONFIG['pattern_weights']['mirrored_pair']
        reason_parts.append("Mirrored pair pattern (BBPPBB or PPBBPP) detected in last 6 hands.")
        pattern_insights.append("Mirrored Pair: BBPPBB or PPBBPP")
        pattern_count += 1
        emotional_tone = "Fascinated"

    trend_bet, trend_score = recent_trend(recent)
    if trend_bet:
        trend_weight = trend_score * (CONFIG['pattern_weights']['trend']['early_multiplier'] if shoe_position < 20 else CONFIG['pattern_weights']['trend']['late_multiplier'])
        scores[trend_bet] += min(trend_weight, CONFIG['pattern_weights']['trend']['base'])
        reason_parts.append(f"Recent trend favors {trend_bet} in last 12 hands.")
        pattern_insights.append(f"Trend: {trend_bet} dominance")
        pattern_count += 1
        emotional_tone = "Hopeful"

    big_road_grid, num_cols = build_big_road(recent)
    if num_cols > 0:
        last_col = [big_road_grid[row][num_cols - 1] for row in range(6)]
        col_length = sum(1 for x in last_col if x in ['P', 'B'])
        if col_length >= 3:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            col_score = (
                CONFIG['pattern_weights']['big_road']['length_3'] if col_length == 3 else
                CONFIG['pattern_weights']['big_road']['length_4'] if col_length == 4 else
                CONFIG['pattern_weights']['big_road']['length_5_plus']
            )
            scores[bet_side] += col_score
            reason_parts.append(f"Big Road column of {col_length} {bet_side}.")
            pattern_insights.append(f"Big Road: {col_length} {bet_side}")
            pattern_count += 1

    big_eye_grid, big_eye_cols = build_big_eye_boy(big_road_grid, num_cols)
    if big_eye_cols > 1:
        last_two_cols = [[big_eye_grid[row][c] for row in range(6)] for c in range(big_eye_cols - 2, big_eye_cols)]
        last_signals = [next((x for x in col if x in ['R', 'B']), None) for col in last_two_cols]
        if all(s == 'R' for s in last_signals if s):
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            scores[last_side] += CONFIG['pattern_weights']['big_eye']['repeat']
            reason_parts.append("Big Eye Boy shows consistent repeat pattern.")
            pattern_insights.append("Big Eye Boy: Consistent repeat")
            pattern_count += 1
        elif all(s == 'B' for s in last_signals if s):
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            scores[opposite_side] += CONFIG['pattern_weights']['big_eye']['break']
            reason_parts.append("Big Eye Boy shows consistent break pattern.")
            pattern_insights.append("Big Eye Boy: Consistent break")
            pattern_count += 1

    small_road_grid, small_road_cols = build_small_road(big_road_grid, num_cols)
    if small_road_cols > 1:
        last_two_cols = [[small_road_grid[row][c] for row in range(6)] for c in range(small_road_cols - 2, small_road_cols)]
        last_signals = [next((x for x in col if x in ['R', 'B']), None) for col in last_two_cols]
        if all(s == 'R' for s in last_signals if s):
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            scores[last_side] += CONFIG['pattern_weights']['small_road']['repeat']
            reason_parts.append("Small Road shows consistent repeat pattern.")
            pattern_insights.append("Small Road: Consistent repeat")
            pattern_count += 1
        elif all(s == 'B' for s in last_signals if s):
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            scores[opposite_side] += CONFIG['pattern_weights']['small_road']['break']
            reason_parts.append("Small Road shows consistent break pattern.")
            pattern_insights.append("Small Road: Consistent break")
            pattern_count += 1

    cockroach_grid, cockroach_cols = build_cockroach_pig(big_road_grid, num_cols)
    if cockroach_cols > 1:
        last_two_cols = [[cockroach_grid[row][c] for row in range(6)] for c in range(cockroach_cols - 2, cockroach_cols)]
        last_signals = [next((x for x in col if x in ['R', 'B']), None) for col in last_two_cols]
        if all(s == 'R' for s in last_signals if s):
            last_side = 'Player' if big_road_grid[0][num_cols - 1] == 'P' else 'Banker'
            scores[last_side] += CONFIG['pattern_weights']['cockroach']['repeat']
            reason_parts.append("Cockroach Pig shows consistent repeat pattern.")
            pattern_insights.append("Cockroach Pig: Consistent repeat")
            pattern_count += 1
        elif all(s == 'B' for s in last_signals if s):
            opposite_side = 'Player' if big_road_grid[0][num_cols - 1] == 'B' else 'Banker'
            scores[opposite_side] += CONFIG['pattern_weights']['cockroach']['break']
            reason_parts.append("Cockroach Pig shows consistent break pattern.")
            pattern_insights.append("Cockroach Pig: Consistent break")
            pattern_count += 1

    freq = defaultdict(int, {'Banker': 0, 'Player': 0, 'Tie': 0})
    for r in recent:
        freq[r] += 1
    total = len(recent)
    entropy = calculate_entropy(freq, total)
    if entropy > CONFIG['entropy_threshold']:
        for key in scores:
            scores[key] *= CONFIG['entropy_reduction']
        reason_parts.append("High randomness detected; lowering pattern confidence.")
        pattern_insights.append("Randomness: High entropy")
        emotional_tone = "Cautious"

    recent_wins = recent[-6:] if len(recent) >= 6 else recent
    for i, result in enumerate(recent_wins):
        if result in ['Banker', 'Player']:
            weight = decay_weight(i, len(recent_wins), CONFIG['half_life'])
            scores[result] += CONFIG['pattern_weights']['recent_momentum'] * weight
    reason_parts.append("Weighted recent momentum applied.")

    if total > 0:
        banker_ratio = freq['Banker'] / total
        player_ratio = freq['Player'] / total
        tie_ratio = freq['Tie'] / total
        scores['Banker'] += (banker_ratio * CONFIG['pattern_weights']['frequency']['banker']) * CONFIG['pattern_weights']['frequency']['tie_boost']
        scores['Player'] += (player_ratio * CONFIG['pattern_weights']['frequency']['player']) * CONFIG['pattern_weights']['frequency']['tie_boost']
        scores['Tie'] += (tie_ratio * CONFIG['pattern_weights']['frequency']['tie']) * CONFIG['pattern_weights']['frequency']['tie_boost'] if tie_ratio > CONFIG['tie_ratio_threshold'] else 0
        reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
        pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    if pattern_count >= 3:
        max_score = max(scores['Banker'], scores['Player'])
        if max_score > 0:
            coherence_bonus = CONFIG['pattern_weights']['coherence']['3_patterns'] if pattern_count == 3 else CONFIG['pattern_weights']['coherence']['4_plus_patterns']
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

# Money management and other functions (unchanged)
def money_management(bankroll, base_bet, strategy, bet_outcome=None):
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if bankroll < min_bet:
        logging.warning(f"Bankroll ({bankroll:.2f}) is less than minimum bet ({min_bet:.2f}).")
        return 0.0

    if strategy == "T3":
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
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, _ = advanced_bet_selection(current_rounds[:-1], st.session_state.ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', '')
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie'):
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
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
                money_management(current_bankroll, base_bet, strategy, bet_outcome='win')
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
            if strategy == "T3":
                money_management(current_bankroll, base_bet, strategy, bet_outcome='loss')
        bankroll_progress.append(current_bankroll)
    return bankroll_history, bet_sizes

def calculate_win_loss_tracker(history: List[str], base_bet: float, strategy: str, ai_mode: str) -> List[str]:
    tracker = []
    st.session_state.t3_level = 1
    st.session_state.t3_results = []
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
            if strategy == "T3":
                money_management(st.session_state.initial_bankroll, base_bet, strategy, bet_outcome='win')
        else:
            tracker.append('L')
            if strategy == "T3":
                money_management(st.session_state.initial_bankroll, base_bet, strategy, bet_outcome='loss')
    return tracker

def main():
    try:
        st.set_page_config(page_title="Mang Baccarat Predictor", page_icon="🎲", layout="wide")
        st.title("Mang Baccarat Predictor")

        # Initialize session state variables
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

        # Improved CSS for responsiveness
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
            .pattern-scroll::-webkit-scrollbar {
                background-color: #888;
                border-radius: 4px;
            }
            .stButton > button {
                width: 100%;
                padding: 10px;
                margin: 5px 0;
                font-size: 1rem;
            }
            .stNumberInput, .stSelectbox {
                width: 100%;
                max-width: 200px;
            }
            .stExpander {
                margin-bottom: 15px;
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
                width: 24px;
                height: 24px;
                display: inline-block;
                margin: 3px;
                border-radius: 50%;
                border: 1px solid #ffffff;
            }
            .display-circle {
                width: 24px;
                height: 24px;
                display: inline-block;
                margin: 3px;
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
                    width: 18px;
                    height: 18px;
                    margin: 2px;
                }
                .stButton > button {
                    font-size: 0.9rem;
                    padding: 8px;
                }
                .stNumberInput input, .stSelectbox div {
                    font-size: 0.9rem;
                }
            }
            </style>
            <script>
            function autoScrollPatterns() {
                const containers = [
                    'bead-bin-scroll', 'big-road-scroll', 'big-eye-scroll',
                    'cockroach-scroll', 'small-road-scroll', 'win-loss-scroll',
                    'double-repeat-scroll', 'triple-repeat-scroll', 'chop-scroll',
                    'mirrored-scroll'
                ];
                containers.forEach(container => {
                    const element = document.getElementById(container);
                    if (element) {
                        element.scrollLeft = element.scrollWidth;
                    }
                });
            }
            window.onload = autoScrollPatterns;
            window.onresize = autoScrollPatterns;
            </script>
        """, unsafe_allow_html=True)

        # Game Settings
        with st.expander("Game Settings", expanded=False):
            cols = st.columns([1, 1, 1, 1], gap="small")
            with cols[0]:
                initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
            with cols[1]:
                base_bet = st.number_input("Base Bet", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
            with cols[2]:
                strategy_options = ["Flat Betting", "T3"]
                money_management_strategy = st.selectbox("Money Management", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
                st.markdown("*Flat: Fixed bets. T3: Adjusts based on last three outcomes.*")
            with cols[3]:
                ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

            st.session_state.initial_bankroll = initial_bankroll
            st.session_state.base_bet = base_bet
            st.session_state.money_management_strategy = money_management_strategy
            st.session_state.ai_mode = ai_mode

        # Input Game Results
        with st.expander("Input Game Results", expanded=True):
            cols = st.columns([1, 1, 1, 1], gap="small")
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
                if st.button("Undo", disabled=len(st.session_state.history) == 0):
                    st.session_state.history.pop()
                    if st.session_state.money_management_strategy == "T3":
                        st.session_state.t3_results = []
                        st.session_state.t3_level = 1
                    st.rerun()

        # Shoe Patterns
        with st.expander("Shoe Patterns", expanded=False):
            pattern_options = ["Bead Bin", "Big Road", "Big Eye", "Small Road", "Cockroach", "Double Repeat", "Triple Repeat", "Chop", "Mirrored Pair", "Win/Loss"]
            selected_patterns = st.multiselect("Select Patterns", pattern_options, default=st.session_state.selected_patterns)
            st.session_state.selected_patterns = selected_patterns

            max_display_cols = 10 if st.session_state.screen_width < 768 else 8

            if "Bead Bin" in selected_patterns:
                st.markdown("### Bead Bin")
                sequence = st.session_state.history[-48:]
                sequence = ['P' if r == 'Player' else 'B' if r == 'Banker' else 'T' for r in sequence]
                grid = [['' for _ in range(max_display_cols)] for _ in range(6)]
                for i, result in enumerate(sequence):
                    col = i // 6
                    row = i % 6
                    if col < max_display_cols:
                        color = '#ff0000' if result == 'P' else '#ff0000' if result == 'B' else '#00cc00' if
                        grid[row][col] = f'<div class="pattern-circle" style="background-color: {color};">></div>'
                    st.markdown('<div id="bead-bin-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in grid:
                        st.markdown(''.join(row), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Big Road" in selected_patterns:
                st.markdown("### Big Road Marker")
                big_road_grid, num_cols = build_big_road(st.session_state.history)
                if num_cols > 0:
                    display_cols = min(num_cols, max_display_cols)
                    st.markdown('<div id="big-road-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = big_road_grid[row][col]
                            if outcome == 'P':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #ff0000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #ff0000;"></div>')
                            elif outcome == 'T':
                                row_display.append(f'<div class="pattern-circle" style="border: 2px solid #00cc00;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Big Eye" in selected_patterns:
                st.markdown("### Big Eye Boy")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🔴: Repeat, 🔵: Break</p>", unsafe_allow_html=True)
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
                                row_display.append(f'<div class="pattern-circle" style="background-color: #ff0000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #0000ff;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Small Road" in selected_patterns:
                st.markdown("### Small Road")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🔴: Repeat, 🔵: Break</p>", unsafe_allow_html=True)
                big_road_grid, num_cols = build_big_road(st.session_state.history)
                small_road_grid, small_road_cols = build_small_road(big_road_grid, num_cols)
                if small_road_cols > 0:
                    display_cols = min(small_road_cols, max_display_cols)
                    st.markdown('<div id="small-road-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                    for row in range(6):
                        row_display = []
                        for col in range(display_cols):
                            outcome = small_road_grid[row][col]
                            if outcome == 'R':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #ff0000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #0000ff;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Cockroach" in selected_patterns:
                st.markdown("### Cockroach Pig")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🔴: Repeat, 🔵: Break</p>", unsafe_allow_html=True)
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
                                row_display.append(f'<div class="pattern-circle" style="background-color: #ff0000;"></div>')
                            elif outcome == 'B':
                                row_display.append(f'<div class="pattern-circle" style="background-color: #0000ff;"></div>')
                            else:
                                row_display.append(f'<div class="display-circle"></div>')
                        st.markdown(''.join(row_display), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if "Double Repeat" in selected_patterns:
                st.markdown("### Double Repeat")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🟢: BBPP or PPBB pattern</p>", unsafe_allow_html=True)
                sequence = st.session_state.history[-12:]
                row_display = []
                for i in range(0, len(sequence) - 3, 2):
                    chunk = sequence[i:i+4]
                    if len(chunk) == 4 and chunk[0] == chunk[1] != 'Tie' and chunk[2] == chunk[3] != 'Tie' and chunk[0] != chunk[2]:
                        row_display.append(f'<div class="pattern-circle" style="background-color: #00cc00;"></div>')
                    else:
                        row_display.append(f'<div class="display-circle"></div>')
                st.markdown('<div id="double-repeat-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if "Triple Repeat" in selected_patterns:
                st.markdown("### Triple Repeat")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🟢: BBBP or PPPB pattern</p>", unsafe_allow_html=True)
                sequence = st.session_state.history[-12:]
                row_display = []
                for i in range(len(sequence) - 3):
                    chunk = sequence[i:i+4]
                    if len(chunk) == 4 and chunk[:3] == ['Banker', 'Banker', 'Banker'] and chunk[3] == 'Player' or \
                       chunk[:3] == ['Player', 'Player', 'Player'] and chunk[3] == 'Banker':
                        row_display.append(f'<div class="pattern-circle" style="background-color: #00cc00;"></div>')
                    else:
                        row_display.append(f'<div class="display-circle"></div>')
                st.markdown('<div id="triple-repeat-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if "Chop" in selected_patterns:
                st.markdown("### Chop")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🟢: BPBP or PBPB alternation</p>", unsafe_allow_html=True)
                sequence = st.session_state.history[-12:]
                row_display = []
                for i in range(len(sequence) - 3):
                    chunk = sequence[i:i+4]
                    if len(chunk) == 4 and all(chunk[j] != chunk[j+1] and chunk[j] != 'Tie' for j in range(len(chunk)-1):
                        row_display.append(f'<div class="pattern-circle" style="background-color: #00cc00;"></div>')
                        else:
                            row_display.append(f'<div class="display-circle"></div>')
                st.markdown('<div id="chop-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if "Mirrored Pair" in selected_patterns:
                st.markdown("### Mirrored Pair")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🟢: BBPPBB or PPBBPP pattern</p>", unsafe_allow_html=True)
                sequence = st.session_state.history[-12:]
                row_display = []
                for i in range(len(sequence) - 5):
                    chunk = sequence[i:i+6]
                    if len(chunk) == 6 and (
                        (chunk[0] == chunk[1] == 'Banker' and chunk[2] == chunk[3] == 'Player' and chunk[4] == chunk'[5] == 'Banker') or \
                        (chunk[0] == chunk[1] == 'Player' and chunk[2] == chunk'[3] == 'Banker' and chunk[4] == chunk'[5] == 'Player')
                    ) and all(c != 'Tie' for c in chunk):
                        row_display.append(f'<div class="pattern-circle" style="background-color: #00cc00;"></div>')
                    else:
                        row_display.append(f'<div class="display-circle"></div>')
                st.markdown('<div id="mirrored-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if "Win/Loss" in selected_patterns:
                st.markdown("### Win/Loss Statistics")
                st.markdown("<p style='font-size: 0.9rem; color: #666;'>🟢: Win, 🔴: Loss, ⬜: Skip/Tie</p>", unsafe_allow_html=True)
                tracker = calculate_win_loss_tracker(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy, st.session_state.ai_mode)[-max_display_cols:]
                row_display = []
                for result in tracker:
                    color = '#00cc00' if result == 'W' else '#ff0000' if result == 'L' else '#A0AEC0'
                    row_display.append(f'<div class="pattern-circle" style="background-color: {color};"></div>')
                st.markdown('<div id="win-loss-scroll" class="pattern-scroll">', unsafe_allow_html=True)
                st.markdown(''.join(row_display), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Prediction
        with st.expander("Prediction", expanded=True):
            bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
            current_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
            recommended_bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy)
            st.markdown("### Advanced Prediction")
            if current_bankroll < max(1.0, st.session_state.base_bet):
                st.warning("Insufficient bankroll ({current_bankroll:.2f}) to place a bet.")
                bet = 'Pass'
                confidence = 0
                reason = "Bankroll too low."
                emotional_tone = "Cautious"
            if bet == 'Pass':
                st.markdown("**No Bet**: Insufficient confidence or bankroll.")
            else:
                st.markdown(f"**Bet**: {bet} | **Confidence**: {confidence}% | **Bet Size**: ${recommended_bet_size:.2f} | **Mood**: {emotional_tone}")
            st.markdown(f"**Reasoning**: {reason}")
            if pattern_insights:
                st.markdown("### Pattern Insights")
                for insight in pattern_insights:
                    st.markdown(f"- {insight}")

        # Bankroll Progress
        with st.expander("Bankroll History", expanded=True):
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
                st.markdown("### Bankroll Chart")
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
            st.markdown(f"**Current Bankroll**: ${current_bankroll:.2f}")

        # Reset
        with st.expander("Reset", expanded=False):
            if st.button("New Game"):
                final_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else st.session_state.initial_bankroll
                st.session_state.history = []
                st.session_state.initial_bankroll = max(1.0, final_bankroll)
                st.session_state.base_bet = min(10, st.session_state.initial_bankroll)
                st.session_state.money_management_strategy = "Flat Betting"
                st.session_state.ai_mode = "Conservative"
                st.session_state.selected_patterns = ["Bead Bin", "Win/Loss"]
                st.session_state.t3_level = 1
                st.session_state.t3_results = []
                st.rerun()

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        st.error(f"Error: {str(e)}. Try refreshing or resetting the game.")

if __name__ == "__main__":
    main()
