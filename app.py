import math
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

def advanced_bet_selection(history: List[str], mode: str = 'Conservative') -> Tuple[str, float, str, str, List[str]]:
    """
    Determines the next bet based on Baccarat game history and betting mode.
    
    Args:
        history: List of game outcomes ('Banker', 'Player', 'Tie').
        mode: Betting strategy mode ('Conservative' or 'Aggressive').
    
    Returns:
        Tuple containing:
        - bet_choice: Recommended bet ('Banker', 'Player', 'Tie', or 'Pass').
        - confidence: Confidence score for the bet (0-100).
        - reason: Explanation of the betting decision.
        - emotional_tone: Descriptive mood of the decision.
        - pattern_insights: List of detected patterns influencing the decision.
    """
    # Configuration parameters
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
            'trend': {'base': 35, 'early_multiplier': 1.0, 'late_multiplier': 0.8},
            'big_road': {'length_3': 25, 'length_4': 35, 'length_5_plus': 45},
            'big_eye': {'repeat': 20, 'break': 15},
            'cockroach': {'repeat': 15, 'break': 12},
            'double_repeat': 30,  # New pattern weight
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
        """
        Detects a double repeat pattern (e.g., BBPP or PPBB).
        Returns True if the last min_length outcomes form pairs of identical outcomes.
        """
        if len(s) < min_length or min_length % 2 != 0:
            return False
        for i in range(0, len(s) - 1, 2):
            if i + 1 >= len(s) or s[i] != s[i + 1] or s[i] == 'Tie':
                return False
            if i + 2 < len(s) and s[i] == s[i + 2]:
                return False
        return True

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

    # Initialize variables
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
    # 1. Streak Detection
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

    # 2. Alternating Pattern (Ping Pong)
    if len(recent) >= 6 and is_alternating(recent[-6:], min_length=6):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[alternate_bet] += CONFIG['pattern_weights']['alternating']
        reason_parts.append("Strong alternating pattern (Ping Pong) in last 6 hands.")
        pattern_insights.append("Ping Pong: Alternating P/B")
        pattern_count += 1
        emotional_tone = "Excited"

    # 3. Zigzag Pattern
    if is_zigzag(recent[-8:]):
        last = recent[-1]
        zigzag_bet = 'Player' if last == 'Banker' else 'Banker'
        zigzag_score = CONFIG['pattern_weights']['zigzag']['early'] if shoe_position < 30 else CONFIG['pattern_weights']['zigzag']['late']
        scores[zigzag_bet] += zigzag_score
        reason_parts.append("Zigzag pattern (P-B-P or B-P-B) detected in last 8 hands.")
        pattern_insights.append("Zigzag: P-B-P/B-P-B")
        pattern_count += 1
        emotional_tone = "Curious"

    # 4. Double Repeat Pattern
    if len(recent) >= 4 and is_double_repeat(recent[-4:], min_length=4):
        last_two = recent[-2:]
        double_bet = 'Banker' if last_two == ['Player', 'Player'] else 'Player' if last_two == ['Banker', 'Banker'] else None
        if double_bet:
            scores[double_bet] += CONFIG['pattern_weights']['double_repeat']
            reason_parts.append("Double repeat pattern (BBPP or PPBB) detected in last 4 hands.")
            pattern_insights.append("Double Repeat: BBPP or PPBB")
            pattern_count += 1
            emotional_tone = "Intrigued"

    # 5. Recent Trend
    trend_bet, trend_score = recent_trend(recent)
    if trend_bet:
        trend_weight = trend_score * (CONFIG['pattern_weights']['trend']['early_multiplier'] if shoe_position < 20 else CONFIG['pattern_weights']['trend']['late_multiplier'])
        scores[trend_bet] += min(trend_weight, CONFIG['pattern_weights']['trend']['base'])
        reason_parts.append(f"Recent trend favors {trend_bet} in last 12 hands.")
        pattern_insights.append(f"Trend: {trend_bet} dominance")
        pattern_count += 1
        emotional_tone = "Hopeful"

    # 6. Big Road Analysis
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

    # 7. Big Eye Boy Analysis
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

    # 8. Cockroach Pig Analysis
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

    # 9. Entropy Adjustment
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

    # 10. Recent Momentum
    recent_wins = recent[-6:] if len(recent) >= 6 else recent
    for i, result in enumerate(recent_wins):
        if result in ['Banker', 'Player']:
            weight = decay_weight(i, len(recent_wins), CONFIG['half_life'])
            scores[result] += CONFIG['pattern_weights']['recent_momentum'] * weight
    reason_parts.append("Weighted recent momentum applied.")

    # 11. Frequency Analysis
    if total > 0:
        banker_ratio = freq['Banker'] / total
        player_ratio = freq['Player'] / total
        tie_ratio = freq['Tie'] / total
        scores['Banker'] += (banker_ratio * CONFIG['pattern_weights']['frequency']['banker']) * CONFIG['pattern_weights']['frequency']['tie_boost']
        scores['Player'] += (player_ratio * CONFIG['pattern_weights']['frequency']['player']) * CONFIG['pattern_weights']['frequency']['tie_boost']
        scores['Tie'] += (tie_ratio * CONFIG['pattern_weights']['frequency']['tie']) * CONFIG['pattern_weights']['frequency']['tie_boost'] if tie_ratio > CONFIG['tie_ratio_threshold'] else 0
        reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
        pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    # 12. Pattern Coherence
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

    # Final Bet Selection
    bet_choice = max(scores, key=scores.get)
    confidence = min(round(max(scores.values(), default=0) * 1.3), 95)

    # Confidence and Mode Adjustments
    confidence_threshold = CONFIG['min_confidence'][mode]
    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}% < {confidence_threshold}%). Passing.")
    elif mode == 'Conservative' and confidence < 75:
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence; proceeding cautiously.")

    # Tie Bet Adjustment
    if bet_choice == 'Tie' and (confidence < CONFIG['tie_confidence_threshold'] or freq['Tie'] / total < CONFIG['tie_ratio_threshold']):
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice] * 1.3), 95)
        reason_parts.append("Tie bet too risky; switching to safer option.")
        emotional_tone = "Cautious"

    # Late Shoe Adjustment
    if shoe_position > CONFIG['late_shoe_threshold']:
        confidence = max(confidence - CONFIG['late_shoe_penalty'], 40)
        reason_parts.append("Late in shoe; increasing caution.")
        emotional_tone = "Cautious"

    reason = " ".join(reason_parts)
    return bet_choice, confidence, reason, emotional_tone, pattern_insights
