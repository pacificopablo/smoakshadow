import streamlit as st

# Normalize input tokens
def normalize_result(s):
    s = s.strip().lower()
    if s == 'banker' or s == 'b':
        return 'Banker'
    if s == 'player' or s == 'p':
        return 'Player'
    if s == 'tie' or s == 't':
        return 'Tie'
    return None

def detect_streak(results):
    if not results:
        return None, 0
    last = results[-1]
    count = 1
    for i in range(len(results) - 2, -1, -1):
        if results[i] == last:
            count += 1
        else:
            break
    return last, count

def is_alternating_pattern(arr):
    if len(arr) < 4:
        return False
    for i in range(len(arr) - 1):
        if arr[i] == arr[i + 1]:
            return False
    return True

def recent_trend_analysis(results, window=10):
    recent = results[-window:] if len(results) >= window else results
    if not recent:
        return None, 0
    freq = frequency_count(recent)
    total = len(recent)
    if total == 0:
        return None, 0
    banker_ratio = freq['Banker'] / total
    player_ratio = freq['Player'] / total
    if banker_ratio > player_ratio + 0.2:
        return 'Banker', banker_ratio * 50
    elif player_ratio > banker_ratio + 0.2:
        return 'Player', player_ratio * 50
    return None, 0

def frequency_count(arr):
    count = {'Banker': 0, 'Player': 0, 'Tie': 0}
    for r in arr:
        if r in count:
            count[r] += 1
    return count

def build_big_road(results):
    big_road = []
    current_column = []
    last_outcome = None
    for result in results:
        mapped = 'P' if result == 'Player' else 'B' if result == 'Banker' else 'T'
        if mapped == 'T':
            if current_column:
                current_column.append('T')
            continue
        if not current_column or (mapped == last_outcome and len(current_column) < 6):
            current_column.append(mapped)
        else:
            big_road.append(current_column)
            current_column = [mapped]
        last_outcome = mapped
    if current_column:
        big_road.append(current_column)
    return big_road

def analyze_big_eye_boy(big_road):
    if len(big_road) < 3:
        return None
    last_col = big_road[-1]
    second_last = big_road[-2]
    third_last = big_road[-3]
    if len(last_col) == len(second_last) and last_col[0] == second_last[0]:
        return 'Repeat'
    elif len(second_last) == len(third_last) and second_last[0] == third_last[0]:
        return 'Repeat'
    return 'Break'

def advanced_bet_selection(results, mode='Conservative'):
    max_recent_count = 30
    recent = results[-max_recent_count:]
    if not recent:
        return 'Pass', 0, "No results yet. Let’s wait for the shoe to develop!", "Cautious", []

    scores = {'Banker': 0, 'Player': 0, 'Tie': 0}
    reason_parts = []
    pattern_insights = []
    emotional_tone = "Neutral"
    confidence = 0

    streak_value, streak_length = detect_streak(recent)
    if streak_length >= 3 and streak_value != "Tie":
        streak_score = min(75 + (streak_length - 3) * 10, 90)
        scores[streak_value] += streak_score
        reason_parts.append(f"Streak of {streak_length} {streak_value} wins detected.")
        pattern_insights.append(f"Streak: {streak_length} {streak_value}")
        emotional_tone = "Optimistic" if streak_length < 5 else "Confident"
        if streak_length >= 5 and mode == 'Aggressive':
            contrarian_bet = 'Player' if streak_value == 'Banker' else 'Banker'
            scores[contrarian_bet] += 30
            reason_parts.append(f"Long streak ({streak_length}); considering a break.")
            pattern_insights.append("Possible streak break")
            emotional_tone = "Skeptical"

    elif len(recent) >= 4 and is_alternating_pattern(recent[-4:]):
        last = recent[-1]
        alternate_bet = 'Player' if last == 'Banker' else 'Banker'
        scores[alternate_bet] += 70
        reason_parts.append("Alternating pattern (chop) in last 4 hands.")
        pattern_insights.append("Chop pattern: Alternating P/B")
        emotional_tone = "Excited"

    trend_bet, trend_score = recent_trend_analysis(recent)
    if trend_bet:
        scores[trend_bet] += trend_score
        reason_parts.append(f"Recent trend favors {trend_bet} in last 10 hands.")
        pattern_insights.append(f"Trend: {trend_bet} dominance")
        emotional_tone = "Hopeful"

    big_road = build_big_road(recent)
    big_eye_signal = analyze_big_eye_boy(big_road)
    if big_road:
        last_col = big_road[-1]
        if len(last_col) >= 3 and last_col[0] in ['P', 'B']:
            bet_side = 'Player' if last_col[0] == 'P' else 'Banker'
            scores[bet_side] += 60
            reason_parts.append(f"Big Road shows a column of {len(last_col)} {bet_side}.")
            pattern_insights.append(f"Big Road: {len(last_col)} {bet_side}")
        if big_eye_signal == 'Repeat' and len(big_road) >= 2:
            last_side = 'Player' if big_road[-1][0] == 'P' else 'Banker'
            scores[last_side] += 50
            reason_parts.append("Big Eye Boy suggests pattern repetition.")
            pattern_insights.append("Big Eye Boy: Repeat pattern")
        elif big_eye_signal == 'Break':
            opposite_side = 'Player' if big_road[-1][0] == 'B' else 'Banker'
            scores[opposite_side] += 40
            reason_parts.append("Big Eye Boy indicates a pattern break.")
            pattern_insights.append("Big Eye Boy: Break pattern")

    freq = frequency_count(recent)
    total = len(recent)
    scores['Banker'] += (freq['Banker'] / total * 0.9) * 50
    scores['Player'] += (freq['Player'] / total * 1.0) * 50
    scores['Tie'] += (freq['Tie'] / total * 0.5) * 50
    reason_parts.append(f"Long-term: Banker {freq['Banker']}, Player {freq['Player']}, Tie {freq['Tie']}.")
    pattern_insights.append(f"Frequency: B:{freq['Banker']}, P:{freq['Player']}, T:{freq['Tie']}")

    bet_choice = max(scores, key=scores.get)
    confidence = min(round(max(scores['Banker'], scores['Player'], scores['Tie'])), 95)

    confidence_threshold = 60 if mode == 'Conservative' else 40
    if confidence < confidence_threshold:
        bet_choice = 'Pass'
        emotional_tone = "Hesitant"
        reason_parts.append(f"Confidence too low ({confidence}% < {confidence_threshold}%). Passing.")
    elif confidence < 70 and mode == 'Conservative':
        emotional_tone = "Cautious"
        reason_parts.append("Moderate confidence; proceeding cautiously.")

    if bet_choice == 'Tie' and confidence < 80:
        scores['Tie'] = 0
        bet_choice = max(scores, key=scores.get)
        confidence = min(round(scores[bet_choice]), 95)
        reason_parts.append("Tie bet too risky; switching to safer option.")
        emotional_tone = "Cautious"

    if len(pattern_insights) > 2 and max(scores.values()) - min(scores.values()) < 20:
        confidence = max(confidence - 15, 40)
        reason_parts.append("Multiple conflicting patterns; lowering confidence.")
        emotional_tone = "Skeptical"

    reason = " ".join(reason_parts)
    return bet_choice, confidence, reason, emotional_tone, pattern_insights

def money_management(bankroll, base_bet, strategy, confidence=None, history=None):
    min_bet = max(1.0, base_bet)
    max_bet = bankroll

    if strategy == "T3":
        if not history or len(history) < 3:
            calculated_bet = base_bet
        else:
            mapped_history = ['P' if r == 'Player' else 'B' if r == 'Banker' else 'T' for r in history]
            recent = mapped_history[-3:]
            last_result = recent[-1]
            streak = all(r == last_result for r in recent)
            if streak and last_result in ['P', 'B']:
                if len(mapped_history) >= 4 and mapped_history[-4] == last_result:
                    calculated_bet = base_bet * 4
                else:
                    calculated_bet = base_bet * 2
            else:
                calculated_bet = base_bet
    elif strategy == "Fixed 5% of Bankroll":
        calculated_bet = bankroll * 0.05
    elif strategy == "Flat Betting":
        calculated_bet = base_bet
    elif strategy == "Confidence-Based":
        if confidence is None:
            confidence = 50
        confidence_factor = confidence / 100.0
        bet_percentage = 0.02 + (confidence_factor * 0.03)
        calculated_bet = bankroll * bet_percentage
    else:
        calculated_bet = base_bet

    bet_size = round(calculated_bet / base_bet) * base_bet
    bet_size = max(min_bet, min(bet_size, max_bet))
    return round(bet_size, 2)

def calculate_bankroll(history, base_bet, strategy):
    bankroll = st.session_state.initial_bankroll if 'initial_bankroll' in st.session_state else 1000.0
    current_bankroll = bankroll
    bankroll_progress = []
    bet_sizes = []
    for i in range(len(history)):
        current_rounds = history[:i + 1]
        bet, confidence, _, _, _ = advanced_bet_selection(current_rounds[:-1], st.session_state.ai_mode) if i != 0 else ('Pass', 0, '', 'Neutral', [])
        actual_result = history[i]
        if bet in (None, 'Pass', 'Tie'):
            bankroll_progress.append(current_bankroll)
            bet_sizes.append(0.0)
            continue
        bet_size = money_management(current_bankroll, base_bet, strategy, confidence, current_rounds)
        bet_sizes.append(bet_size)
        if actual_result == bet:
            if bet == 'Banker':
                win_amount = bet_size * 0.95
                current_bankroll += win_amount
            else:
                current_bankroll += bet_size
        elif actual_result == 'Tie':
            bankroll_progress.append(current_bankroll)
            continue
        else:
            current_bankroll -= bet_size
        bankroll_progress.append(current_bankroll)
    return bankroll_progress, bet_sizes

def main():
    st.set_page_config(page_title="Smart Baccarat Predictor with Emotions", page_icon="🎲", layout="centered")
    st.title("Smart Baccarat Predictor with Emotions")

    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.initial_bankroll = 1000.0
        st.session_state.base_bet = 10.0
        st.session_state.money_management_strategy = "Fixed 5% of Bankroll"
        st.session_state.ai_mode = "Conservative"

    # Game Settings
    with st.expander("Game Settings", expanded=False):
        col_init, col_base, col_strategy, col_mode = st.columns(4)
        with col_init:
            initial_bankroll = st.number_input("Initial Bankroll", min_value=1.0, value=st.session_state.initial_bankroll, step=10.0, format="%.2f")
        with col_base:
            base_bet = st.number_input("Base Bet (Unit Size)", min_value=1.0, max_value=initial_bankroll, value=st.session_state.base_bet, step=1.0, format="%.2f")
        with col_strategy:
            strategy_options = ["Fixed 5% of Bankroll", "Flat Betting", "Confidence-Based", "T3"]
            money_management_strategy = st.selectbox("Money Management Strategy", strategy_options, index=strategy_options.index(st.session_state.money_management_strategy))
        with col_mode:
            ai_mode = st.selectbox("AI Mode", ["Conservative", "Aggressive"], index=["Conservative", "Aggressive"].index(st.session_state.ai_mode))

        st.session_state.initial_bankroll = initial_bankroll
        st.session_state.base_bet = base_bet
        st.session_state.money_management_strategy = money_management_strategy
        st.session_state.ai_mode = ai_mode

        st.markdown(f"**Selected Money Management Strategy:** {money_management_strategy}")

    # Game Input Buttons
    with st.expander("Input Game Results", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Banker"):
                st.session_state.history.append("Banker")
                st.rerun()
        with col2:
            if st.button("Player"):
                st.session_state.history.append("Player")
                st.rerun()
        with col3:
            if st.button("Tie"):
                st.session_state.history.append("Tie")
                st.rerun()
        with col4:
            if st.button("Undo", disabled=len(st.session_state.history) == 0):
                if st.session_state.history:
                    st.session_state.history.pop()
                    st.rerun()
                else:
                    st.warning("Nothing to undo!")

    # Shoe Patterns
    with st.expander("Shoe Patterns", expanded=False):
        st.markdown("### Bead Plate")
        sequence = [r for r in st.session_state.history][-84:]
        sequence = ['P' if r == 'Player' else 'B' if r == 'Banker' else 'T' for r in sequence]
        grid = [['' for _ in range(14)] for _ in range(6)]
        for i, result in enumerate(sequence):
            if result in ['P', 'B', 'T']:
                col = i // 6
                row = i % 6
                if col < 14:
                    color = '#3182ce' if result == 'P' else '#e53e3e' if result == 'B' else '#38a169'
                    grid[row][col] = f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>'
        for row in grid:
            st.markdown(' '.join(row), unsafe_allow_html=True)
        if not st.session_state.history:
            st.write("_No results yet. Click the buttons above to add results._")

        st.markdown("### Big Road")
        big_road = build_big_road(st.session_state.history)
        if big_road:
            for col in big_road[:14]:
                col_display = []
                for outcome in col[:6]:
                    if outcome == 'P':
                        col_display.append('<div style="width: 20px; height: 20px; background-color: #3182ce; border-radius: 50%; display: inline-block;"></div>')
                    elif outcome == 'B':
                        col_display.append('<div style="width: 20px; height: 20px; background-color: #e53e3e; border-radius: 50%; display: inline-block;"></div>')
                    elif outcome == 'T':
                        col_display.append('<div style="width: 20px; height: 20px; border: 2px solid #38a169; border-radius: 50%; display: inline-block;"></div>')
                st.markdown(' '.join(col_display), unsafe_allow_html=True)
        else:
            st.write("_No Big Road data yet._")

    # Bet Prediction
    with st.expander("Prediction for Next Bet", expanded=True):
        bet, confidence, reason, emotional_tone, pattern_insights = advanced_bet_selection(st.session_state.history, st.session_state.ai_mode)
        st.markdown("### Prediction for Next Bet")
        if bet == 'Pass':
            st.warning("I’m not betting this time! The pattern is too unclear.")
            st.info(reason)
            recommended_bet_size = 0.0
        else:
            current_bankroll = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)[0][-1] if st.session_state.history else initial_bankroll
            recommended_bet_size = money_management(current_bankroll, st.session_state.base_bet, st.session_state.money_management_strategy, confidence, st.session_state.history)
            st.success(f"Predicted Bet: **{bet}**    Confidence: **{confidence}%**    Recommended Bet Size: **${recommended_bet_size:.2f}**    Emotion: **{emotional_tone}**")
            st.write(reason)
        if pattern_insights:
            st.markdown("#### Detected Patterns")
            for insight in pattern_insights:
                st.write(f"- {insight}")

    # Bankroll Progression
    with st.expander("Bankroll and Bet Size Progression", expanded=False):
        bankroll_progress, bet_sizes = calculate_bankroll(st.session_state.history, st.session_state.base_bet, st.session_state.money_management_strategy)
        if bankroll_progress:
            st.markdown("### Bankroll and Bet Size Progression (Newest to Oldest)")
            total_hands = len(bankroll_progress)
            for i, (val, bet_size) in enumerate(zip(reversed(bankroll_progress), reversed(bet_sizes))):
                hand_number = total_hands - i
                bet_display = f"Bet: ${bet_size:.2f}" if bet_size > 0 else "Bet: None (No prediction, Tie, or Pass)"
                st.write(f"Hand {hand_number}: Bankroll ${val:.2f}, {bet_display}")
            st.markdown(f"**Current Bankroll:** ${bankroll_progress[-1]:.2f}")
        else:
            st.markdown(f"**Current Bankroll:** ${initial_bankroll:.2f}")

    # Reset Game
    with st.expander("Reset Game", expanded=False):
        if st.button("Reset History and Bankroll"):
            st.session_state.history = []
            st.session_state.initial_bankroll = 1000.0
            st.session_state.base_bet = 10.0
            st.session_state.money_management_strategy = "Fixed 5% of Bankroll"
            st.session_state.ai_mode = "Conservative"
            st.rerun()

if __name__ == "__main__":
    main()
