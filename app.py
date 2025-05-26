# --- Constants ---
GRID = [
    [0, 1, 2, 3, 4, 4, 3, 2, 1],
    [1, 0, 1, 3, 4, 4, 4, 3, 2],
    [2, 1, 0, 2, 3, 4, 5, 4, 3],
    [3, 3, 2, 0, 2, 4, 5, 6, 5],
    [4, 4, 3, 2, 0, 2, 5, 7, 7],
    [4, 4, 4, 4, 2, 0, 3, 7, 9],
    [3, 4, 5, 5, 5, 3, 0, 5, 9],
    [2, 3, 4, 6, 7, 7, 5, 0, 8],
    [1, 2, 3, 5, 7, 9, 9, 8, 0],
    [1, 1, 2, 3, 5, 8, 11, 15, 15],
    [0, 0, 1, 2, 4, 8, 15, 15, 30]
]

# --- AI Prediction ---
def predict_next_outcome(sequence, model, scaler):
    if len(sequence) < 4 or model is None or scaler is None:
        return 'P', 0.5
    window = sequence[-4:]
    window_key = ''.join(window)
    if window_key in PREDICTION_CACHE:
        return PREDICTION_CACHE[window_key]
    features = [OUTCOME_MAPPING[window[j]] for j in range(4)] + [
        st.session_state.time_before_last.get(k, len(sequence) + 1) / (len(sequence) + 1)
        for k in ['P', 'B']
    ] + [st.session_state.current_streak / 10.0, st.session_state.current_chop_count / 10.0,
         st.session_state.bets_won / max(st.session_state.bets_placed, 1)]
    shoe_features = get_shoe_features(sequence)
    X_scaled = scaler.transform([features + shoe_features])
    probs = model.predict_proba(X_scaled)[0]
    predicted_idx = np.argmax(probs)
    result = (REVERSE_MAPPING[predicted_idx], probs[predicted_idx])
    PREDICTION_CACHE[window_key] = result
    if len(PREDICTION_CACHE) > 1000:
        PREDICTION_CACHE.pop(next(iter(PREDICTION_CACHE)))
    return result

# --- Betting Logic ---
def place_result(result: str):
    try:
        if st.session_state.stop_loss_enabled and st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage:
            if not st.session_state.safety_net_enabled:
                reset_session()
                st.warning(f"Stop-loss triggered at {st.session_state.stop_loss_percentage*100:.0f}%. Game reset.")
                return
        if st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.safety_net_percentage and st.session_state.safety_net_enabled:
            reset_session()
            st.info(f"Safety net triggered at {st.session_state.safety_net_percentage*100:.0f}%. Game reset.")
        if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
            reset_session()
            st.success(f"Win limit reached at {st.session_state.win_limit*100:.0f}%. Game reset.")
            return
        profit = st.session_state.bankroll - st.session_state.initial_bankroll
        if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0 and profit >= st.session_state.initial_bankroll * st.session_state.target_profit_percentage:
            reset_session()
            st.success(f"Target profit reached: ${profit:.2f} ({st.session_state.target_profit_percentage*100:.0f}%). Game reset.")
            return
        if st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0 and profit >= st.session_state.target_profit_units:
            reset_session()
            st.success(f"Target profit reached: ${profit:.2f} (Target: ${st.session_state.target_profit_units:.2f}). Game reset.")
            return

        previous_state = {
            'bankroll': st.session_state.bankroll, 't3_level': st.session_state.t3_level, 't3_results': st.session_state.t3_results.copy(),
            'parlay_step': st.session_state.parlay_step, 'parlay_wins': st.session_state.parlay_wins, 'parlay_using_base': st.session_state.parlay_using_base,
            'parlay_step_changes': st.session_state.parlay_step_changes, 'parlay_peak_step': st.session_state.parlay_peak_step,
            'moon_level': st.session_state.moon_level, 'moon_level_changes': st.session_state.moon_level_changes, 'moon_peak_level': st.session_state.moon_peak_level,
            'four_tier_level': st.session_state.four_tier_level, 'four_tier_step': st.session_state.four_tier_step, 'four_tier_losses': st.session_state.four_tier_losses,
            'flatbet_levelup_level': st.session_state.flatbet_levelup_level, 'flatbet_levelup_net_loss': st.session_state.flatbet_levelup_net_loss,
            'bets_placed': st.session_state.bets_placed, 'bets_won': st.session_state.bets_won, 'pending_bet': st.session_state.pending_bet, 
            'shoe_completed': st.session_state.shoe_completed, 'grid_pos': st.session_state.grid_pos.copy(),
            'oscar_cycle_profit': st.session_state.oscar_cycle_profit, 'oscar_current_bet_level': st.session_state.oscar_current_bet_level,
            'current_streak': st.session_state.current_streak, 'current_streak_type': st.session_state.current_streak_type,
            'longest_streak': st.session_state.longest_streak, 'longest_streak_type': st.session_state.longest_streak_type,
            'current_chop_count': st.session_state.current_chop_count, 'longest_chop': st.session_state.longest_chop,
            'level_1222': st.session_state.level_1222, 'next_bet_multiplier_1222': st.session_state.next_bet_multiplier_1222,
            'rounds_1222': st.session_state.rounds_1222, 'level_start_bankroll_1222': st.session_state.level_start_bankroll_1222,
            'last_positions': st.session_state.last_positions.copy(), 'time_before_last': st.session_state.time_before_last.copy(),
            'prediction_accuracy': st.session_state.prediction_accuracy.copy()
        }

        if result in ['P', 'B']:
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']] + [result]
            if len(valid_sequence) == 1 or st.session_state.current_streak_type != result:
                st.session_state.current_streak = 1
                st.session_state.current_streak_type = result
            else:
                st.session_state.current_streak += 1
            if st.session_state.current_streak > st.session_state.longest_streak:
                st.session_state.longest_streak = st.session_state.current_streak
                st.session_state.longest_streak_type = result
            if len(valid_sequence) > 1 and valid_sequence[-2] != result:
                st.session_state.current_chop_count += 1
            else:
                st.session_state.current_chop_count = 0
            if st.session_state.current_chop_count > st.session_state.longest_chop:
                st.session_state.longest_chop = st.session_state.current_chop_count
        else:
            st.session_state.current_streak = 0
            st.session_state.current_streak_type = None
            if st.session_state.current_chop_count > st.session_state.longest_chop:
                st.session_state.longest_chop = st.session_state.current_chop_count
            st.session_state.current_chop_count = 0

        bet_amount = 0
        bet_selection = None
        bet_outcome = None
        confidence = 0.0
        if st.session_state.pending_bet and result in ['P', 'B']:
            bet_amount, bet_selection = st.session_state.pending_bet
            st.session_state.bets_placed += 1
            if result == bet_selection:
                winnings = bet_amount * (0.95 if bet_selection == 'B' else 1.0)
                st.session_state.bankroll += winnings
                st.session_state.bets_won += 1
                bet_outcome = 'win'
                if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                    if st.session_state.money_management == 'T3':
                        if not st.session_state.t3_results:
                            st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                        st.session_state.t3_results.append('W')
                    elif st.session_state.money_management == 'Parlay16':
                        st.session_state.parlay_wins += 1
                        if st.session_state.parlay_wins == 2:
                            old_step = st.session_state.parlay_step
                            st.session_state.parlay_step = 1
                            st.session_state.parlay_wins = 0
                            st.session_state.parlay_using_base = True
                            if old_step != st.session_state.parlay_step:
                                st.session_state.parlay_step_changes += 1
                            st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                        else:
                            st.session_state.parlay_using_base = False
                    elif st.session_state.money_management == 'Moon':
                        st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                    elif st.session_state.money_management == 'FourTier':
                        st.session_state.four_tier_level = 1
                        st.session_state.four_tier_step = 1
                        st.session_state.four_tier_losses = 0
                        st.session_state.shoe_completed = True
                        st.session_state.advice = "Win recorded. Reset for a new shoe."
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += winnings / st.session_state.base_bet
                    elif st.session_state.money_management == 'Grid':
                        st.session_state.grid_pos[1] += 1
                        if st.session_state.grid_pos[1] >= len(GRID[0]):
                            st.session_state.grid_pos[1] = 0
                            if st.session_state.grid_pos[0] < len(GRID) - 1:
                                st.session_state.grid_pos[0] += 1
                        if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                            st.session_state.grid_pos = [0, 0]
                    elif st.session_state.money_management == 'OscarGrind':
                        st.session_state.oscar_cycle_profit += winnings
                        if st.session_state.oscar_cycle_profit >= st.session_state.base_bet:
                            st.session_state.oscar_current_bet_level = 1
                            st.session_state.oscar_cycle_profit = 0.0
                        else:
                            next_bet_level = st.session_state.oscar_current_bet_level + 1
                            potential_winnings = st.session_state.base_bet * next_bet_level * (0.95 if bet_selection == 'B' else 1.0)
                            if st.session_state.oscar_cycle_profit + potential_winnings > st.session_state.base_bet:
                                next_bet_level = max(1, int((st.session_state.base_bet - st.session_state.oscar_cycle_profit) / (st.session_state.base_bet * (0.95 if bet_selection == 'B' else 1.0)) + 0.99))
                            st.session_state.oscar_current_bet_level = next_bet_level
                    elif st.session_state.money_management == '1222':
                        st.session_state.next_bet_multiplier_1222 = 2
            else:
                st.session_state.bankroll -= bet_amount
                bet_outcome = 'loss'
                if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                    if st.session_state.money_management == 'T3':
                        st.session_state.t3_results.append('L')
                    elif st.session_state.money_management == 'Parlay16':
                        st.session_state.parlay_wins = 0
                        old_step = st.session_state.parlay_step
                        st.session_state.parlay_step = min(st.session_state.parlay_step + 1, 16)
                        st.session_state.parlay_using_base = True
                        if old_step != st.session_state.parlay_step:
                            st.session_state.parlay_step_changes += 1
                        st.session_state.parlay_peak_step = max(st.session_state.parlay_peak_step, old_step)
                    elif st.session_state.money_management == 'Moon':
                        old_level = st.session_state.moon_level
                        st.session_state.moon_level += 1
                        if old_level != st.session_state.moon_level:
                            st.session_state.moon_level_changes += 1
                        st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                    elif st.session_state.money_management == 'FourTier':
                        st.session_state.four_tier_losses += 1
                        if st.session_state.four_tier_losses == 1:
                            st.session_state.four_tier_step = 2
                        elif st.session_state.four_tier_losses >= 2:
                            st.session_state.four_tier_level = min(st.session_state.four_tier_level + 1, 4)
                            st.session_state.four_tier_step = 1
                            st.session_state.four_tier_losses = 0
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss -= bet_amount / st.session_state.base_bet
                        current_level = st.session_state.flatbet_levelup_level
                        if current_level < 5 and st.session_state.flatbet_levelup_net_loss <= FLATBET_LEVELUP_THRESHOLDS[current_level]:
                            st.session_state.flatbet_levelup_level = min(st.session_state.flatbet_levelup_level + 1, 5)
                            st.session_state.flatbet_levelup_net_loss = 0.0
                    elif st.session_state.money_management == 'Grid':
                        st.session_state.grid_pos[0] += 1
                        if st.session_state.grid_pos[0] >= len(GRID):
                            st.session_state.grid_pos = [0, 0]
                        if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                            st.session_state.grid_pos = [0, 0]
                    elif st.session_state.money_management == '1222':
                        st.session_state.next_bet_multiplier_1222 = 1
            if st.session_state.money_management == 'T3' and len(st.session_state.t3_results) == 3:
                wins = st.session_state.t3_results.count('W')
                losses = st.session_state.t3_results.count('L')
                st.session_state.t3_level = max(1, st.session_state.t3_level - 1 if wins > losses else st.session_state.t3_level + 1 if losses > wins else st.session_state.t3_level)
                st.session_state.t3_results = []
            if st.session_state.money_management == '1222' and bet_amount > 0:
                st.session_state.rounds_1222 += 1
                if st.session_state.rounds_1222 >= 5:
                    if st.session_state.bankroll >= st.session_state.peak_bankroll:
                        st.session_state.level_1222 = 1
                        st.session_state.next_bet_multiplier_1222 = 1
                        st.session_state.rounds_1222 = 0
                        st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
                    elif st.session_state.bankroll > st.session_state.level_start_bankroll_1222:
                        st.session_state.level_1222 = max(1, st.session_state.level_1222 - 1)
                        st.session_state.next_bet_multiplier_1222 = 1
                        st.session_state.rounds_1222 = 0
                        st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
                    else:
                        st.session_state.level_1222 += 1
                        st.session_state.next_bet_multiplier_1222 = 1
                        st.session_state.rounds_1222 = 0
                        st.session_state.level_start_bankroll_1222 = st.session_state.bankroll
            st.session_state.peak_bankroll = max(st.session_state.peak_bankroll, st.session_state.bankroll)
            st.session_state.pending_bet = None

        if result in ['P', 'B', 'T']:
            st.session_state.sequence.append(result)
            current_position = len(st.session_state.sequence)
            st.session_state.last_positions[result].append(current_position)
            if len(st.session_state.last_positions[result]) > 2:
                st.session_state.last_positions[result].pop(0)
            for outcome in ['P', 'B', 'T']:
                if len(st.session_state.last_positions[outcome]) >= 2:
                    st.session_state.time_before_last[outcome] = current_position - st.session_state.last_positions[outcome][-2]
                else:
                    st.session_state.time_before_last[outcome] = current_position + 1

        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
        if len(valid_sequence) >= 5:
            st.session_state.ml_model, st.session_state.ml_scaler = train_ml_model(valid_sequence)

        if st.session_state.bet_history and st.session_state.bet_history[-1].get('Bet_Selection'):
            last_bet = st.session_state.bet_history[-1]
            predicted = last_bet['Bet_Selection']
            actual = result
            if predicted == actual:
                total_preds = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection') == predicted)
                correct_preds = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection') == predicted and h['Result'] == predicted)
                st.session_state.prediction_accuracy[predicted] = (correct_preds / total_preds * 100) if total_preds > 0 else 0.0
                total_bets = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection'))
                total_correct = sum(1 for h in st.session_state.bet_history if h.get('Bet_Selection') and h['Result'] == h['Bet_Selection'])
                st.session_state.prediction_accuracy['total'] = (total_correct / total_bets * 100) if total_bets > 0 else 0.0

        st.session_state.bet_history.append({
            "Result": result,
            "Bet_Amount": bet_amount,
            "Bet_Selection": bet_selection,
            "Bet_Outcome": bet_outcome,
            "Money_Management": st.session_state.money_management,
            "AI_Prediction": st.session_state.advice,
            "Confidence": f"{confidence:.1f}%",
            "Previous_State": previous_state
        })
        if len(st.session_state.bet_history) > HISTORY_LIMIT:
            st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

        if len(valid_sequence) < START_BETTING_HAND:
            st.session_state.pending_bet = None
            st.session_state.advice = f"Need {START_BETTING_HAND - len(valid_sequence)} more results to start betting"
        elif len(valid_sequence) >= START_BETTING_HAND and result in ['P', 'B']:
            if len(st.session_state.sequence) >= SHOE_SIZE:
                st.session_state.shoe_completed = True
            if st.session_state.shoe_completed and not st.session_state.safety_net_enabled:
                st.session_state.pending_bet = None
                st.session_state.advice = "Shoe completed. AI-only betting stopped."
                return
            bet_selection, confidence = predict_next_outcome(valid_sequence, st.session_state.ml_model, st.session_state.ml_scaler)
            confidence *= 100
            dynamic_threshold = get_dynamic_confidence_threshold()
            strategy_used = 'AI'

            if confidence >= dynamic_threshold and bet_selection in ['P', 'B']:
                bet_amount = calculate_bet_amount(bet_selection, confidence)
                if bet_amount <= st.session_state.bankroll and bet_amount > 0:
                    st.session_state.pending_bet = (bet_amount, bet_selection)
                    strategy_info = f"{st.session_state.money_management}"
                    if st.session_state.shoe_completed and st.session_state.safety_net_enabled:
                        strategy_info = "Safety Net (Flatbet)"
                    elif st.session_state.money_management == 'T3':
                        strategy_info += f" Level {st.session_state.t3_level}"
                    elif st.session_state.money_management == 'Parlay16':
                        strategy_info += f" Step {st.session_state.parlay_step}/16"
                    elif st.session_state.money_management == 'Moon':
                        strategy_info += f" Level {st.session_state.moon_level}"
                    elif st.session_state.money_management == 'FourTier':
                        strategy_info += f" Level {st.session_state.four_tier_level} Step {st.session_state.four_tier_step}"
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        strategy_info += f" Level {st.session_state.flatbet_levelup_level}"
                    elif st.session_state.money_management == 'Grid':
                        strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                    elif st.session_state.money_management == 'OscarGrind':
                        strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level}"
                    elif st.session_state.money_management == '1222':
                        strategy_info += f" Level {st.session_state.level_1222}, Rounds {st.session_state.rounds_1222}, Bet: {st.session_state.next_bet_multiplier_1222 * st.session_state.level_1222}u"
                    st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {strategy_used}: {confidence:.1f}%, Threshold: {dynamic_threshold:.1f}%)"
                else:
                    st.session_state.pending_bet = None
                    st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll or loss streak)"
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (low confidence: {confidence:.1f}% < {dynamic_threshold:.1f}% or Tie)"
    except Exception as e:
        st.error(f"Error in place_result: {str(e)}")

# --- Insights ---
def render_insights():
    with st.expander("AI Insights", expanded=False):
        if not st.session_state.sequence:
            st.write("No data available for insights.")
        else:
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B', 'T']]
            counts = Counter(valid_sequence)
            total = len(valid_sequence)
            st.markdown(f"**Shoe Composition**:<br>"
                        f"Player: {counts.get('P', 0)} ({counts.get('P', 0)/total*100:.1f}%)<br>"
                        f"Banker: {counts.get('B', 0)} ({counts.get('B', 0)/total*100:.1f}%)<br>"
                        f"Tie: {counts.get('T', 0)} ({counts.get('T', 0)/total*100:.1f}%)",
                        unsafe_allow_html=True)
            bigrams = Counter(extract_ngrams(valid_sequence, 2))
            st.markdown("**Top Bigrams**:<br>" + "<br>".join(
                [f"{k}: {v} ({v/len(bigrams)*100:.1f}%)" for k, v in bigrams.most_common(3)] if bigrams else ["No bigrams available"]
            ), unsafe_allow_html=True)
            
            if st.session_state.bet_history:
                confidences = [float(h["Confidence"].replace("%", "")) for h in st.session_state.bet_history[-5:] if h.get("Confidence")]
                if confidences:
                    st.markdown("**Recent Confidence Trend**")
                    st.markdown("""
                    ```chartjs
                    {
                        "type": "line",
                        "data": {
                            "labels": ["Bet 1", "Bet 2", "Bet 3", "Bet 4", "Bet 5"][:len(confidences)],
                            "datasets": [{
                                "label": "Confidence (%)",
                                "data": confidences,
                                "borderColor": "#3182ce",
                                "backgroundColor": "rgba(49, 100, 206, 0.2)",
                                "fill": true,
                                "pointRadius": 4
                            }]
                        },
                        "options": {
                            "scales": {
                                "y": {"min": 50, "max": 100, "title": {"display": true, "text": "Confidence (%)"}},
                                "x": {"title": {"display": true, "text": "Bet Number"}}
                            },
                            "plugins": {"legend": {"display": false}}
                        }
                    }
