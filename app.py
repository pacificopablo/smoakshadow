import random
import logging

# Set up logging to debug blank page issues
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def place_result(result: str):
    logging.debug(f"Entering place_result with result: {result}")
    try:
        # Check limits
        logging.debug("Checking stop-loss and safety net limits")
        if st.session_state.stop_loss_enabled:
            stop_loss_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.stop_loss_percentage
            if stop_loss_triggered and not st.session_state.safety_net_enabled:
                reset_session()
                st.warning(f"Stop-loss triggered at {st.session_state.stop_loss_percentage*100:.0f}% of initial bankroll. Game reset.")
                logging.info("Stop-loss triggered, session reset")
                return

        safety_net_triggered = st.session_state.bankroll <= st.session_state.initial_bankroll * st.session_state.safety_net_percentage
        if safety_net_triggered and st.session_state.safety_net_enabled:
            reset_session()
            st.info(f"Safety net triggered at {st.session_state.safety_net_percentage*100:.0f}%. Game reset to base bet.")
            logging.info("Safety net triggered, session reset")
            return
        
        if st.session_state.bankroll >= st.session_state.initial_bankroll * st.session_state.win_limit:
            reset_session()
            st.success(f"Win limit reached at {st.session_state.win_limit*100:.0f}% of initial bankroll. Game reset.")
            logging.info("Win limit reached, session reset")
            return

        current_profit = st.session_state.bankroll - st.session_state.initial_bankroll
        if st.session_state.target_profit_option == 'Profit %' and st.session_state.target_profit_percentage > 0:
            if current_profit >= st.session_state.initial_bankroll * st.session_state.target_profit_percentage:
                reset_session()
                st.success(f"Target profit reached: ${current_profit:.2f} ({st.session_state.target_profit_percentage*100:.0f}%). Game reset.")
                logging.info(f"Target profit reached: ${current_profit:.2f}")
                return
        elif st.session_state.target_profit_option == 'Units' and st.session_state.target_profit_units > 0:
            if current_profit >= st.session_state.target_profit_units:
                reset_session()
                st.success(f"Target profit reached: ${current_profit:.2f} (Target: ${st.session_state.target_profit_units:.2f}). Game reset.")
                logging.info(f"Target profit (units) reached: ${current_profit:.2f}")
                return

        # Save previous state for undo
        logging.debug("Saving previous state for undo")
        previous_state = {
            'bankroll': st.session_state.bankroll,
            't3_level': st.session_state.t3_level,
            't3_results': st.session_state.t3_results.copy(),
            'parlay_step': st.session_state.parlay_step,
            'parlay_wins': st.session_state.parlay_wins,
            'parlay_using_base': st.session_state.parlay_using_base,
            'parlay_step_changes': st.session_state.parlay_step_changes,
            'parlay_peak_step': st.session_state.parlay_peak_step,
            'moon_level': st.session_state.moon_level,
            'moon_level_changes': st.session_state.moon_level_changes,
            'moon_peak_level': st.session_state.moon_peak_level,
            'four_tier_level': st.session_state.four_tier_level,
            'four_tier_step': st.session_state.four_tier_step,
            'four_tier_losses': st.session_state.four_tier_losses,
            'flatbet_levelup_level': st.session_state.flatbet_levelup_level,
            'flatbet_levelup_net_loss': st.session_state.flatbet_levelup_net_loss,
            'bets_placed': st.session_state.bets_placed,
            'bets_won': st.session_state.bets_won,
            'transition_counts': st.session_state.transition_counts.copy(),
            'pending_bet': st.session_state.pending_bet,
            'shoe_completed': st.session_state.shoe_completed,
            'grid_pos': st.session_state.grid_pos.copy(),
            'oscar_cycle_profit': st.session_state.oscar_cycle_profit,
            'oscar_current_bet_level': st.session_state.oscar_current_bet_level,
            'sequence_bet_index': st.session_state.sequence_bet_index
        }

        # Update transition counts
        logging.debug("Updating transition counts")
        if len(st.session_state.sequence) >= 1 and result in ['P', 'B']:
            prev_result = st.session_state.sequence[-1]
            if prev_result in ['P', 'B']:
                transition = f"{prev_result}{result}"
                st.session_state.transition_counts[transition] += 1

        # Resolve pending bet
        bet_amount = 0
        bet_selection = None
        bet_outcome = None
        logging.debug("Resolving pending bet")
        if st.session_state.pending_bet and result in ['P', 'B']:
            bet_amount, bet_selection = st.session_state.pending_bet
            st.session_state.bets_placed += 1
            if result == bet_selection:
                if bet_selection == 'B':
                    winnings = bet_amount * 0.95
                    st.session_state.bankroll += winnings
                    if st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += winnings / st.session_state.base_bet
                    elif st.session_state.money_management == 'OscarGrind':
                        st.session_state.oscar_cycle_profit += winnings
                else:
                    winnings = bet_amount
                    st.session_state.bankroll += winnings
                    if st.session_state.money_management == 'FlatbetLevelUp':
                        st.session_state.flatbet_levelup_net_loss += winnings / st.session_state.base_bet
                    elif st.session_state.money_management == 'OscarGrind':
                        st.session_state.oscar_cycle_profit += winnings
                st.session_state.bets_won += 1
                bet_outcome = 'win'
                st.session_state.sequence_bet_index = 0  # Reset sequence on win
                if not (st.session_state.shoe_completed and st.session_state.safety_net_enabled):
                    if st.session_state.money_management == 'T3':
                        if len(st.session_state.t3_results) == 0:
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
                        old_level = st.session_state.moon_level
                        st.session_state.moon_level = old_level
                        if old_level != st.session_state.moon_level:
                            st.session_state.moon_level_changes += 1
                        st.session_state.moon_peak_level = max(st.session_state.moon_peak_level, st.session_state.moon_level)
                    elif st.session_state.money_management == 'FourTier':
                        st.session_state.four_tier_level = 1
                        st.session_state.four_tier_step = 1
                        st.session_state.four_tier_losses = 0
                        st.session_state.shoe_completed = True
                        st.session_state.advice = "Win recorded. Reset for a new shoe."
                    elif st.session_state.money_management == 'FlatbetLevelUp':
                        pass
                    elif st.session_state.money_management == 'Grid':
                        st.session_state.grid_pos[1] += 1
                        if st.session_state.grid_pos[1] >= len(GRID[0]):
                            st.session_state.grid_pos[1] = 0
                            if st.session_state.grid_pos[0] < len(GRID) - 1:
                                st.session_state.grid_pos[0] += 1
                        if GRID[st.session_state.grid_pos[0]][st.session_state.grid_pos[1]] == 0:
                            st.session_state.grid_pos = [0, 0]
                    elif st.session_state.money_management == 'OscarGrind':
                        if st.session_state.oscar_cycle_profit >= st.session_state.base_bet:
                            st.session_state.oscar_current_bet_level = 1
                            st.session_state.oscar_cycle_profit = 0.0
                        else:
                            next_bet_level = st.session_state.oscar_current_bet_level + 1
                            potential_winnings = st.session_state.base_bet * next_bet_level * (0.95 if bet_selection == 'B' else 1.0)
                            if st.session_state.oscar_cycle_profit + potential_winnings > st.session_state.base_bet:
                                next_bet_level = max(1, int((st.session_state.base_bet - st.session_state.oscar_cycle_profit) / (st.session_state.base_bet * (0.95 if bet_selection == 'B' else 1.0)) + 0.99))
                            st.session_state.oscar_current_bet_level = next_bet_level
            else:
                st.session_state.bankroll -= bet_amount
                if st.session_state.money_management == 'FlatbetLevelUp':
                    st.session_state.flatbet_levelup_net_loss -= bet_amount / st.session_state.base_bet
                elif st.session_state.money_management == 'OscarGrind':
                    st.session_state.oscar_cycle_profit -= bet_amount
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
                    elif st.session_state.money_management == 'OscarGrind':
                        pass  # Bet level stays the same after a loss
            if st.session_state.money_management == 'T3' and len(st.session_state.t3_results) == 3:
                wins = st.session_state.t3_results.count('W')
                losses = st.session_state.t3_results.count('L')
                if wins > losses:
                    st.session_state.t3_level = max(1, st.session_state.t3_level - 1)
                elif losses > wins:
                    st.session_state.t3_level += 1
                st.session_state.t3_results = []
            st.session_state.pending_bet = None
            logging.debug(f"Pending bet resolved: outcome={bet_outcome}, bankroll={st.session_state.bankroll}")

        # Add result to sequence
        logging.debug(f"Adding result {result} to sequence")
        if result in ['P', 'B', 'T']:
            st.session_state.sequence.append(result)

        # Increment sequence_bet_index for P or B results (unless reset by a win)
        if result in ['P', 'B'] and (bet_outcome != 'win'):
            st.session_state.sequence_bet_index += 1

        # Store bet history
        logging.debug("Storing bet history")
        st.session_state.bet_history.append({
            "Result": result,
            "Bet_Amount": bet_amount,
            "Bet_Selection": bet_selection,
            "Bet_Outcome": bet_outcome,
            "T3_Level": st.session_state.t3_level if st.session_state.money_management == 'T3' else "-",
            "Parlay_Step": st.session_state.parlay_step if st.session_state.money_management == 'Parlay16' else "-",
            "Moon_Level": st.session_state.moon_level if st.session_state.money_management == 'Moon' else "-",
            "FourTier_Level": st.session_state.four_tier_level if st.session_state.money_management == 'FourTier' else "-",
            "FourTier_Step": st.session_state.four_tier_step if st.session_state.money_management == 'FourTier' else "-",
            "FlatbetLevelUp_Level": st.session_state.flatbet_levelup_level if st.session_state.money_management == 'FlatbetLevelUp' else "-",
            "FlatbetLevelUp_Net_Loss": round(st.session_state.flatbet_levelup_net_loss, 2) if st.session_state.money_management == 'FlatbetLevelUp' else "-",
            "Grid_Pos": f"({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})" if st.session_state.money_management == 'Grid' else "-",
            "Oscar_Bet_Level": st.session_state.oscar_current_bet_level if st.session_state.money_management == 'OscarGrind' else "-",
            "Oscar_Cycle_Profit": round(st.session_state.oscar_cycle_profit, 2) if st.session_state.money_management == 'OscarGrind' else "-",
            "Sequence_Bet_Index": st.session_state.sequence_bet_index % len(BET_SEQUENCE) if st.session_state.sequence_bet_index > 0 or bet_outcome == 'win' else "-",
            "Money_Management": st.session_state.money_management,
            "Safety_Net": "On" if st.session_state.safety_net_enabled else "Off",
            "Previous_State": previous_state
        })
        if len(st.session_state.bet_history) > HISTORY_LIMIT:
            st.session_state.bet_history = st.session_state.bet_history[-HISTORY_LIMIT:]

        # AI-driven bet selection (from aigood)
        logging.debug("Starting AI-driven bet selection")
        if result in ['P', 'B']:
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']][-6:]
            p_count = valid_sequence.count('P')
            total = len(valid_sequence)
            p_prob = p_count / total if total > 0 else 0.5
            b_prob = 1 - p_prob
            streak = False
            if len(valid_sequence) >= 3 and len(set(valid_sequence[-3:])) == 1:
                streak = True
                if valid_sequence[-1] == 'P':
                    p_prob += 0.2
                    b_prob -= 0.2
                else:
                    b_prob += 0.2
                    p_prob -= 0.2
            p_prob = max(0, min(1, p_prob + random.uniform(-0.1, 0.1)))
            b_prob = 1 - p_prob
            bet_selection = 'P' if random.random() < p_prob else 'B'
            rationale = f"AI Probability: P {p_prob*100:.0f}%, B {b_prob*100:.0f}%"
            if streak:
                rationale += f", Streak of {valid_sequence[-1]}"
            bet_amount = calculate_bet_amount(bet_selection)
            if bet_amount <= st.session_state.bankroll:
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
                    strategy_info += f" Level {st.session_state.flatbet_levelup_level} Net Loss {st.session_state.flatbet_levelup_net_loss:.2f}"
                elif st.session_state.money_management == 'Grid':
                    strategy_info += f" Grid ({st.session_state.grid_pos[0]},{st.session_state.grid_pos[1]})"
                elif st.session_state.money_management == 'OscarGrind':
                    strategy_info += f" Bet Level {st.session_state.oscar_current_bet_level} Cycle Profit ${st.session_state.oscar_cycle_profit:.2f}"
                st.session_state.advice = f"Bet ${bet_amount:.2f} on {bet_selection} ({strategy_info}, {rationale})"
                logging.debug(f"Bet placed: ${bet_amount:.2f} on {bet_selection}, rationale: {rationale}")
            else:
                st.session_state.pending_bet = None
                st.session_state.advice = f"Skip betting (bet ${bet_amount:.2f} exceeds bankroll)"
                logging.debug(f"Bet skipped: amount ${bet_amount:.2f} exceeds bankroll")

        if len(st.session_state.sequence) >= SHOE_SIZE:
            reset_session()
            st.success(f"Shoe of {SHOE_SIZE} hands completed. Game reset.")
            logging.info("Shoe completed, session reset")
    except Exception as e:
        st.error(f"Error processing result: {str(e)}")
        logging.error(f"Error in place_result: {str(e)}", exc_info=True)
