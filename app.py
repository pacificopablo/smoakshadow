import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
import time
from typing import Tuple, Dict, Optional, List
import uuid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Constants ---
# (Existing constants remain unchanged)

# --- Machine Learning Setup ---
def prepare_ml_features(sequence: List[str], streak_count: int, chop_count: int, double_count: int, shoe_bias: float) -> np.ndarray:
    # Encode the last 8 outcomes (or fewer if sequence is shorter)
    le = LabelEncoder()
    le.fit(['P', 'B', 'T'])
    padded_sequence = ['None'] * (8 - len(sequence)) + sequence[-8:] if len(sequence) < 8 else sequence[-8:]
    encoded_sequence = le.transform(padded_sequence).reshape(1, -1)
    
    # Additional features: pattern metrics and shoe bias
    pattern_features = np.array([[streak_count, chop_count, double_count, shoe_bias]])
    
    # Combine sequence and pattern features
    features = np.hstack((encoded_sequence, pattern_features))
    return features

def train_ml_model(history: List[Dict]) -> RandomForestClassifier:
    if len(history) < 10:
        # Return a default model if insufficient data
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    X, y = [], []
    le = LabelEncoder()
    le.fit(['P', 'B', 'T'])
    
    for i in range(len(history) - 1):
        if history[i]['Result'] in ['P', 'B', 'T']:
            sequence = [h['Result'] for h in history[:i+1] if h['Result'] in ['P', 'B', 'T']]
            if len(sequence) >= 8:
                _, _, _, _, streak_count, chop_count, double_count, _, shoe_bias, _ = analyze_patterns(sequence)
                features = prepare_ml_features(sequence, streak_count, chop_count, double_count, shoe_bias)
                X.append(features.flatten())
                y.append(history[i+1]['Result'])
    
    if not X or not y:
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    X = np.array(X)
    y = le.transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- Modified Prediction Logic ---
def smart_predict() -> Tuple[Optional[str], float, Dict]:
    sequence = [x for x in st.session_state.sequence if x in ['P', 'B', 'T']]
    if len(sequence) < 8:
        return None, 0.0, {'Status': 'Waiting for 9th hand'}
    
    recent_sequence = sequence[-WINDOW_SIZE:] if len(sequence) >= WINDOW_SIZE else sequence
    (bigram_transitions, trigram_transitions, fourgram_transitions, pattern_transitions,
     streak_count, chop_count, double_count, volatility, shoe_bias, insights) = analyze_patterns(recent_sequence)
    st.session_state.pattern_volatility = volatility
    st.session_state.trend_score = {'streak': insights['streak'], 'chop': insights['chop'], 'double': insights['double']}
    
    prior_p, prior_b = 44.62 / 100, 45.86 / 100
    weights = calculate_weights(streak_count, chop_count, double_count, shoe_bias)
    prob_p = prob_b = total_weight = 0
    insights = {'Volatility': f"{volatility:.2f}"}
    
    # Machine Learning Prediction
    ml_model = train_ml_model(st.session_state.history)
    features = prepare_ml_features(recent_sequence, streak_count, chop_count, double_count, shoe_bias)
    ml_probs = ml_model.predict_proba(features)[0]
    le = LabelEncoder()
    le.fit(['P', 'B', 'T'])
    ml_pred = le.inverse_transform([np.argmax(ml_probs)])[0]
    ml_conf = max(ml_probs) * 100
    prob_p += weights.get('ml', 0.3) * ml_probs[le.transform(['P'])[0]]
    prob_b += weights.get('ml', 0.3) * ml_probs[le.transform(['B'])[0]]
    total_weight += weights.get('ml', 0.3)
    insights['ML_Model'] = f"{weights.get('ml', 0.3)*100:.0f}% (P: {ml_probs[le.transform(['P'])[0]]*100:.1f}%, B: {ml_probs[le.transform(['B'])[0]]*100:.1f}%)"
    
    # Existing pattern-based predictions
    if len(recent_sequence) >= 2:
        bigram = tuple(recent_sequence[-2:])
        total = sum(bigram_transitions[bigram].values())
        if total > 0:
            p_prob = bigram_transitions[bigram]['P'] / total
            b_prob = bigram_transitions[bigram]['B'] / total
            prob_p += weights['bigram'] * (prior_p + p_prob) / (1 + total)
            prob_b += weights['bigram'] * (prior_b + b_prob) / (1 + total)
            total_weight += weights['bigram']
            insights['Bigram'] = f"{weights['bigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
    
    if len(recent_sequence) >= 3:
        trigram = tuple(recent_sequence[-3:])
        total = sum(trigram_transitions[trigram].values())
        if total > 0:
            p_prob = trigram_transitions[trigram]['P'] / total
            b_prob = trigram_transitions[trigram]['B'] / total
            prob_p += weights['trigram'] * (prior_p + p_prob) / (1 + total)
            prob_b += weights['trigram'] * (prior_b + b_prob) / (1 + total)
            total_weight += weights['trigram']
            insights['Trigram'] = f"{weights['trigram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
    
    fourgram_pred = None
    fourgram_conf = 0.0
    if len(recent_sequence) >= 4:
        fourgram = recent_sequence[-4:]
        fourgram_key = ''.join(fourgram)
        fourgram_pred = 'B' if fourgram_key in ['PPPP', 'BBBB', 'PPBB', 'BBPP'] else 'P'
        fourgram_conf = 60.0 if fourgram_key in ['PPPP', 'BBBB'] else 50.0
        total = sum(fourgram_transitions[tuple(fourgram)].values())
        if total > 0:
            p_prob = fourgram_transitions[tuple(fourgram)]['P'] / total
            b_prob = fourgram_transitions[tuple(fourgram)]['B'] / total
            prob_p += weights['fourgram'] * (prior_p + p_prob) / (1 + total)
            prob_b += weights['fourgram'] * (prior_b + b_prob) / (1 + total)
            total_weight += weights['fourgram']
            insights['Fourgram'] = f"{weights['fourgram']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
    
    markov_pred = None
    markov_conf = 0.0
    if len(recent_sequence) >= 2:
        last_two = recent_sequence[-2:]
        transitions = defaultdict(lambda: {'P': 0, 'B': 0, 'T': 0})
        for i in range(len(recent_sequence) - 2):
            current = recent_sequence[i:i+2]
            next_outcome = recent_sequence[i+2]
            transitions[''.join(current)][next_outcome] += 1
        current = ''.join(last_two)
        if current in transitions:
            total = sum(transitions[current].values())
            if total > 0:
                p_prob = transitions[current]['P'] / total
                b_prob = transitions[current]['B'] / total
                markov_pred = 'P' if p_prob > b_prob else 'B'
                markov_conf = max(p_prob, b_prob) * 100
                prob_p += weights['markov'] * (prior_p + p_prob) / (1 + total)
                prob_b += weights['markov'] * (prior_b + b_prob) / (1 + total)
                total_weight += weights['markov']
                insights['Markov'] = f"{weights['markov']*100:.0f}% (P: {p_prob*100:.1f}%, B: {b_prob*100:.1f}%)"
    
    if streak_count >= 2:
        streak_prob = min(0.7, 0.5 + streak_count * 0.05) * (0.8 if streak_count > 4 else 1.0)
        current_streak = recent_sequence[-1]
        if current_streak == 'P':
            prob_p += weights['streak'] * streak_prob
            prob_b += weights['streak'] * (1 - streak_prob)
        else:
            prob_b += weights['streak'] * streak_prob
            prob_p += weights['streak'] * (1 - streak_prob)
        total_weight += weights['streak']
        insights['Streak'] = f"{weights['streak']*100:.0f}% ({streak_count} {current_streak})"
    
    if chop_count >= 2:
        next_pred = 'B' if recent_sequence[-1] == 'P' else 'P'
        if next_pred == 'P':
            prob_p += weights['chop'] * 0.6
            prob_b += weights['chop'] * 0.4
        else:
            prob_b += weights['chop'] * 0.6
            prob_p += weights['chop'] * 0.4
        total_weight += weights['chop']
        insights['Chop'] = f"{weights['chop']*100:.0f}% ({chop_count} alternations)"
    
    if double_count >= 1 and len(recent_sequence) >= 2 and recent_sequence[-1] == recent_sequence[-2]:
        double_prob = 0.6
        if recent_sequence[-1] == 'P':
            prob_p += weights['double'] * double_prob
            prob_b += weights['double'] * (1 - double_prob)
        else:
            prob_b += weights['double'] * double_prob
            prob_p += weights['double'] * (1 - double_prob)
        total_weight += weights['double']
        insights['Double'] = f"{weights['double']*100:.0f}% ({recent_sequence[-1]}{recent_sequence[-1]})"
    
    if total_weight > 0:
        prob_p = (prob_p / total_weight) * 100
        prob_b = (prob_b / total_weight) * 100
    else:
        prob_p, prob_b = 44.62, 45.86
    
    if shoe_bias > 0.1:
        prob_p *= 1.05
        prob_b *= 0.95
    elif shoe_bias < -0.1:
        prob_b *= 1.05
        prob_p *= 0.95
    
    final_pred = ml_pred if ml_conf > max(fourgram_conf, markov_conf) else (fourgram_pred if fourgram_conf > markov_conf else markov_pred)
    final_conf = max(ml_conf, fourgram_conf, markov_conf)
    if not final_pred:
        final_pred = 'P' if prob_p > prob_b else 'B'
        final_conf = max(prob_p, prob_b)
    
    if insights.get('streak', 0.0) > 0.6 and recent_sequence and recent_sequence[-1] != 'T':
        final_pred = recent_sequence[-1]
        final_conf += 10.0
    elif insights.get('chop', 0.0) > 0.6 and recent_sequence and recent_sequence[-1] != 'T':
        final_pred = 'P' if recent_sequence[-1] == 'B' else 'B'
        final_conf += 5.0
    
    recent_accuracy = (st.session_state.prediction_accuracy['P'] + st.session_state.prediction_accuracy['B']) / max(st.session_state.prediction_accuracy['total'], 1)
    threshold = 32.0 + (st.session_state.consecutive_losses * 0.5) - (recent_accuracy * 0.8)
    threshold = min(max(threshold, 32.0), 42.0)
    insights['Threshold'] = f"{threshold:.1f}%"
    
    if volatility > 0.5:
        threshold += 1.5
        insights['Volatility'] = f"High (Adjustment: +1.5% threshold)"
    
    final_conf = min(final_conf, 100.0)
    if final_pred == 'P' and prob_p >= threshold:
        return 'P', prob_p, insights
    elif final_pred == 'B' and prob_b >= threshold:
        return 'B', prob_b, insights
    return None, max(prob_p, prob_b), insights

# (Rest of the original code remains unchanged)
