import streamlit as st
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import uuid

# Constants
SEQUENCE_LENGTH = 6

def initialize_session_state():
    """Initialize session state with default values."""
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
        st.session_state.transition_counts = {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0}
        st.session_state.prediction = None
        st.session_state.confidence = 0.0
        st.session_state.model = None
        st.session_state.le = None
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.model, st.session_state.le = train_model()

def reset_session():
    """Reset session state to initial values."""
    st.session_state.update({
        'sequence': [],
        'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
        'prediction': None,
        'confidence': 0.0,
        'session_id': str(uuid.uuid4())
    })
    st.session_state.model, st.session_state.le = train_model()

def generate_baccarat_data(num_games=10000):
    """Generate synthetic Baccarat data with realistic probabilities and streaks."""
    outcomes = ['P', 'B', 'T']
    weights = [0.4462, 0.4586, 0.0952]
    result = []
    i = 0
    while i < num_games:
        outcome = random.choices(outcomes, weights=weights, k=1)[0]
        streak_length = random.choices([1, 2, 3, 4, 5, 6], weights=[0.4, 0.3, 0.15, 0.1, 0.03, 0.02])[0]
        result.extend([outcome] * streak_length)
        i += streak_length
    return result[:num_games]

def prepare_data(outcomes, sequence_length=6):
    """Prepare training data for the Random Forest model."""
    try:
        le = LabelEncoder()
        encoded_outcomes = le.fit_transform(outcomes)
        X, y = [], []
        for i in range(len(encoded_outcomes) - sequence_length):
            X.append(encoded_outcomes[i:i + sequence_length])
            y.append(encoded_outcomes[i + sequence_length])
        return np.array(X), np.array(y), le
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return np.array([]), np.array([]), None

def train_model():
    """Train the Random Forest model on synthetic Baccarat data."""
    try:
        outcomes = generate_baccarat_data()
        X, y, le = prepare_data(outcomes, SEQUENCE_LENGTH)
        if X.size == 0 or y.size == 0 or le is None:
            st.error("Failed to generate training data.")
            return None, None
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, le
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def place_result(result):
    """Process a game result and update prediction."""
    try:
        # Append result and update transition counts
        st.session_state.sequence.append(result)
        if len(st.session_state.sequence) >= 2:
            prev_result = st.session_state.sequence[-2]
            if result in ['P', 'B'] and prev_result in ['P', 'B']:
                transition = f"{prev_result}{result}"
                st.session_state.transition_counts[transition] += 1

        # Predict next outcome if enough data
        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
        if len(valid_sequence) < SEQUENCE_LENGTH:
            st.session_state.prediction = None
            st.session_state.confidence = 0.0
        elif len(valid_sequence) >= SEQUENCE_LENGTH:
            prediction_sequence = valid_sequence[-SEQUENCE_LENGTH:]
            encoded_input = st.session_state.le.transform(prediction_sequence)
            input_array = np.array([encoded_input])
            prediction_probs = st.session_state.model.predict_proba(input_array)[0]
            predicted_class = np.argmax(prediction_probs)
            predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
            confidence = np.max(prediction_probs) * 100
            st.session_state.prediction = predicted_outcome
            st.session_state.confidence = confidence
    except Exception as e:
        st.error(f"Error processing result: {str(e)}")

def render_session_setup():
    """Render UI for starting or resetting a session."""
    with st.expander("Session Setup", expanded=len(st.session_state.sequence) == 0):
        try:
            if st.button("Start Session"):
                initialize_session_state()
                st.success("Session started!")
            if st.session_state.sequence and st.button("Reset Session"):
                reset_session()
                st.success("Session reset!")
        except Exception as e:
            st.error(f"Error setting up session: {str(e)}")

def render_result_input():
    """Render UI for inputting game results."""
    with st.expander("Result Input", expanded=True):
        try:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Player (P)"):
                    place_result('P')
            with col2:
                if st.button("Banker (B)"):
                    place_result('B')
            with col3:
                if st.button("Tie (T)"):
                    place_result('T')
            with col4:
                if st.button("Undo Last"):
                    if st.session_state.sequence:
                        last_result = st.session_state.sequence.pop()
                        if len(st.session_state.sequence) >= 1 and last_result in ['P', 'B'] and st.session_state.sequence[-1] in ['P', 'B']:
                            transition = f"{st.session_state.sequence[-1]}{last_result}"
                            st.session_state.transition_counts[transition] = max(0, st.session_state.transition_counts[transition] - 1)
                        valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
                        if len(valid_sequence) >= SEQUENCE_LENGTH:
                            prediction_sequence = valid_sequence[-SEQUENCE_LENGTH:]
                            encoded_input = st.session_state.le.transform(prediction_sequence)
                            input_array = np.array([encoded_input])
                            prediction_probs = st.session_state.model.predict_proba(input_array)[0]
                            predicted_class = np.argmax(prediction_probs)
                            predicted_outcome = st.session_state.le.inverse_transform([predicted_class])[0]
                            confidence = np.max(prediction_probs) * 100
                            st.session_state.prediction = predicted_outcome
                            st.session_state.confidence = confidence
                        else:
                            st.session_state.prediction = None
                            st.session_state.confidence = 0.0
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")

def render_prediction():
    """Render prediction details."""
    with st.expander("Prediction", expanded=True):
        try:
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
            if len(valid_sequence) < SEQUENCE_LENGTH:
                st.markdown(
                    f"<div style='background-color: #edf2f7; padding: 15px; border-radius: 8px;'>"
                    f"<p style='font-size:1.2rem; font-weight:bold; margin:0; color:#2d3748;'>"
                    f"Need {SEQUENCE_LENGTH - len(valid_sequence)} more Player or Banker results</p></div>",
                    unsafe_allow_html=True
                )
            elif st.session_state.prediction:
                prediction_color = '#3182ce' if st.session_state.prediction == 'P' else '#e53e3e' if st.session_state.prediction == 'B' else '#2d3748'
                st.markdown(
                    f"<div style='background-color: #edf2f7; padding: 15px; border-radius: 8px;'>"
                    f"<p style='font-size:1.2rem; font-weight:bold; margin:0; color:{prediction_color};'>"
                    f"Predicted Outcome: {st.session_state.prediction}</p>"
                    f"<p style='font-size:1rem; margin:5px 0;'>"
                    f"Confidence: {st.session_state.confidence:.1f}%</p></div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No prediction available.")
        except Exception as e:
            st.error(f"Error rendering prediction: {str(e)}")

def render_status():
    """Render session status."""
    with st.expander("Session Status", expanded=True):
        try:
            st.markdown(f"**Hands Played**: {len(st.session_state.sequence)}")
            valid_sequence = [r for r in st.session_state.sequence if r in ['P', 'B']]
            streak_info = "No streak detected"
            if len(valid_sequence) >= 4 and len(set(valid_sequence[-4:])) == 1:
                streak_length = len([x for x in valid_sequence[::-1] if x == valid_sequence[-1]])
                streak_info = f"Streak detected: {valid_sequence[-1]} x {streak_length}"
            st.markdown(f"**Streak Status**: {streak_info}")
            st.markdown(f"**Transition Counts**: PP: {st.session_state.transition_counts['PP']}, "
                        f"PB: {st.session_state.transition_counts['PB']}, "
                        f"BP: {st.session_state.transition_counts['BP']}, "
                        f"BB: {st.session_state.transition_counts['BB']}")
        except Exception as e:
            st.error(f"Error rendering status: {str(e)}")

def render_bead_plate():
    """Render the bead plate visualization of game results."""
    with st.expander("Bead Plate"):
        try:
            if not st.session_state.sequence:
                st.info("No results yet. Enter results to see the bead plate.")
                return

            rows = 6
            results = st.session_state.sequence
            num_results = len(results)
            cols = (num_results + rows - 1) // rows

            grid = [['' for _ in range(cols)] for _ in range(rows)]
            for i, result in enumerate(results):
                row = i % rows
                col = i // rows
                grid[row][col] = result

            html = """
            <style>
                .bead-plate-table {
                    border-collapse: collapse;
                    margin: 10px 0;
                }
                .bead-plate-table td {
                    width: 30px;
                    height: 30px;
                    text-align: center;
                    vertical-align: middle;
                    border: 1px solid #ccc;
                    font-weight: bold;
                    font-size: 14px;
                }
                .player {
                    background-color: #3182ce;
                    color: white;
                    border-radius: 50%;
                }
                .banker {
                    background-color: #e53e3e;
                    color: white;
                    border-radius: 50%;
                }
                .tie {
                    background-color: #38a169;
                    color: white;
                    border-radius: 50%;
                }
                .empty {
                    background-color: #f7fafc;
                }
            </style>
            <table class='bead-plate-table'>
            """

            for row in range(rows):
                html += "<tr>"
                for col in range(cols):
                    result = grid[row][col]
                    if result == 'P':
                        html += "<td class='player'>P</td>"
                    elif result == 'B':
                        html += "<td class='banker'>B</td>"
                    elif result == 'T':
                        html += "<td class='tie'>T</td>"
                    else:
                        html += "<td class='empty'></td>"
                html += "</tr>"
            html += "</table>"

            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error rendering bead plate: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Baccarat Prediction Assistant", layout="wide")
    st.title("Baccarat Prediction Assistant")
    initialize_session_state()
    render_session_setup()
    render_result_input()
    render_prediction()
    render_status()
    render_bead_plate()

if __name__ == "__main__":
    main()
