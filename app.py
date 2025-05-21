import streamlit as st
import uuid

def initialize_session_state():
    """Initialize session state with default values."""
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
        st.session_state.transition_counts = {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0}
        st.session_state.session_id = str(uuid.uuid4())

def reset_session():
    """Reset session state to initial values."""
    st.session_state.update({
        'sequence': [],
        'transition_counts': {'PP': 0, 'PB': 0, 'BP': 0, 'BB': 0},
        'session_id': str(uuid.uuid4())
    })

def place_result(result):
    """Append a game result and update transition counts."""
    try:
        st.session_state.sequence.append(result)
        if len(st.session_state.sequence) >= 2:
            prev_result = st.session_state.sequence[-2]
            if result in ['P', 'B'] and prev_result in ['P', 'B']:
                transition = f"{prev_result}{result}"
                st.session_state.transition_counts[transition] += 1
    except Exception as e:
        st.error(f"Error processing result: {e}")

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
            st.error(f"Error setting up session: {e}")

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
        except Exception as e:
            st.error(f"Error processing input: {e}")

def render_status():
    """Render session status with hands played, streak status, and transition counts."""
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
            st.error(f"Error rendering status: {e}")

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
            st.error(f"Error rendering bead plate: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Baccarat Result Tracker", layout="wide")
    st.title("Baccarat Result Tracker")
    initialize_session_state()
    render_session_setup()
    render_result_input()
    render_status()
    render_bead_plate()

if __name__ == "__main__":
    main()
