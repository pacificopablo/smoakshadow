import streamlit as st
import logging
import plotly.graph_objects as go
import math
import sqlite3
import google_auth_oauthlib.flow
import googleapiclient.discovery
from datetime import datetime, timedelta
import jwt
import extra_streamlit_components as stx
from authlib.integrations.requests_client import OAuth2Session
import os

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY_KEY,
            bankroll REAL DEFAULT 1000.0,
            base_bet REAL DEFAULT 10.0,
            money_management_strategy TEXT DEFAULT "Flat Betting",
            ai_mode TEXT DEFAULT "Conservative",
            history TEXT DEFAULT "[]"
        )
    ''')
    conn.commit()
    conn.close()

def save_user(email, bankroll=None, base_bet=None, strategy=None, ai_mode=None, history=None):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO users (email, bankroll, base_bet, money_management_strategy, ai_mode, history)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        email,
        bankroll if bankroll is not None else 1000.0,
        base_bet if base_bet is not None else 10.0,
        strategy if strategy is not None else "Flat Betting",
        ai_mode if ai_mode is not None else "Conservative",
        str(history) if history is not None else "[]"
    ))
    conn.commit()
    conn.close()

def load_user(email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT bankroll, base_bet, money_management_strategy, ai_mode, history FROM users WHERE email = ?', (email,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {
            'bankroll': result[0],
            'base_bet': result[1],
            'money_management_strategy': result[2],
            'ai_mode': result[3],
            'history': eval(result[4])  # Convert string back to list
        }
    return None

# Authentication setup
class AuthTokenManager:
    def __init__(self, cookie_name, token_key):
        self.cookie_manager = stx.CookieManager()
        self.cookie_name = cookie_name
        self.token_key = token_key

    def get_decoded_token(self):
        token = self.cookie_manager.get(self.cookie_name)
        if not token:
            return None
        try:
            decoded = jwt.decode(token, self.token_key, algorithms=["HS256"])
            return decoded
        except jwt.ExpiredSignatureError:
            st.toast(":red[Session expired, please log in again]")
            self.delete_token()
            return None
        except jwt.InvalidTokenError:
            return None

    def set_token(self, email):
        exp_date = (datetime.now() + timedelta(days=7)).timestamp()
        token = jwt.encode(
            {"email": email, "exp": exp_date},
            self.token_key,
            algorithm="HS256"
        )
        self.cookie_manager.set(
            self.cookie_name,
            token,
            expires_at=datetime.fromtimestamp(exp_date)
        )

    def delete_token(self):
        try:
            self.cookie_manager.delete(self.cookie_name)
        except KeyError:
            pass

class Authenticator:
    def __init__(self, secret_path, redirect_uri, token_key):
        self.secret_path = secret_path
        self.redirect_uri = redirect_uri
        self.auth_token_manager = AuthTokenManager("auth_jwt", token_key)

    def _initialize_flow(self):
        return google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            self.secret_path,
            scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
            redirect_uri=self.redirect_uri
        )

    def get_auth_url(self):
        flow = self._initialize_flow()
        auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true")
        return auth_url

    def login(self):
        if not st.session_state.get("connected", False):
            auth_url = self.get_auth_url()
            st.link_button("Login with Google", auth_url)

    def check_auth(self):
        if st.session_state.get("connected", False):
            st.toast(":green[User is authenticated]")
            return True
        if st.session_state.get("logout"):
            st.toast(":green[User logged out]")
            return False

        token = self.auth_token_manager.get_decoded_token()
        if token:
            st.session_state["connected"] = True
            st.session_state["user_info"] = {"email": token["email"]}
            st.rerun()

        auth_code = st.query_params.get("code")
        if auth_code:
            flow = self._initialize_flow()
            flow.fetch_token(code=auth_code)
            creds = flow.credentials
            oauth_service = googleapiclient.discovery.build("oauth2", "v2", credentials=creds)
            user_info = oauth_service.userinfo().get().execute()
            email = user_info.get("email")
            if email:
                self.auth_token_manager.set_token(email)
                st.session_state["connected"] = True
                st.session_state["user_info"] = {"email": email, "name": user_info.get("name", "")}
                st.query_params.clear()
                save_user(email)  # Initialize user in DB if new
                st.rerun()
        return False

    def logout(self):
        st.session_state["logout"] = True
        st.session_state["user_info"] = None
        st.session_state["connected"] = False
        self.auth_token_manager.delete_token()

def auth_page():
    st.set_page_config(page_title="Mang Baccarat Predictor - Authentication", page_icon="🔒", layout="centered")
    st.title("Mang Baccarat Predictor - Authentication")
    st.markdown("""
        <style>
        .auth-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid #e1e1e1;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .stButton > button {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            background-color: #4285F4;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .stButton > button:hover {
            opacity: 0.9;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        authenticator = Authenticator(
            secret_path="client_secret.json",
            redirect_uri=st.secrets["auth"]["redirect_uri"],
            token_key=st.secrets["auth"]["token_key"]
        )
        authenticator.login()
        authenticator.check_auth()
        st.markdown('</div>', unsafe_allow_html=True)

# [Rest of the original functions remain unchanged: normalize, detect_streak, is_alternating, is_zigzag, recent_trend, frequency_count, build_big_road, build_big_eye_boy, build_cockroach_pig, advanced_bet_selection, money_management, calculate_bankroll, calculate_win_loss_tracker]

def main():
    init_db()
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'logout' not in st.session_state:
        st.session_state.logout = False

    authenticator = Authenticator(
        secret_path="client_secret.json",
        redirect_uri=st.secrets["auth"]["redirect_uri"],
        token_key=st.secrets["auth"]["token_key"]
    )

    if not authenticator.check_auth():
        auth_page()
        return

    try:
        st.set_page_config(page_title="Mang Baccarat Predictor", page_icon="🎲", layout="wide")
        email = st.session_state.user_info["email"]
        user_data = load_user(email)
        if not user_data:
            save_user(email)
            user_data = load_user(email)

        st.title(f"Mang Baccarat Predictor - Welcome, {st.session_state.user_info.get('name', email)}")

        if 'history' not in st.session_state:
            st.session_state.history = user_data['history']
        if 'initial_bankroll' not in st.session_state:
            st.session_state.initial_bankroll = user_data['bankroll']
        if 'base_bet' not in st.session_state:
            st.session_state.base_bet = user_data['base_bet']
        if 'money_management_strategy' not in st.session_state:
            st.session_state.money_management_strategy = user_data['money_management_strategy']
        if 'ai_mode' not in st.session_state:
            st.session_state.ai_mode = user_data['ai_mode']
        if 'selected_patterns' not in st.session_state:
            st.session_state.selected_patterns = ["Bead Bin", "Win/Loss"]
        if 't3_level' not in st.session_state:
            st.session_state.t3_level = 1
        if 't3_results' not in st.session_state:
            st.session_state.t3_results = []
        if 'screen_width' not in st.session_state:
            st.session_state.screen_width = 1024

        # [Rest of the original main() function remains unchanged, including CSS, game settings, input buttons, patterns, prediction, bankroll progress]

        with st.expander("Reset", expanded=False):
            cols = st.columns(2)
            with cols[0]:
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
                    save_user(
                        email,
                        bankroll=st.session_state.initial_bankroll,
                        base_bet=st.session_state.base_bet,
                        strategy=st.session_state.money_management_strategy,
                        ai_mode=st.session_state.ai_mode,
                        history=st.session_state.history
                    )
                    st.rerun()
            with cols[1]:
                if st.button("Logout"):
                    authenticator.logout()
                    save_user(
                        email,
                        bankroll=st.session_state.initial_bankroll,
                        base_bet=st.session_state.base_bet,
                        strategy=st.session_state.money_management_strategy,
                        ai_mode=st.session_state.ai_mode,
                        history=st.session_state.history
                    )
                    st.rerun()

    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        st.error(f"Unexpected error: {str(e)}. Contact support if this persists.")

if __name__ == "__main__":
    main()
