import streamlit as st
import streamlit.components.v1 as components
import time

# Target redirect URL
redirect_url = "https://pacificopablo.github.io/BFC-TRACKER/"

# Page setup
st.set_page_config(
    page_title="Welcome to Baccarat Fund Club",
    page_icon="🃏",
    layout="centered"
)

# Baccarat background image (royalty-free)
background_image_url = "https://cdn.pixabay.com/photo/2017/01/18/19/19/roulette-1992501_1280.jpg"

# Custom CSS for full-page background and styling
st.markdown(f"""
    <style>
    body {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.65);
        padding-top: 200px;
        padding-bottom: 200px;
        color: white;
        text-align: center;
    }}
    .stButton>button {{
        font-size: 1.2em;
        padding: 0.6em 2.5em;
        border-radius: 10px;
        background-color: #00b894;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #019d79;
        transform: scale(1.05);
    }}
    </style>
""", unsafe_allow_html=True)

# Welcome message
st.markdown("## 🃏 Welcome to Baccarat Fund Club")
st.markdown("### Click the **Enter** button to go to your dashboard")

# Enter button with loading spinner
if st.button("🎲 Enter"):
    with st.spinner("Loading... Redirecting to your dashboard..."):
        time.sleep(2)  # Simulated loading
        components.html(f"""
            <script>
                window.location.href = "{redirect_url}";
            </script>
        """, height=0)
