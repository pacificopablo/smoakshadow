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

# Custom CSS for a professional design
st.markdown(f"""
    <style>
    body {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.75);
        padding: 40px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        max-width: 450px;
        margin: 50px auto;
        color: #f0f0f0;
        text-align: center;
    }}
    .welcome-header {{
        font-size: 2em;
        font-weight: 600;
        color: #d4af37; /* Gold tone for elegance */
        margin-bottom: 15px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }}
    .welcome-text {{
        font-size: 1.1em;
        color: #dcdcdc;
        margin-bottom: 25px;
    }}
    .stButton>button {{
        font-size: 1.2em;
        padding: 12px 30px;
        border-radius: 8px;
        background-color: #2e7d32; /* Deep green for professionalism */
        color: white;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}
    .stButton>button:hover {{
        background-color: #1b5e20;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }}
    </style>
""", unsafe_allow_html=True)

# Welcome message with enhanced styling
st.markdown('<div class="welcome-header">🃏 Welcome to Baccarat Fund Club</div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-text">Click the <strong>Enter</strong> button to access your dashboard</div>', unsafe_allow_html=True)

# Enter button with loading spinner
if st.button("🎲 Enter"):
    with st.spinner("Redirecting to your dashboard..."):
        time.sleep(2)  # Simulated loading
        components.html(f"""
            <script>
                window.location.href = "{redirect_url}";
            </script>
        """, height=0)

# Optional footer for professionalism
st.markdown('<div style="text-align: center; font-size: 0.8em; color: #bbb; margin-top: 20px;">© 2025 Baccarat Fund Club. All rights reserved.</div>', unsafe_allow_html=True)
