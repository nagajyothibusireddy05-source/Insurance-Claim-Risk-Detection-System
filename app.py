import streamlit as st
from common_app import auth_css, authenticate_user, is_authenticated, render_globe

st.set_page_config(page_title="Login | Insurance Claim Risk Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")
auth_css()

if is_authenticated():
    st.switch_page("pages/2_Overview.py")

left, right = st.columns([1.15, .85], gap="large")
with left:
    st.markdown('<div class="brand-pill">AI Insurance Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Insurance Claim<br>Risk Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Secure access to risk analytics, explainable machine learning and investigator intelligence.</div>', unsafe_allow_html=True)
    render_globe(500)
    st.markdown('<div class="small-note">Dynamic risk intelligence • Explainable AI • Human-in-the-loop investigation</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown("### 🔐 Welcome Back")
    st.write("Sign in to continue to your dashboard.")
    with st.form("login_form"):
        username=st.text_input("Username", placeholder="Enter your username")
        password=st.text_input("Password", type="password", placeholder="Enter your password")
        submitted=st.form_submit_button("Sign In →", use_container_width=True)
    if submitted:
        if authenticate_user(username,password):
            st.session_state["authenticated"]=True
            st.session_state["username"]=username.strip()
            st.switch_page("pages/2_Overview.py")
        else:
            st.error("Invalid username or password.")
    st.divider()
    st.write("Don't have an account?")
    st.page_link("pages/1_Register.py", label="✨ Create a new account", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
