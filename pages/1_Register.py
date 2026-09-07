import streamlit as st
from common_app import auth_css, register_user, render_globe

st.set_page_config(page_title="Register | Insurance Claim Risk Detection", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")
auth_css()

left,right=st.columns([.9,1.1],gap="large")
with left:
    st.markdown('<div class="brand-pill">Create Your Secure Workspace</div>',unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Join the Risk<br>Intelligence Hub</div>',unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Create an investigator account and access the claim risk analytics platform.</div>',unsafe_allow_html=True)
    render_globe(470)
with right:
    st.markdown('<div class="auth-card">',unsafe_allow_html=True)
    st.markdown("### ✨ Create Account")
    st.write("Create a local prototype account for your project demo.")
    with st.form("register_form"):
        username=st.text_input("Username",placeholder="Choose a username")
        email=st.text_input("Email",placeholder="you@example.com")
        password=st.text_input("Password",type="password",placeholder="Minimum 6 characters")
        confirm=st.text_input("Confirm Password",type="password",placeholder="Re-enter password")
        submitted=st.form_submit_button("Create Account →",use_container_width=True)
    if submitted:
        if password!=confirm:
            st.error("Passwords do not match.")
        else:
            ok,msg=register_user(username,email,password)
            if ok:
                st.success(msg)
                st.page_link("app.py",label="🔐 Go to Login",use_container_width=True)
            else:
                st.error(msg)
    st.divider(); st.write("Already have an account?")
    st.page_link("app.py",label="← Back to Login",use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)
