
import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import (
    clean_data,
    engineer_features,
    train_compare,
    risk_category,
    TARGET,
    top_contributors,
)


@st.cache_data(show_spinner=False)
def load_default():
    return pd.read_csv("insurance_claims_synthetic.csv")


@st.cache_resource(show_spinner=False)
def train_cached(df_hashable: str, csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    cleaned, info = clean_data(df)
    featured = engineer_features(cleaned)
    comparison, fitted, test = train_compare(featured)
    return cleaned, featured, info, comparison, fitted, test


def get_project_data():
    uploaded = st.session_state.get("_claims_uploaded_file")

    if uploaded is None:
        raw = load_default()
        with open("insurance_claims_synthetic.csv", "rb") as f:
            csv_bytes = f.read()
    else:
        raw = pd.read_csv(uploaded)
        csv_bytes = uploaded

    cleaned, featured, info, comparison, fitted, test = train_cached(
        str(hash(csv_bytes)), csv_bytes
    )

    return raw, cleaned, featured, info, comparison, fitted, test


def setup_css():
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 38px;
                font-weight: 800;
                margin-bottom: 2px;
            }

            .subtitle {
                color: #6b7280;
                font-size: 16px;
                margin-bottom: 18px;
            }

            .risk-high {
                background: #fee2e2;
                color: #b91c1c;
                padding: 8px 14px;
                border-radius: 999px;
                font-weight: 700;
                text-align: center;
            }

            .risk-medium {
                background: #fef3c7;
                color: #b45309;
                padding: 8px 14px;
                border-radius: 999px;
                font-weight: 700;
                text-align: center;
            }

            .risk-low {
                background: #dcfce7;
                color: #15803d;
                padding: 8px 14px;
                border-radius: 999px;
                font-weight: 700;
                text-align: center;
            }

            div[data-testid="stMetric"] {
                border: 1px solid #e5e7eb;
                padding: 12px;
                border-radius: 12px;
                background: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls(comparison):
    model_names = comparison["Model"].tolist()

    st.sidebar.header("⚙️ Controls")
    if is_authenticated():
        st.sidebar.caption(f"Signed in as **{st.session_state.get('username', 'User')}**")

    selected_model = st.sidebar.selectbox(
        "🤖 Select Algorithm",
        ["All Models"] + model_names,
        key="selected_algorithm",
    )

    comparison_metric = st.sidebar.selectbox(
        "📊 Rank / Compare By",
        ["CV PR-AUC", "PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"],
        key="comparison_metric",
    )

    risk_filter = st.sidebar.multiselect(
        "🎯 Risk Category",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
        key="risk_filter",
    )

    st.sidebar.divider()

    st.sidebar.subheader("🔗 Pages")

    st.sidebar.page_link("pages/2_Overview.py", label="📋 Overview & Content")
    st.sidebar.page_link("pages/3_Graphs.py", label="📊 All Graphs")
    st.sidebar.page_link("pages/4_Tables.py", label="📑 All Tables")

    st.sidebar.divider()

    if is_authenticated() and st.sidebar.button("🚪 Logout", use_container_width=True):
        logout_user()
        st.switch_page("app.py")

    st.sidebar.divider()

    st.sidebar.caption("Available Algorithms")
    for name in model_names:
        st.sidebar.caption(f"• {name}")

    return selected_model, comparison_metric, risk_filter


def active_model_data(selected_model, comparison, fitted):
    if selected_model == "All Models":
        active_model_name = comparison.iloc[0]["Model"]
    else:
        active_model_name = selected_model

    active_pipe = fitted[active_model_name]["pipeline"]
    active_metrics = fitted[active_model_name]["metrics"]

    return active_model_name, active_pipe, active_metrics


def score_claims(cleaned, featured, active_pipe, risk_filter):
    X_all = featured.drop(columns=[TARGET])
    all_probs = active_pipe.predict_proba(X_all)[:, 1]

    scored_all = cleaned.copy()
    scored_all["Fraud Risk Probability"] = all_probs
    scored_all["Predicted Risk"] = [risk_category(p) for p in all_probs]
    scored_all["Days to submit"] = featured["Days to submit"].values
    scored_all["Claim to repair ratio"] = featured["Claim to repair ratio"].values
    scored_all["Invoice variance pct"] = featured["Invoice variance pct"].values
    scored_all["Missing document count"] = featured["Missing document count"].values

    scored = scored_all[
        scored_all["Predicted Risk"].isin(risk_filter)
    ].copy()

    return scored_all, scored, all_probs

# ============================================================
# AUTHENTICATION + LOGIN UI HELPERS
# ============================================================
import hashlib
import json
import secrets
from pathlib import Path

USERS_FILE = Path("users.json")


def _load_users():
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    ).hex()
    return salt, digest


def register_user(username, email, password):
    users = _load_users()
    key = username.strip().lower()
    email_key = email.strip().lower()
    if not key or not email_key or not password:
        return False, "Please fill in all fields."
    if len(key) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 6:
        return False, "Password must contain at least 6 characters."
    if key in users:
        return False, "Username already exists."
    if any(v.get("email", "").lower() == email_key for v in users.values()):
        return False, "An account with this email already exists."
    salt, password_hash = _hash_password(password)
    users[key] = {"username": username.strip(), "email": email.strip(), "salt": salt, "password_hash": password_hash}
    _save_users(users)
    return True, "Registration successful. You can now log in."


def authenticate_user(username, password):
    record = _load_users().get(username.strip().lower())
    if not record:
        return False
    _, digest = _hash_password(password, record["salt"])
    return secrets.compare_digest(digest, record["password_hash"])


def logout_user():
    st.session_state["authenticated"] = False
    st.session_state.pop("username", None)


def is_authenticated():
    return bool(st.session_state.get("authenticated", False))


def require_auth():
    if not is_authenticated():
        st.warning("Please log in to access this dashboard.")
        st.page_link("app.py", label="🔐 Go to Login")
        st.stop()


def render_globe(height=500):
    import streamlit.components.v1 as components
    html = r'''<!doctype html><html><body style="margin:0;background:transparent;overflow:hidden">
    <canvas id="g" style="width:100%;height:100%;display:block"></canvas>
    <script>
    const c=document.getElementById('g'),x=c.getContext('2d'); let d=devicePixelRatio;
    function size(){let r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;x.setTransform(d,0,0,d,0,0)} size();addEventListener('resize',size);
    const pts=[],N=900; for(let i=0;i<N;i++){let y=1-2*i/(N-1),q=Math.sqrt(1-y*y),a=Math.PI*(3-Math.sqrt(5))*i;pts.push({x:Math.cos(a)*q,y,z:Math.sin(a)*q,p:Math.random()*6.28})}
    let rot=0,sc=0,target=0,last=performance.now();
    function draw(t){let dt=(t-last)/1000;last=t;rot+=dt*.6;if(Math.random()<dt*.08)target=target>.3?0:1;sc+=(target-sc)*Math.min(1,dt*1.4);
      let w=c.clientWidth,h=c.clientHeight,R=Math.min(w,h)*.35,cx=w/2,cy=h/2;x.clearRect(0,0,w,h);let out=[];
      for(const p of pts){let xx=p.x*Math.cos(rot)-p.z*Math.sin(rot),zz=p.x*Math.sin(rot)+p.z*Math.cos(rot),yy=p.y;let s=1+sc*(1.5+.35*Math.sin(p.p+t*.001));xx*=s;yy*=s;zz*=s;let k=1.12-zz*.28;out.push({sx:cx+xx*R/k,sy:cy+yy*R/k,z:zz})}
      out.sort((a,b)=>a.z-b.z);for(const q of out){let dep=(q.z+1.6)/3.2;x.beginPath();x.arc(q.sx,q.sy,.7+1.4*dep,0,6.283);x.fillStyle=`rgba(86,199,255,${.12+.8*dep})`;x.fill()}requestAnimationFrame(draw)}requestAnimationFrame(draw);
    </script></body></html>'''
    components.html(html, height=height, scrolling=False)


def auth_css():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{background:radial-gradient(circle at 10% 15%,#12284a 0,#071322 45%,#02060c 100%)}
    [data-testid="stHeader"]{background:transparent}
    [data-testid="stSidebar"]{background:#071322}
    .auth-title{color:#f7fbff;font-size:42px;font-weight:800;line-height:1.05;margin:8px 0}
    .auth-subtitle{color:#a9bad1;font-size:16px;line-height:1.55}
    .brand-pill{display:inline-block;color:#8edcff;border:1px solid rgba(142,220,255,.3);border-radius:999px;padding:7px 13px;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
    .auth-card{background:rgba(8,20,37,.86);border:1px solid rgba(142,220,255,.18);border-radius:22px;padding:30px;box-shadow:0 22px 70px rgba(0,0,0,.4)}
    .small-note{color:#8298b5;font-size:12px}
    </style>
    """, unsafe_allow_html=True)
