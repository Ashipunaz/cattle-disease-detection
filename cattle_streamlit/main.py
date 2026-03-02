import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import warnings
import tempfile
import json
import os
import io

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES  = ['fmd', 'healthy', 'lumpy skin', 'mastitis']
IMG_SIZE     = (224, 224)
MAX_IMAGES   = 5
ARCH_PATH    = 'cattle_final_archi.json'
WEIGHTS_PATH = 'cattle_final.weights.h5'

# ─────────────────────────────────────────────────────────────
# DISEASE INFO (unchanged, but we may reference later)
# ─────────────────────────────────────────────────────────────
DISEASE_INFO = {
    'fmd': {
        'full_name':      'Foot and Mouth Disease',
        'short':          'FMD',
        'emoji':          '🟠',
        'severity':       'Urgent',
        'severity_color': '#c0392b',
        'severity_bg':    'rgba(192,57,43,0.10)',
        'requires_vet':   True,
        'what_you_see':   'Blisters on the mouth, tongue, and hooves. The animal may be drooling heavily or struggling to walk.',
        'what_to_do':     'Separate this animal from the rest of the herd right away. Do not move animals off the farm. Call your vet or the nearest livestock office today - FMD spreads very fast.',
        'urgency_msg':    'Act today. Every hour matters.',
    },
    'healthy': {
        'full_name':      'No Disease Detected',
        'short':          'Healthy',
        'emoji':          '🟢',
        'severity':       'All Clear',
        'severity_color': '#27ae60',
        'severity_bg':    'rgba(39,174,96,0.10)',
        'requires_vet':   False,
        'what_you_see':   'The animal shows no visible signs of disease.',
        'what_to_do':     'Your animal looks healthy. Keep up regular check-ups, ensure clean water and feed, and stay on your vaccination schedule to keep it that way.',
        'urgency_msg':    'Continue routine care.',
    },
    'lumpy skin': {
        'full_name':      'Lumpy Skin Disease',
        'short':          'LSD',
        'emoji':          '🔴',
        'severity':       'Urgent',
        'severity_color': '#c0392b',
        'severity_bg':    'rgba(192,57,43,0.10)',
        'requires_vet':   True,
        'what_you_see':   'Round raised lumps or nodules appearing across the skin. The animal may have a fever and reduced milk output.',
        'what_to_do':     'Separate the animal from the herd immediately. LSD spreads through insect bites - treat the whole herd with insect repellent. Contact your vet to vaccinate the animals at risk.',
        'urgency_msg':    'Isolate today. Protect the herd.',
    },
    'mastitis': {
        'full_name':      'Mastitis',
        'short':          'Mastitis',
        'emoji':          '🟡',
        'severity':       'Needs Attention',
        'severity_color': '#d68910',
        'severity_bg':    'rgba(214,137,16,0.10)',
        'requires_vet':   True,
        'what_you_see':   'The udder looks swollen or feels warm and painful. Milk may appear watery, lumpy, or discoloured.',
        'what_to_do':     'Contact your vet for antibiotic treatment. Milk the affected quarters separately and discard that milk. Wash hands and equipment between animals to stop it spreading.',
        'urgency_msg':    'Book a vet visit soon.',
    },
}

# ─────────────────────────────────────────────────────────────
# PDF HELPER (unchanged except rounding)
# ─────────────────────────────────────────────────────────────
def safe(text):
    return (str(text)
        .replace('\u2014', '-').replace('\u2013', '-')
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2192', '->').replace('\u2022', '-')
    )

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cattle Health Check",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# THEME MANAGEMENT
# ─────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'   # default

def set_theme(theme_name):
    st.session_state.theme = theme_name

# ─────────────────────────────────────────────────────────────
# CSS — two themes: dark (default) and light (earthy light)
# ─────────────────────────────────────────────────────────────
def get_css(theme):
    if theme == 'light':
        # Light mode – soft cream background, dark text
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700;900&family=Inter:wght@300;400;500;600&display=swap');

        :root {
            --bg:       #faf7f2;
            --bg2:      #f5efe8;
            --bg3:      #ebe3d9;
            --bg4:      #e0d5c8;
            --bg5:      #d2c4b4;
            --amber:    #b86d1c;
            --amber2:   #a85e12;
            --amber3:   #8f4e0f;
            --amber4:   #6b3c0c;
            --cream:    #2e2a25;
            --cream2:   #3a332c;
            --cream3:   #4d4238;
            --cream4:   #635548;
            --cream5:   #7b6b5b;
            --red:      #c0392b;
            --red-bg:   rgba(192,57,43,0.08);
            --gold:     #d68910;
            --gold-bg:  rgba(214,137,16,0.08);
            --green:    #27ae60;
            --green-bg: rgba(39,174,96,0.08);
            --border:   rgba(100,70,40,0.20);
            --border2:  rgba(100,70,40,0.35);
        }

        * { font-family: 'Inter', sans-serif; box-sizing: border-box; }
        h1,h2,h3,h4 { font-family: 'Fraunces', serif !important; }

        .stApp { background-color: var(--bg) !important; }
        .block-container { padding: 2.5rem 3rem 5rem !important; }
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }

        /* ── SIDEBAR ── */
        [data-testid="stSidebar"] {
            background: var(--bg2) !important;
            border-right: 1px solid var(--border2) !important;
        }
        [data-testid="stSidebar"] > div { padding: 0 !important; }

        [data-testid="stSidebar"] .stRadio > label { display: none !important; }
        [data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 0 !important; }
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child { display: none !important; }
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
            display: flex !important;
            align-items: center !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0.65rem 1.4rem !important;
            margin: 0 !important;
            cursor: pointer !important;
            color: var(--cream4) !important;
            font-size: 0.88rem !important;
            font-weight: 400 !important;
            width: 100% !important;
            border-left: 3px solid transparent !important;
            transition: all 0.15s ease !important;
        }
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
            background: var(--bg3) !important;
            color: var(--cream) !important;
            border-left-color: var(--amber) !important;
        }
        [data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
            background: var(--bg4) !important;
            color: var(--cream) !important;
            border-left: 3px solid var(--amber) !important;
            font-weight: 600 !important;
        }

        .sb-brand {
            padding: 1.6rem 1.4rem 1.2rem;
            border-bottom: 1px solid var(--border);
        }
        .sb-icon { font-size: 1.8rem; margin-bottom: 0.5rem; display: block; }
        .sb-name {
            font-family: 'Fraunces', serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--cream);
            line-height: 1.3;
            margin-bottom: 0.3rem;
        }
        .sb-tagline {
            font-size: 0.76rem;
            color: var(--cream5);
            line-height: 1.5;
            font-weight: 300;
        }
        .sb-nav-label {
            font-size: 0.58rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            padding: 1.1rem 1.4rem 0.35rem;
        }
        .sb-tip {
            margin: 1rem 1rem 1.5rem;
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            font-size: 0.74rem;
            color: var(--cream4);
            line-height: 1.6;
        }
        .sb-tip strong { color: var(--amber); font-weight: 600; }

        /* ── PAGE HEADER ── */
        .pg-hd {
            padding-bottom: 1.2rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }
        .pg-title {
            font-family: 'Fraunces', serif;
            font-size: 2rem;
            font-weight: 700;
            color: var(--cream);
            margin: 0 0 0.2rem;
            line-height: 1.1;
        }
        .pg-sub { font-size: 0.88rem; color: var(--cream5); font-weight: 300; }

        /* ── STEPS ── */
        .step-no {
            font-size: 0.58rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 0.2rem;
        }
        .step-title {
            font-family: 'Fraunces', serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--cream);
            margin-bottom: 0.9rem;
        }

        /* ── UPLOAD ── */
        div[data-testid="stFileUploaderDropzone"] {
            background: var(--bg2) !important;
            border: 2px dashed var(--border2) !important;
            border-radius: 12px !important;
            color: var(--cream3) !important;
        }
        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--amber) !important;
        }
        .upload-hint {
            border: 2px dashed var(--border2);
            border-radius: 12px;
            padding: 2.5rem 2rem;
            text-align: center;
            background: var(--bg2);
        }

        /* ── RESULT CARD ── */
        .res-card {
            background: var(--bg2);
            border: 1px solid var(--border2);
            border-radius: 16px;
            padding: 1.6rem;
        }
        .res-verdict {
            font-family: 'Fraunces', serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--cream);
            margin-bottom: 0.3rem;
            line-height: 1.2;
        }
        .res-conf {
            font-size: 0.8rem;
            color: var(--cream5);
            margin-bottom: 1rem;
        }
        .urgency-bar {
            border-radius: 8px;
            padding: 0.65rem 1rem;
            margin-bottom: 1.1rem;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .section-lbl {
            font-size: 0.6rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.13em;
            margin-bottom: 0.3rem;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }
        .what-you-see {
            font-size: 0.84rem;
            color: var(--cream3);
            font-style: italic;
            line-height: 1.65;
            margin-bottom: 1rem;
            padding: 0.7rem 0.9rem;
            border-left: 2px solid var(--border2);
            border-radius: 0 6px 6px 0;
            background: var(--bg3);
        }
        .action-block {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.95rem 1.1rem;
            font-size: 0.86rem;
            color: var(--cream2);
            line-height: 1.7;
            margin-bottom: 1rem;
        }
        .m-row { display: flex; gap: 8px; margin-top: 0.9rem; }
        .m-box {
            flex: 1;
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem 0.4rem;
            text-align: center;
        }
        .m-val {
            font-family: 'Fraunces', serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--amber);
        }
        .m-lbl {
            font-size: 0.58rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 3px;
        }

        /* ── PROB BARS ── */
        .p-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--cream3);
            margin-bottom: 3px;
        }
        .p-head span:last-child { color: var(--cream); font-weight: 500; }
        .stProgress > div > div { background: var(--amber) !important; }

        /* ── HOME ── */
        .hero {
            background: var(--bg2);
            border: 1px solid var(--border2);
            border-radius: 20px;
            padding: 3rem 2.8rem;
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: '🐄';
            position: absolute;
            right: 2.5rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 7rem;
            opacity: 0.06;
            pointer-events: none;
        }
        .hero-eyebrow {
            font-size: 0.7rem;
            color: var(--amber);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 600;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-family: 'Fraunces', serif;
            font-size: 2.6rem;
            font-weight: 900;
            color: var(--cream);
            line-height: 1.05;
            margin-bottom: 1rem;
        }
        .hero-title em {
            font-style: italic;
            color: var(--amber);
        }
        .hero-body {
            font-size: 0.95rem;
            color: var(--cream4);
            font-weight: 300;
            max-width: 500px;
            line-height: 1.75;
            margin-bottom: 1.8rem;
        }
        .hero-cta-note { font-size: 0.75rem; color: var(--cream5); margin-top: 0.8rem; }
        .trust-row { display: flex; gap: 1.5rem; flex-wrap: wrap; }
        .trust-item {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            color: var(--cream4);
        }
        .trust-dot { color: var(--amber); }

        .section-heading {
            font-family: 'Fraunces', serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--cream);
            margin-bottom: 1.1rem;
        }

        .tip-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.3rem;
            height: 100%;
        }
        .tip-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
        .tip-title {
            font-family: 'Fraunces', serif;
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--cream2);
            margin-bottom: 0.35rem;
        }
        .tip-body { font-size: 0.79rem; color: var(--cream5); line-height: 1.6; }

        .disease-ref-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.75rem;
        }
        .drc-header { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.5rem; }
        .drc-name {
            font-family: 'Fraunces', serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--cream2);
        }
        .drc-sev {
            font-size: 0.63rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 2px 9px;
            border-radius: 20px;
            margin-left: auto;
        }
        .drc-body { font-size: 0.79rem; color: var(--cream5); line-height: 1.55; }
        .drc-action {
            font-size: 0.79rem;
            color: var(--cream4);
            margin-top: 0.45rem;
            padding-top: 0.45rem;
            border-top: 1px solid var(--border);
            line-height: 1.5;
        }

        /* ── HISTORY ── */
        .hist-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.95rem 1.2rem;
            margin-bottom: 0.65rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .hist-main {
            font-family: 'Fraunces', serif;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--cream2);
        }
        .hist-meta { font-size: 0.72rem; color: var(--cream5); margin-top: 2px; line-height: 1.5; }

        /* ── ABOUT ── */
        .about-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.4rem;
            margin-bottom: 1.1rem;
        }
        .about-title {
            font-family: 'Fraunces', serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--cream2);
            margin-bottom: 0.75rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid var(--border);
        }
        .about-text { font-size: 0.86rem; color: var(--cream4); line-height: 1.75; }
        .ab-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
        .ab-table th {
            color: var(--cream5);
            text-align: left;
            padding: 7px 10px;
            font-size: 0.67rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 1px solid var(--border);
        }
        .ab-table td { padding: 8px 10px; color: var(--cream4); border-bottom: 1px solid var(--border); }
        .ab-table tr:last-child td { border-bottom: none; }

        /* ── BUTTONS ── */
        .stButton > button {
            background: var(--amber) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.65rem 1.5rem !important;
            transition: background 0.2s !important;
            letter-spacing: 0.01em !important;
        }
        .stButton > button:hover { background: var(--amber2) !important; }

        /* streamlit metric */
        [data-testid="stMetricValue"] { color: var(--amber) !important; font-family: 'Fraunces', serif !important; }
        [data-testid="stMetricLabel"] { color: var(--cream5) !important; font-size: 0.75rem !important; }

        /* warning banner */
        .stAlert { background: var(--bg3) !important; border-color: var(--border2) !important; }

        /* ── DISCLAIMER ── */
        .disclaimer {
            font-size: 0.72rem;
            color: var(--cream5);
            text-align: center;
            margin-top: 3rem;
            padding: 0.85rem 1.2rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg2);
            line-height: 1.65;
        }

        /* scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--bg5); border-radius: 3px; }
        </style>
        """
    else:  # dark theme (default)
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700;900&family=Inter:wght@300;400;500;600&display=swap');

        :root {
            --bg:       #1a1410;
            --bg2:      #221c16;
            --bg3:      #2a2219;
            --bg4:      #32291f;
            --bg5:      #3d3228;
            --amber:    #c8822a;
            --amber2:   #e09640;
            --amber3:   #f0b060;
            --amber4:   #f8cc88;
            --cream:    #f5ead8;
            --cream2:   #e8d8bc;
            --cream3:   #c8b898;
            --cream4:   #9a8870;
            --cream5:   #6a5c48;
            --red:      #c0392b;
            --red-bg:   rgba(192,57,43,0.12);
            --gold:     #d68910;
            --gold-bg:  rgba(214,137,16,0.12);
            --green:    #27ae60;
            --green-bg: rgba(39,174,96,0.12);
            --border:   rgba(200,130,42,0.20);
            --border2:  rgba(200,130,42,0.40);
        }

        * { font-family: 'Inter', sans-serif; box-sizing: border-box; }
        h1,h2,h3,h4 { font-family: 'Fraunces', serif !important; }

        .stApp { background-color: var(--bg) !important; }
        .block-container { padding: 2.5rem 3rem 5rem !important; }
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }

        /* ── SIDEBAR ── */
        [data-testid="stSidebar"] {
            background: var(--bg2) !important;
            border-right: 1px solid var(--border2) !important;
        }
        [data-testid="stSidebar"] > div { padding: 0 !important; }

        [data-testid="stSidebar"] .stRadio > label { display: none !important; }
        [data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 0 !important; }
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child { display: none !important; }
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
            display: flex !important;
            align-items: center !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0.65rem 1.4rem !important;
            margin: 0 !important;
            cursor: pointer !important;
            color: var(--cream3) !important;
            font-size: 0.88rem !important;
            font-weight: 400 !important;
            width: 100% !important;
            border-left: 3px solid transparent !important;
            transition: all 0.15s ease !important;
        }
        [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
            background: var(--bg3) !important;
            color: var(--cream) !important;
            border-left-color: var(--amber2) !important;
        }
        [data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
            background: var(--bg4) !important;
            color: var(--cream) !important;
            border-left: 3px solid var(--amber) !important;
            font-weight: 600 !important;
        }

        .sb-brand {
            padding: 1.6rem 1.4rem 1.2rem;
            border-bottom: 1px solid var(--border);
        }
        .sb-icon { font-size: 1.8rem; margin-bottom: 0.5rem; display: block; }
        .sb-name {
            font-family: 'Fraunces', serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--cream);
            line-height: 1.3;
            margin-bottom: 0.3rem;
        }
        .sb-tagline {
            font-size: 0.76rem;
            color: var(--cream5);
            line-height: 1.5;
            font-weight: 300;
        }
        .sb-nav-label {
            font-size: 0.58rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            padding: 1.1rem 1.4rem 0.35rem;
        }
        .sb-tip {
            margin: 1rem 1rem 1.5rem;
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            font-size: 0.74rem;
            color: var(--cream4);
            line-height: 1.6;
        }
        .sb-tip strong { color: var(--amber3); font-weight: 600; }

        /* ── PAGE HEADER ── */
        .pg-hd {
            padding-bottom: 1.2rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }
        .pg-title {
            font-family: 'Fraunces', serif;
            font-size: 2rem;
            font-weight: 700;
            color: var(--cream);
            margin: 0 0 0.2rem;
            line-height: 1.1;
        }
        .pg-sub { font-size: 0.88rem; color: var(--cream5); font-weight: 300; }

        /* ── STEPS ── */
        .step-no {
            font-size: 0.58rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 0.2rem;
        }
        .step-title {
            font-family: 'Fraunces', serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--cream);
            margin-bottom: 0.9rem;
        }

        /* ── UPLOAD ── */
        div[data-testid="stFileUploaderDropzone"] {
            background: var(--bg2) !important;
            border: 2px dashed var(--border2) !important;
            border-radius: 12px !important;
            color: var(--cream3) !important;
        }
        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--amber) !important;
        }
        .upload-hint {
            border: 2px dashed var(--border2);
            border-radius: 12px;
            padding: 2.5rem 2rem;
            text-align: center;
            background: var(--bg2);
        }

        /* ── RESULT CARD ── */
        .res-card {
            background: var(--bg2);
            border: 1px solid var(--border2);
            border-radius: 16px;
            padding: 1.6rem;
        }
        .res-verdict {
            font-family: 'Fraunces', serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--cream);
            margin-bottom: 0.3rem;
            line-height: 1.2;
        }
        .res-conf {
            font-size: 0.8rem;
            color: var(--cream5);
            margin-bottom: 1rem;
        }
        .urgency-bar {
            border-radius: 8px;
            padding: 0.65rem 1rem;
            margin-bottom: 1.1rem;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .section-lbl {
            font-size: 0.6rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.13em;
            margin-bottom: 0.3rem;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }
        .what-you-see {
            font-size: 0.84rem;
            color: var(--cream3);
            font-style: italic;
            line-height: 1.65;
            margin-bottom: 1rem;
            padding: 0.7rem 0.9rem;
            border-left: 2px solid var(--border2);
            border-radius: 0 6px 6px 0;
            background: var(--bg3);
        }
        .action-block {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.95rem 1.1rem;
            font-size: 0.86rem;
            color: var(--cream2);
            line-height: 1.7;
            margin-bottom: 1rem;
        }
        .m-row { display: flex; gap: 8px; margin-top: 0.9rem; }
        .m-box {
            flex: 1;
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem 0.4rem;
            text-align: center;
        }
        .m-val {
            font-family: 'Fraunces', serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--amber3);
        }
        .m-lbl {
            font-size: 0.58rem;
            color: var(--cream5);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 3px;
        }

        /* ── PROB BARS ── */
        .p-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--cream3);
            margin-bottom: 3px;
        }
        .p-head span:last-child { color: var(--cream); font-weight: 500; }
        .stProgress > div > div { background: var(--amber) !important; }

        /* ── HOME ── */
        .hero {
            background: var(--bg2);
            border: 1px solid var(--border2);
            border-radius: 20px;
            padding: 3rem 2.8rem;
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: '🐄';
            position: absolute;
            right: 2.5rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 7rem;
            opacity: 0.06;
            pointer-events: none;
        }
        .hero-eyebrow {
            font-size: 0.7rem;
            color: var(--amber);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 600;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-family: 'Fraunces', serif;
            font-size: 2.6rem;
            font-weight: 900;
            color: var(--cream);
            line-height: 1.05;
            margin-bottom: 1rem;
        }
        .hero-title em {
            font-style: italic;
            color: var(--amber3);
        }
        .hero-body {
            font-size: 0.95rem;
            color: var(--cream4);
            font-weight: 300;
            max-width: 500px;
            line-height: 1.75;
            margin-bottom: 1.8rem;
        }
        .hero-cta-note { font-size: 0.75rem; color: var(--cream5); margin-top: 0.8rem; }
        .trust-row { display: flex; gap: 1.5rem; flex-wrap: wrap; }
        .trust-item {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            color: var(--cream4);
        }
        .trust-dot { color: var(--amber); }

        .section-heading {
            font-family: 'Fraunces', serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--cream);
            margin-bottom: 1.1rem;
        }

        .tip-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.3rem;
            height: 100%;
        }
        .tip-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
        .tip-title {
            font-family: 'Fraunces', serif;
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--cream2);
            margin-bottom: 0.35rem;
        }
        .tip-body { font-size: 0.79rem; color: var(--cream5); line-height: 1.6; }

        .disease-ref-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.75rem;
        }
        .drc-header { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.5rem; }
        .drc-name {
            font-family: 'Fraunces', serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--cream2);
        }
        .drc-sev {
            font-size: 0.63rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 2px 9px;
            border-radius: 20px;
            margin-left: auto;
        }
        .drc-body { font-size: 0.79rem; color: var(--cream5); line-height: 1.55; }
        .drc-action {
            font-size: 0.79rem;
            color: var(--cream4);
            margin-top: 0.45rem;
            padding-top: 0.45rem;
            border-top: 1px solid var(--border);
            line-height: 1.5;
        }

        /* ── HISTORY ── */
        .hist-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.95rem 1.2rem;
            margin-bottom: 0.65rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .hist-main {
            font-family: 'Fraunces', serif;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--cream2);
        }
        .hist-meta { font-size: 0.72rem; color: var(--cream5); margin-top: 2px; line-height: 1.5; }

        /* ── ABOUT ── */
        .about-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.4rem;
            margin-bottom: 1.1rem;
        }
        .about-title {
            font-family: 'Fraunces', serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--cream2);
            margin-bottom: 0.75rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid var(--border);
        }
        .about-text { font-size: 0.86rem; color: var(--cream4); line-height: 1.75; }
        .ab-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
        .ab-table th {
            color: var(--cream5);
            text-align: left;
            padding: 7px 10px;
            font-size: 0.67rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 1px solid var(--border);
        }
        .ab-table td { padding: 8px 10px; color: var(--cream4); border-bottom: 1px solid var(--border); }
        .ab-table tr:last-child td { border-bottom: none; }

        /* ── BUTTONS ── */
        .stButton > button {
            background: var(--amber) !important;
            color: var(--bg) !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.65rem 1.5rem !important;
            transition: background 0.2s !important;
            letter-spacing: 0.01em !important;
        }
        .stButton > button:hover { background: var(--amber2) !important; }

        /* streamlit metric */
        [data-testid="stMetricValue"] { color: var(--amber3) !important; font-family: 'Fraunces', serif !important; }
        [data-testid="stMetricLabel"] { color: var(--cream5) !important; font-size: 0.75rem !important; }

        /* warning banner */
        .stAlert { background: var(--bg3) !important; border-color: var(--border2) !important; }

        /* ── DISCLAIMER ── */
        .disclaimer {
            font-size: 0.72rem;
            color: var(--cream5);
            text-align: center;
            margin-top: 3rem;
            padding: 0.85rem 1.2rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg2);
            line-height: 1.65;
        }

        /* scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--bg5); border-radius: 3px; }
        </style>
        """

# Inject the CSS based on current theme
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL (unchanged)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(ARCH_PATH, 'r') as f:
        config = json.load(f)
    m = tf.keras.Model.from_config(config)
    m.load_weights(WEIGHTS_PATH)
    m.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def predict(image: Image.Image, model):
    img   = image.convert('RGB').resize(IMG_SIZE)
    arr   = np.expand_dims(np.array(img, dtype=np.float32), 0)
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]) * 100, preds

# ─────────────────────────────────────────────────────────────
# PDF (unchanged except rounding)
# ─────────────────────────────────────────────────────────────
PAGE_W      = 210
PAGE_H      = 297
M           = 12          # margin
CONT_W      = PAGE_W - M * 2
IMG_W       = 68
COL2_X      = M + IMG_W + 6
COL2_W      = PAGE_W - COL2_X - M
FOOTER_Y    = PAGE_H - 11
MAX_Y       = PAGE_H - 20  # content guard


def _footer(pdf):
    pdf.set_xy(M, FOOTER_Y)
    pdf.set_font('Helvetica', 'I', 6)
    pdf.set_text_color(140, 120, 95)
    pdf.cell(CONT_W, 4,
        safe('AI-assisted only. Does NOT replace a veterinary diagnosis. '
             'Always consult a licensed vet.  |  Cattle Health Check Kenya'),
        align='C')


def _compress(pil_image, quality=55):
    img  = pil_image.convert('RGB')
    w, h = img.size
    px   = int(IMG_W * 3.7795)
    if w > px:
        img = img.resize((px, int(h * px / w)), Image.LANCZOS)
    h_mm = IMG_W * (img.size[1] / img.size[0])
    buf  = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, optimize=True)
    buf.seek(0)
    return buf, h_mm


def generate_pdf(results):
    pdf = FPDF(unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)   # we control all page breaks

    for rec_idx, r in enumerate(results):
        pdf.add_page()
        info = DISEASE_INFO[r['disease']]

        # ── Header bar ──────────────────────────────────────────
        pdf.set_fill_color(34, 28, 22)
        pdf.rect(0, 0, PAGE_W, 22, 'F')

        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(245, 234, 216)
        pdf.set_xy(M, 5)
        pdf.cell(CONT_W, 7, safe('Cattle Health Check  -  Disease Report'))

        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(154, 136, 112)
        pdf.set_xy(M, 14)
        pdf.cell(CONT_W, 5, safe(
            f'Date: {datetime.now().strftime("%d %B %Y  %I:%M %p")}  |  '
            f'Animal {rec_idx + 1} of {len(results)}  |  Photo: {r["filename"]}'
        ))

        y = 26  # content cursor starts here

        # ── Photo ────────────────────────────────────────────────
        img_buf, img_h_mm = _compress(r['image'])
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(img_buf.read())
            tmp_path = tmp.name
        try:
            pdf.image(tmp_path, x=M, y=y, w=IMG_W)
        except Exception:
            img_h_mm = 0
        finally:
            os.unlink(tmp_path)

        # ── Right column: verdict ────────────────────────────────
        ry = y

        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(245, 234, 216)
        pdf.set_xy(COL2_X, ry)
        pdf.multi_cell(COL2_W, 6.5, safe(info['full_name']))
        ry = pdf.get_y()

        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(154, 136, 112)
        pdf.set_xy(COL2_X, ry)
        pdf.cell(COL2_W, 5, safe(f'Confidence: {r["confidence"]:.0f}%'))  # whole number
        ry += 6

        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(200, 130, 42)
        pdf.set_xy(COL2_X, ry)
        pdf.cell(COL2_W, 5, safe(info['urgency_msg']))
        ry += 7

        for label, value in [
            ('Vet Required', 'Yes' if info['requires_vet'] else 'No'),
            ('Status',       info['severity']),
        ]:
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(154, 136, 112)
            pdf.set_xy(COL2_X, ry)
            pdf.cell(28, 5, safe(label + ':'), ln=False)
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(50, 40, 30)
            pdf.cell(COL2_W - 28, 5, safe(value))
            ry += 5.5

        # Advance y past whichever column is taller
        y = max(y + img_h_mm, ry) + 5

        # ── Divider ──────────────────────────────────────────────
        pdf.set_draw_color(200, 130, 42)
        pdf.line(M, y, PAGE_W - M, y)
        y += 4

        # ── What you are seeing ──────────────────────────────────
        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(154, 136, 112)
            pdf.set_xy(M, y)
            pdf.cell(CONT_W, 5, 'What You Are Seeing:')
            y += 5.5
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(70, 58, 44)
            pdf.set_xy(M, y)
            pdf.multi_cell(CONT_W, 4.5, safe(info['what_you_see']))
            y = pdf.get_y() + 3

        # ── What to do ───────────────────────────────────────────
        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(154, 136, 112)
            pdf.set_xy(M, y)
            pdf.cell(CONT_W, 5, 'What To Do:')
            y += 5.5
            pdf.set_fill_color(245, 240, 230)
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(50, 38, 25)
            pdf.set_xy(M, y)
            pdf.multi_cell(CONT_W, 4.5, safe(info['what_to_do']), fill=True)
            y = pdf.get_y() + 5

        # ── Analysis Breakdown ───────────────────────────────────
        LABEL_W = 78
        PCT_W   = 16
        BAR_X   = M + LABEL_W + PCT_W
        BAR_W   = CONT_W - LABEL_W - PCT_W   # 92mm

        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(154, 136, 112)
            pdf.set_xy(M, y)
            pdf.cell(CONT_W, 5, 'Analysis Breakdown:')
            y += 6

            for cname, prob in zip(CLASS_NAMES, r['all_preds']):
                if y >= MAX_Y:
                    break

                pct    = float(prob) * 100          # e.g. 99.8
                is_top = cname == r['disease']
                label  = DISEASE_INFO[cname]['full_name']
                fill_w = BAR_W * (pct / 100.0)     # scale 0-100 -> 0-BAR_W mm

                # Label
                pdf.set_font('Helvetica', 'B' if is_top else '', 7.5)
                pdf.set_text_color(40, 32, 22)
                pdf.set_xy(M, y)
                pdf.cell(LABEL_W, 5, safe(('> ' if is_top else '   ') + label), ln=False)

                # Percentage value — right-aligned in its column (whole number)
                pdf.set_font('Helvetica', 'B' if is_top else '', 7.5)
                pdf.set_text_color(200, 130, 42)
                pdf.cell(PCT_W, 5, f'{pct:.0f}%', ln=False)

                # Bar background (light)
                bar_y = y + 1.2
                pdf.set_fill_color(220, 200, 170)
                pdf.rect(BAR_X, bar_y, BAR_W, 2.8, 'F')

                # Bar fill (proportional to pct)
                if fill_w > 0.1:
                    if is_top:
                        pdf.set_fill_color(180, 100, 25)    # amber dark for winner
                    else:
                        pdf.set_fill_color(195, 160, 110)   # muted for others
                    pdf.rect(BAR_X, bar_y, fill_w, 2.8, 'F')

                y += 6.5

        # ── Footer — absolute, never overflows ───────────────────
        _footer(pdf)

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'results' not in st.session_state:
    st.session_state.results = []

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
with st.spinner('Getting ready...'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f'Could not load the AI model: {e}')
        st.stop()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-brand'>
        <span class='sb-icon'>🐄</span>
        <div class='sb-name'>Cattle Health Check</div>
        <div class='sb-tagline'>Take a photo. Know what is wrong.<br>Know what to do next.</div>
    </div>
    """, unsafe_allow_html=True)

    # Theme switcher
    st.markdown("<div class='sb-nav-label'>Appearance</div>", unsafe_allow_html=True)
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("🌙 Dark", use_container_width=True):
            set_theme('dark')
            st.rerun()
    with theme_col2:
        if st.button("☀️ Light", use_container_width=True):
            set_theme('light')
            st.rerun()

    st.markdown("<div class='sb-nav-label'>Menu</div>", unsafe_allow_html=True)

    page = st.radio(
        "menu",
        ["🏠  Home", "📷  Check My Cattle", "📋  Past Checks", "❓  How It Works"],
        label_visibility='collapsed'
    )

    st.markdown("""
    <div class='sb-tip'>
        <strong>Best results:</strong><br>
        Take the photo in good natural light. Get close enough to see the animal clearly.
        One animal per photo.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div class='hero'>
        <div class='hero-eyebrow'>For Kenyan Cattle Farmers</div>
        <div class='hero-title'>Is your cow<br><em>sick or healthy?</em></div>
        <div class='hero-body'>
            Take a photo of your animal and upload it here.
            In seconds you will know what condition your cattle may have
            and exactly what steps to take next.
        </div>
        <div class='trust-row'>
            <div class='trust-item'><span class='trust-dot'>✓</span> Results in under 5 seconds</div>
            <div class='trust-item'><span class='trust-dot'>✓</span> Free to use</div>
            <div class='trust-item'><span class='trust-dot'>✓</span> 94% accurate</div>   <!-- rounded from 93.5 -->
            <div class='trust-item'><span class='trust-dot'>✓</span> Works on any phone</div>
        </div>
        <div class='hero-cta-note'>
            Use the menu on the left and tap "Check My Cattle" to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-heading'>Tips for a Good Photo</div>", unsafe_allow_html=True)
    cols = st.columns(4, gap="small")
    tips = [
        ("☀️", "Good Light",       "Take the photo outside in daylight or in a well-lit shed. Avoid dark or blurry images."),
        ("📍", "One Animal",       "One animal per photo. Make sure the animal fills most of the frame."),
        ("🔍", "Show the Problem", "If you see a wound, swelling, or rash, get close enough for it to be visible."),
        ("📱", "Any Phone",        "Any smartphone camera will work. No special equipment needed."),
    ]
    for col, (icon, title, body) in zip(cols, tips):
        with col:
            st.markdown(f"""
            <div class='tip-card'>
                <div class='tip-icon'>{icon}</div>
                <div class='tip-title'>{title}</div>
                <div class='tip-body'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><div class='section-heading'>What We Check For</div>", unsafe_allow_html=True)
    for key, info in DISEASE_INFO.items():
        st.markdown(f"""
        <div class='disease-ref-card'>
            <div class='drc-header'>
                <span style='font-size:1.3rem'>{info['emoji']}</span>
                <span class='drc-name'>{info['full_name']}</span>
                <span class='drc-sev'
                    style='background:{info["severity_bg"]};color:{info["severity_color"]}'>
                    {info['severity']}
                </span>
            </div>
            <div class='drc-body'>
                <strong style='color:var(--cream3)'>Signs:</strong> {info['what_you_see']}
            </div>
            <div class='drc-action'>
                <strong>If detected:</strong> {info['urgency_msg']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
        This tool gives you a starting point, not a final answer.
        Always follow up with a licensed veterinary officer for treatment decisions.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHECK MY CATTLE
# ══════════════════════════════════════════════════════════════
elif page == "📷  Check My Cattle":

    st.markdown("""
    <div class='pg-hd'>
        <div class='pg-title'>Check My Cattle</div>
        <div class='pg-sub'>Upload photos and get your results in seconds</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='step-no'>Step 1 of 3</div>", unsafe_allow_html=True)
    st.markdown("<div class='step-title'>Upload Your Photos</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "upload", type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True, label_visibility='collapsed'
    )

    if uploaded_files and len(uploaded_files) > MAX_IMAGES:
        st.warning(f'You can upload up to {MAX_IMAGES} photos at a time. Only the first {MAX_IMAGES} will be checked.')
        uploaded_files = uploaded_files[:MAX_IMAGES]

    if uploaded_files:
        st.markdown(
            f"<div style='color:var(--cream4);font-size:0.84rem;margin:0.7rem 0'>"
            f"{len(uploaded_files)} photo(s) uploaded.</div>",
            unsafe_allow_html=True
        )
        prev_cols = st.columns(min(len(uploaded_files), 5))
        for i, (col, uf) in enumerate(zip(prev_cols, uploaded_files)):
            with col:
                st.image(Image.open(uf), caption=f'Animal {i+1}', use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='step-no'>Step 2 of 3</div>", unsafe_allow_html=True)
        st.markdown("<div class='step-title'>Run the Check</div>", unsafe_allow_html=True)

        if st.button("🔍  Check My Cattle Now", use_container_width=True):
            results  = []
            progress = st.progress(0)
            status   = st.empty()

            for i, uf in enumerate(uploaded_files):
                status.markdown(
                    f"<div style='color:var(--cream4);font-size:0.88rem'>"
                    f"Checking animal {i+1} of {len(uploaded_files)}...</div>",
                    unsafe_allow_html=True
                )
                uf.seek(0)
                image = Image.open(uf)
                predicted_class, confidence, all_preds = predict(image, model)
                results.append({
                    'filename':   uf.name,
                    'image':      image.copy(),
                    'disease':    predicted_class,
                    'confidence': confidence,
                    'all_preds':  all_preds,
                    'timestamp':  datetime.now().strftime('%d %b %Y  %I:%M %p')
                })
                st.session_state.history.append({
                    'filename':   uf.name,
                    'disease':    predicted_class,
                    'confidence': confidence,
                    'severity':   DISEASE_INFO[predicted_class]['severity'],
                    'timestamp':  datetime.now().strftime('%d %b %Y  %I:%M %p')
                })
                progress.progress((i + 1) / len(uploaded_files))

            status.success(f'Done! Results ready for {len(results)} animal(s).')
            st.session_state.results = results

    else:
        st.markdown("""
        <div class='upload-hint'>
            <div style='font-size:2.2rem;margin-bottom:0.7rem'>📷</div>
            <div style='color:var(--cream3);font-size:0.92rem;font-weight:500;margin-bottom:0.3rem'>
                Click "Browse files" or drag your photos here
            </div>
            <div style='color:var(--cream5);font-size:0.79rem'>
                JPG or PNG &nbsp;·&nbsp; Up to 5 photos &nbsp;·&nbsp; One animal per photo
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RESULTS ──
    if st.session_state.results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='step-no'>Step 3 of 3</div>", unsafe_allow_html=True)
        st.markdown("<div class='step-title'>Your Results</div>", unsafe_allow_html=True)

        for i, r in enumerate(st.session_state.results):
            info = DISEASE_INFO[r['disease']]
            if i > 0:
                st.markdown("---")

            st.markdown(
                f"<div style='font-size:0.65rem;color:var(--cream5);text-transform:uppercase;"
                f"letter-spacing:0.12em;margin-bottom:0.6rem'>"
                f"Animal {i+1} &nbsp;·&nbsp; {r['filename']}</div>",
                unsafe_allow_html=True
            )

            col_img, col_res = st.columns([1, 1], gap="large")
            with col_img:
                st.image(r['image'], use_column_width=True)

            with col_res:
                st.markdown(f"""
                <div class='res-card'>
                    <div class='res-verdict'>{info['emoji']} {info['full_name']}</div>
                    <div class='res-conf'>Confidence: {r['confidence']:.0f}%</div>   <!-- whole number -->
                    <div class='urgency-bar'
                        style='background:{info["severity_bg"]};
                               color:{info["severity_color"]};
                               border:1px solid {info["severity_color"]}33'>
                        {info['urgency_msg']}
                    </div>
                    <div class='section-lbl'>What You Are Seeing</div>
                    <div class='what-you-see'>{info['what_you_see']}</div>
                    <div class='section-lbl'>What To Do</div>
                    <div class='action-block'>{info['what_to_do']}</div>
                    <div class='m-row'>
                        <div class='m-box'>
                            <div class='m-val'>{r['confidence']:.0f}%</div>   <!-- whole number -->
                            <div class='m-lbl'>Confidence</div>
                        </div>
                        <div class='m-box'>
                            <div class='m-val'
                                style='color:{info["severity_color"]};font-size:0.82rem'>
                                {info['severity']}
                            </div>
                            <div class='m-lbl'>Status</div>
                        </div>
                        <div class='m-box'>
                            <div class='m-val'>{'Yes' if info['requires_vet'] else 'No'}</div>
                            <div class='m-lbl'>Call Vet</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander(f"  See full analysis breakdown — Animal {i+1}"):
                for cname, prob in zip(CLASS_NAMES, r['all_preds']):
                    pct    = float(prob) * 100
                    is_top = cname == r['disease']
                    label  = DISEASE_INFO[cname]['full_name']
                    st.markdown(f"""
                    <div style='margin:5px 0'>
                        <div class='p-head'>
                            <span>{'✔ ' if is_top else ''}{label}</span>
                            <span>{pct:.0f}%</span>   <!-- whole number -->
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(pct / 100)

        # ── DOWNLOAD ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='color:var(--cream5);font-size:0.82rem;margin-bottom:0.8rem'>"
            "Save your results as a PDF to share with your vet or keep for your records.</div>",
            unsafe_allow_html=True
        )
        try:
            pdf_bytes = generate_pdf(st.session_state.results)
            filename  = f"CattleHealthCheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            size_kb   = len(pdf_bytes) / 1024
            st.download_button(
                label="📥  Save Report as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime='application/pdf',
                use_container_width=True
            )
            st.markdown(
                f"<div style='color:var(--cream5);font-size:0.72rem;margin-top:0.4rem;text-align:center'>"
                f"Report size: ~{size_kb:.0f} KB &nbsp;·&nbsp; "
                f"{len(st.session_state.results)} page(s)</div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f'Could not generate PDF: {e}')

    st.markdown("""
    <div class='disclaimer'>
        These results are a guide only and do not replace a professional veterinary examination.
        If your animal is seriously ill, contact your vet immediately.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAST CHECKS
# ══════════════════════════════════════════════════════════════
elif page == "📋  Past Checks":

    st.markdown("""
    <div class='pg-hd'>
        <div class='pg-title'>Past Checks</div>
        <div class='pg-sub'>All checks you have done in this session</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("You have not done any checks yet. Go to 'Check My Cattle' to get started.")
    else:
        total    = len(st.session_state.history)
        healthy  = sum(1 for h in st.session_state.history if h['disease'] == 'healthy')
        diseased = total - healthy
        avg_conf = sum(h['confidence'] for h in st.session_state.history) / total

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Animals Checked", total)
        with c2: st.metric("Healthy", healthy)
        with c3: st.metric("Disease Found", diseased)
        with c4: st.metric("Avg Confidence", f"{avg_conf:.0f}%")   # whole number

        st.markdown("---")

        for h in reversed(st.session_state.history):
            info      = DISEASE_INFO[h['disease']]
            sev_color = info['severity_color']
            st.markdown(f"""
            <div class='hist-card'>
                <div style='font-size:1.7rem;flex-shrink:0'>{info['emoji']}</div>
                <div>
                    <div class='hist-main'>{info['full_name']}</div>
                    <div class='hist-meta'>
                        {h['filename']} &nbsp;·&nbsp;
                        {h['confidence']:.0f}% confidence &nbsp;·&nbsp;   <!-- whole number -->
                        Status: <span style='color:{sev_color};font-weight:600'>
                            {h['severity']}
                        </span> &nbsp;·&nbsp; {h['timestamp']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️  Clear History"):
            st.session_state.history = []
            st.session_state.results = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif page == "❓  How It Works":

    st.markdown("""
    <div class='pg-hd'>
        <div class='pg-title'>How It Works</div>
        <div class='pg-sub'>Simple answers to common questions</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-card'>
        <div class='about-title'>What does this tool do?</div>
        <div class='about-text'>
            You take a photo of your cattle and upload it here. The tool analyses the image
            and tells you whether the animal looks healthy or shows signs of a common disease.
            It then tells you exactly what to do next.
        </div>
    </div>
    <div class='about-card'>
        <div class='about-title'>Which diseases can it find?</div>
        <div class='about-text'>
            The tool currently checks for four conditions: Foot and Mouth Disease (FMD),
            Lumpy Skin Disease (LSD), Mastitis, and Healthy (no disease). These are among
            the most common and costly cattle diseases in Kenya.
        </div>
    </div>
    <div class='about-card'>
        <div class='about-title'>How accurate is it?</div>
        <div class='about-text'>
            The tool was trained on 2,800 images and correctly identifies the condition
            94 out of 100 times in testing. A clear, well-lit photo taken close to the
            animal will give you better results.
        </div>
    </div>
    <div class='about-card'>
        <div class='about-title'>Does it replace my vet?</div>
        <div class='about-text'>
            No. This tool helps you understand what might be wrong and what to do first.
            It is not a substitute for a licensed veterinary officer, especially for serious
            or urgent conditions. Think of it as a first check, not a final diagnosis.
        </div>
    </div>
    <div class='about-card'>
        <div class='about-title'>Is my data safe?</div>
        <div class='about-text'>
            Your photos are used only for analysis and are not stored or shared.
            Your history is only kept while your browser is open and clears when you close it.
        </div>
    </div>
    <div class='about-card'>
        <div class='about-title'>Per-Condition Accuracy</div>
        <table class='ab-table'>
            <tr><th>Condition</th><th>Correctly Identified</th><th>F1 Score</th></tr>
            <tr><td>Foot and Mouth Disease</td><td>89%</td><td>0.90</td></tr>
            <tr><td>Healthy Cattle</td><td>93%</td><td>0.94</td></tr>
            <tr><td>Lumpy Skin Disease</td><td>97%</td><td>0.95</td></tr>
            <tr><td>Mastitis</td><td>92%</td><td>0.89</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
        This tool provides AI-assisted guidance only and does NOT replace a licensed veterinary diagnosis.
        Always consult a vet for serious conditions or before administering treatment.
    </div>
    """, unsafe_allow_html=True)