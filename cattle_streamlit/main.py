import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import warnings
import json
import os
import io
import threading

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

CLASS_NAMES  = ['fmd', 'healthy', 'lumpy skin', 'mastitis']
IMG_SIZE     = (224, 224)
MAX_IMAGES   = 5
ARCH_PATH    = 'cattle_final_archi.json'
WEIGHTS_PATH = 'cattle_final.weights.h5'

DISEASE_INFO = {
    'fmd': {
        'full_name':    'Foot and Mouth Disease',
        'emoji':        '🟠',
        'severity':     'Urgent',
        'requires_vet': True,
        'what_you_see': 'Blisters on the mouth, tongue, and hooves. The animal may be drooling heavily or struggling to walk.',
        'what_to_do':   'Separate this animal from the rest of the herd right away. Do not move animals off the farm. Call your vet or the nearest livestock office today — FMD spreads very fast.',
        'urgency_msg':  'Act today. Every hour matters.',
    },
    'healthy': {
        'full_name':    'No Disease Detected',
        'emoji':        '🟢',
        'severity':     'All Clear',
        'requires_vet': False,
        'what_you_see': 'The animal shows no visible signs of disease.',
        'what_to_do':   'Your animal looks healthy. Keep up regular check-ups, ensure clean water and feed, and stay on your vaccination schedule.',
        'urgency_msg':  'Continue routine care.',
    },
    'lumpy skin': {
        'full_name':    'Lumpy Skin Disease',
        'emoji':        '🔴',
        'severity':     'Urgent',
        'requires_vet': True,
        'what_you_see': 'Round raised lumps or nodules appearing across the skin. The animal may have a fever and reduced milk output.',
        'what_to_do':   'Separate the animal from the herd immediately. LSD spreads through insect bites — treat the whole herd with insect repellent. Contact your vet to vaccinate at-risk animals.',
        'urgency_msg':  'Isolate today. Protect the herd.',
    },
    'mastitis': {
        'full_name':    'Mastitis',
        'emoji':        '🟡',
        'severity':     'Needs Attention',
        'requires_vet': True,
        'what_you_see': 'The udder looks swollen or feels warm and painful. Milk may appear watery, lumpy, or discoloured.',
        'what_to_do':   'Contact your vet for antibiotic treatment. Milk the affected quarters separately and discard that milk. Wash hands and equipment between animals.',
        'urgency_msg':  'Book a vet visit soon.',
    },
}

def safe(text):
    return (str(text)
        .replace('\u2014', '-').replace('\u2013', '-')
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2192', '->').replace('\u2022', '-')
    )

# Pure function — no Streamlit calls whatsoever
def build_pdf(results):
    PAGE_W = 210; PAGE_H = 297; M = 12
    CONT_W = PAGE_W - M * 2; IMG_W = 68
    COL2_X = M + IMG_W + 6; COL2_W = PAGE_W - COL2_X - M
    FOOTER_Y = PAGE_H - 11; MAX_Y = PAGE_H - 20

    pdf = FPDF(unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)

    for rec_idx, r in enumerate(results):
        pdf.add_page()
        info = DISEASE_INFO[r['disease']]

        # Header
        pdf.set_fill_color(31, 41, 55)
        pdf.rect(0, 0, PAGE_W, 22, 'F')
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(M, 5)
        pdf.cell(CONT_W, 7, safe('Cattle Health Check  -  Disease Report'))
        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(156, 163, 175)
        pdf.set_xy(M, 14)
        pdf.cell(CONT_W, 5, safe(
            f'Date: {datetime.now().strftime("%d %B %Y  %I:%M %p")}  |  '
            f'Animal {rec_idx+1} of {len(results)}  |  Photo: {r["filename"]}'
        ))

        y = 26

        # Write image to a plain file path — no context managers, no try/except
        img_h_mm = 0
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'_tmp_img_{rec_idx}.jpg')
        pil_img  = Image.open(io.BytesIO(r['image_bytes'])).convert('RGB')
        w, h     = pil_img.size
        px       = int(IMG_W * 3.7795)
        if w > px:
            pil_img = pil_img.resize((px, int(h * px / w)), Image.LANCZOS)
        img_h_mm = IMG_W * (pil_img.size[1] / pil_img.size[0])
        pil_img.save(img_path, format='JPEG', quality=55, optimize=True)
        pdf.image(img_path, x=M, y=y, w=IMG_W)
        os.remove(img_path)

        # Right column
        ry = y
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(31, 41, 55)
        pdf.set_xy(COL2_X, ry)
        pdf.multi_cell(COL2_W, 6.5, safe(info['full_name']))
        ry = pdf.get_y()
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(107, 114, 128)
        pdf.set_xy(COL2_X, ry)
        pdf.cell(COL2_W, 5, safe(f'Confidence: {r["confidence"]:.0f}%'))
        ry += 6
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(21, 128, 61)
        pdf.set_xy(COL2_X, ry)
        pdf.cell(COL2_W, 5, safe(info['urgency_msg']))
        ry += 7
        for lbl, val in [('Vet Required', 'Yes' if info['requires_vet'] else 'No'), ('Status', info['severity'])]:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(COL2_X, ry); pdf.cell(28, 5, safe(lbl + ':'), ln=False)
            pdf.set_font('Helvetica', '', 8); pdf.set_text_color(55, 65, 81)
            pdf.cell(COL2_W - 28, 5, safe(val)); ry += 5.5

        y = max(y + img_h_mm, ry) + 5
        pdf.set_draw_color(209, 213, 219)
        pdf.line(M, y, PAGE_W - M, y)
        y += 4

        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y); pdf.cell(CONT_W, 5, 'What You Are Seeing:'); y += 5.5
            pdf.set_font('Helvetica', 'I', 8); pdf.set_text_color(75, 85, 99)
            pdf.set_xy(M, y); pdf.multi_cell(CONT_W, 4.5, safe(info['what_you_see'])); y = pdf.get_y() + 3

        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y); pdf.cell(CONT_W, 5, 'What To Do:'); y += 5.5
            pdf.set_fill_color(249, 250, 251); pdf.set_font('Helvetica', '', 8); pdf.set_text_color(55, 65, 81)
            pdf.set_xy(M, y); pdf.multi_cell(CONT_W, 4.5, safe(info['what_to_do']), fill=True); y = pdf.get_y() + 5

        LABEL_W = 78; PCT_W = 16
        BAR_X   = M + LABEL_W + PCT_W; BAR_W = CONT_W - LABEL_W - PCT_W
        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y); pdf.cell(CONT_W, 5, 'Analysis Breakdown:'); y += 6
            for cname, prob in zip(CLASS_NAMES, r['all_preds']):
                if y >= MAX_Y: break
                pct    = float(prob) * 100
                is_top = cname == r['disease']
                fill_w = BAR_W * (pct / 100.0)
                pdf.set_font('Helvetica', 'B' if is_top else '', 7.5); pdf.set_text_color(31, 41, 55)
                pdf.set_xy(M, y)
                pdf.cell(LABEL_W, 5, safe(('> ' if is_top else '   ') + DISEASE_INFO[cname]['full_name']), ln=False)
                pdf.set_font('Helvetica', 'B' if is_top else '', 7.5); pdf.set_text_color(21, 128, 61)
                pdf.cell(PCT_W, 5, f'{pct:.0f}%', ln=False)
                bar_y = y + 1.2
                pdf.set_fill_color(229, 231, 235); pdf.rect(BAR_X, bar_y, BAR_W, 2.8, 'F')
                if fill_w > 0.1:
                    pdf.set_fill_color(21, 128, 61) if is_top else pdf.set_fill_color(156, 163, 175)
                    pdf.rect(BAR_X, bar_y, fill_w, 2.8, 'F')
                y += 6.5

        # Footer
        pdf.set_xy(M, FOOTER_Y)
        pdf.set_font('Helvetica', 'I', 6)
        pdf.set_text_color(140, 140, 140)
        pdf.cell(CONT_W, 4,
            safe('AI-assisted only. Does NOT replace a veterinary diagnosis. '
                 'Always consult a licensed vet.  |  Cattle Health Check Kenya'),
            align='C')

    return bytes(pdf.output())


st.set_page_config(
    page_title="Cattle Health Check",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 17px !important; }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.5rem !important; }
    .stMetric label { font-size: 1rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .stButton > button { font-size: 1.1rem !important; padding: 0.6rem 1.2rem !important; }
    .stCaption { font-size: 0.95rem !important; }
    p, li, .stMarkdown { font-size: 1.05rem !important; line-height: 1.7 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    keras_path = WEIGHTS_PATH.replace('.weights.h5', '.keras')

    if os.path.exists(keras_path):
        return tf.keras.models.load_model(keras_path)

    with open(ARCH_PATH, 'r') as f:
        config = json.load(f)

    def fix_config(obj):
        if isinstance(obj, dict):
            obj.pop('groups', None)
            for v in obj.values():
                fix_config(v)
        elif isinstance(obj, list):
            for item in obj:
                fix_config(item)

    fix_config(config)
    m = tf.keras.Model.from_config(config)
    m.load_weights(WEIGHTS_PATH)
    m.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    m.save(keras_path)
    return m


def predict(image, model):
    img  = image.convert('RGB').resize(IMG_SIZE)
    arr  = np.expand_dims(np.array(img, dtype=np.float32), 0)
    pred = model.predict(arr, verbose=0)[0]
    idx  = int(np.argmax(pred))
    return CLASS_NAMES[idx], float(pred[idx]) * 100, pred


# ── SESSION STATE ─────────────────────────────────────────────
for key, default in [('history', []), ('results', []), ('pdf_cache', None), ('pdf_building', False)]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.results and 'image_bytes' not in st.session_state.results[0]:
    st.session_state.results   = []
    st.session_state.history   = []
    st.session_state.pdf_cache = None

# Build PDF in a background thread — completely outside Streamlit's render loop
def _build_in_background(results):
    result_holder = {}
    def worker():
        result_holder['pdf'] = build_pdf(results)
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return result_holder.get('pdf')

if st.session_state.results and st.session_state.pdf_cache is None and not st.session_state.pdf_building:
    st.session_state.pdf_building = True
    st.session_state.pdf_cache    = _build_in_background(st.session_state.results)
    st.session_state.pdf_building = False

# ── LOAD MODEL ────────────────────────────────────────────────
try:
    model = load_model()
except Exception as e:
    st.error(f'Could not load the AI model: {e}')
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🐄 Cattle Health Check")
    st.caption("Take a photo. Know what's wrong. Know what to do next.")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠  Home", "📷  Check My Cattle", "📋  Past Checks", "❓  How It Works"],
        label_visibility="collapsed"
    )
    st.divider()
    st.info("📸 **Best results:** Take photos in good natural light, close enough to see the animal clearly. One animal per photo.")


# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("🐄 Cattle Health Check")
    st.subheader("Is your cow sick or healthy?")
    st.write("Take a photo of your animal and upload it here. In seconds you will know what condition your cattle may have and exactly what steps to take next.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Speed", "< 5 sec", "results")
    col2.metric("Cost", "Free", "to use")
    col3.metric("Accuracy", "94%", "overall")
    col4.metric("Device", "Any", "phone works")

    st.divider()
    st.subheader("Tips for a Good Photo")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("☀️ **Good Light**\n\nTake the photo outside in daylight or in a well-lit shed.")
    with c2:
        st.info("📍 **One Animal**\n\nOne animal per photo. Make sure it fills most of the frame.")
    with c3:
        st.info("🔍 **Show the Problem**\n\nIf you see a wound or rash, get close enough for it to be visible.")
    with c4:
        st.info("📱 **Any Phone**\n\nAny smartphone camera will work.")

    st.divider()
    st.subheader("What We Check For")
    for key, info in DISEASE_INFO.items():
        with st.expander(f"{info['emoji']}  {info['full_name']}  —  {info['severity']}"):
            st.write(f"**Signs:** {info['what_you_see']}")
            st.write(f"**If detected:** {info['urgency_msg']}")

    st.divider()
    st.caption("⚠️ This tool gives you a starting point, not a final answer. Always follow up with a licensed veterinary officer.")


# ══════════════════════════════════════════════════════════════
# CHECK MY CATTLE
# ══════════════════════════════════════════════════════════════
elif page == "📷  Check My Cattle":
    st.title("📷 Check My Cattle")
    st.write("Upload photos and get your results in seconds.")
    st.divider()

    st.markdown("**Step 1 — Upload Your Photos**")
    uploaded_files = st.file_uploader(
        "Choose photos", type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True, label_visibility='collapsed'
    )

    if uploaded_files and len(uploaded_files) > MAX_IMAGES:
        st.warning(f'Maximum {MAX_IMAGES} photos at a time. Only the first {MAX_IMAGES} will be checked.')
        uploaded_files = uploaded_files[:MAX_IMAGES]

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} photo(s) uploaded.")
        prev_cols = st.columns(min(len(uploaded_files), 5))
        for i, (col, uf) in enumerate(zip(prev_cols, uploaded_files)):
            with col:
                st.image(Image.open(uf), caption=f'Animal {i+1}', use_column_width=True)

        st.divider()
        st.markdown("**Step 2 — Run the Check**")

        if st.button("🔍  Check My Cattle Now", use_container_width=True, type="primary"):
            results  = []
            progress = st.progress(0)
            status   = st.empty()

            for i, uf in enumerate(uploaded_files):
                status.info(f"Checking photo {i+1} of {len(uploaded_files)}...")
                uf.seek(0)
                image_bytes = uf.read()
                image = Image.open(io.BytesIO(image_bytes))
                predicted_class, confidence, all_preds = predict(image, model)
                results.append({
                    'filename':    uf.name,
                    'image_bytes': image_bytes,
                    'disease':     predicted_class,
                    'confidence':  confidence,
                    'all_preds':   all_preds,
                    'timestamp':   datetime.now().strftime('%d %b %Y  %I:%M %p')
                })
                st.session_state.history.append({
                    'filename':   uf.name,
                    'disease':    predicted_class,
                    'confidence': confidence,
                    'severity':   DISEASE_INFO[predicted_class]['severity'],
                    'timestamp':  datetime.now().strftime('%d %b %Y  %I:%M %p')
                })
                progress.progress((i + 1) / len(uploaded_files))

            progress.empty()
            status.success(f'Done! {len(results)} animal(s) checked. Go to Download Report to save your PDF.')
            st.session_state.results   = results
            st.session_state.pdf_cache = None  # invalidate so PDF rebuilds fresh

    else:
        st.info("📷 Click **Browse files** above or drag and drop your photos here.\n\nJPG or PNG · Up to 5 photos · One animal per photo")

    if st.session_state.results:
        st.divider()
        st.markdown("**Step 3 — Your Results**")

        for i, r in enumerate(st.session_state.results):
            info = DISEASE_INFO[r['disease']]
            if i > 0:
                st.divider()

            st.caption(f"Animal {i+1}  ·  {r['filename']}")
            col_img, col_res = st.columns([1, 1], gap="large")

            with col_img:
                st.image(Image.open(io.BytesIO(r['image_bytes'])), use_column_width=True)

            with col_res:
                st.subheader(f"{info['emoji']}  {info['full_name']}")
                if info['severity'] == 'All Clear':
                    st.success(f"✅  {info['urgency_msg']}")
                elif info['severity'] == 'Needs Attention':
                    st.warning(f"⚠️  {info['urgency_msg']}")
                else:
                    st.error(f"🚨  {info['urgency_msg']}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{r['confidence']:.0f}%")
                c2.metric("Status", info['severity'])
                c3.metric("Call Vet", "Yes" if info['requires_vet'] else "No")
                st.markdown(f"**What You Are Seeing**\n\n*{info['what_you_see']}*")
                st.markdown(f"**What To Do**\n\n{info['what_to_do']}")

            with st.expander(f"Full analysis breakdown — Animal {i+1}"):
                for cname, prob in zip(CLASS_NAMES, r['all_preds']):
                    pct   = float(prob) * 100
                    color = "#15803d" if cname == r['disease'] else "#6b7280"
                    st.write(f"{'✔ ' if cname == r['disease'] else ''}{DISEASE_INFO[cname]['full_name']} — **{pct:.0f}%**")
                    st.markdown(
                        f'<div style="background:#374151;border-radius:4px;height:10px;margin-bottom:12px;">'
                        f'<div style="background:{color};width:{pct:.1f}%;height:10px;border-radius:4px;"></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        st.divider()
        st.caption("Save your results as a PDF to share with your vet or keep for your records.")
        if st.session_state.pdf_cache is None:
            bar   = st.progress(0)
            label = st.empty()
            label.write("Generating PDF...")
            import time
            for pct in range(0, 90, 10):
                bar.progress(pct)
                time.sleep(0.05)
            st.session_state.pdf_cache = _build_in_background(st.session_state.results)
            bar.progress(100)
            label.empty()
            bar.empty()
        if st.session_state.pdf_cache is not None:
            size_kb  = len(st.session_state.pdf_cache) / 1024
            filename = f"CattleHealthCheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="📥  Save Report as PDF",
                data=st.session_state.pdf_cache,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
            st.caption(f"~{size_kb:.0f} KB  ·  {len(st.session_state.results)} page(s)")

    st.divider()
    st.caption("⚠️ These results are a guide only and do not replace a professional veterinary examination.")


elif page == "📋  Past Checks":
    st.title("📋 Past Checks")
    st.write("All checks done in this session.")
    st.divider()

    if not st.session_state.history:
        st.info("No checks yet. Go to **Check My Cattle** to get started.")
    else:
        total    = len(st.session_state.history)
        healthy  = sum(1 for h in st.session_state.history if h['disease'] == 'healthy')
        diseased = total - healthy
        avg_conf = sum(h['confidence'] for h in st.session_state.history) / total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Animals Checked", total)
        c2.metric("Healthy", healthy)
        c3.metric("Disease Found", diseased)
        c4.metric("Avg Confidence", f"{avg_conf:.0f}%")

        st.divider()
        for h in reversed(st.session_state.history):
            info = DISEASE_INFO[h['disease']]
            st.write(
                f"{info['emoji']}  **{info['full_name']}**  —  "
                f"{h['filename']}  —  {h['confidence']:.0f}% confidence  —  "
                f"*{h['severity']}*  —  {h['timestamp']}"
            )

        st.divider()
        if st.button("🗑️  Clear History"):
            st.session_state.history   = []
            st.session_state.results   = []
            st.session_state.pdf_cache = None
            st.experimental_rerun()


# ══════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif page == "❓  How It Works":
    st.title("❓ How It Works")
    st.write("Simple answers to common questions.")
    st.divider()

    with st.expander("What does this tool do?", expanded=True):
        st.write("You take a photo of your cattle and upload it here. The tool analyses the image for signs of disease and tells you exactly what the condition may be and what to do next.")

    with st.expander("Which diseases can it find?"):
        st.write("The tool currently checks for four conditions: Foot and Mouth Disease (FMD), Lumpy Skin Disease (LSD), Mastitis, and Healthy (no disease). These are among the most common and costly cattle diseases in Kenya.")

    with st.expander("How accurate is it?"):
        st.write("The tool was trained on 2,800 images and correctly identifies the condition 94 out of 100 times in testing. A clear, well-lit photo taken close to the animal will give you better results.")
        st.table({
            "Condition":            ["Foot and Mouth Disease", "Healthy Cattle", "Lumpy Skin Disease", "Mastitis"],
            "Correctly Identified": ["89%", "93%", "97%", "92%"],
            "F1 Score":             ["0.90", "0.94", "0.95", "0.89"],
        })

    with st.expander("Does it replace my vet?"):
        st.write("No. This tool helps you understand what might be wrong and what to do first. It is not a substitute for a licensed veterinary officer.")

    with st.expander("Is my data safe?"):
        st.write("Your photos are used only for analysis and are not stored or shared. Your history is only kept while your browser is open and clears when you close it.")

    st.divider()
    st.caption("⚠️ This tool provides AI-assisted guidance only and does NOT replace a licensed veterinary diagnosis.")