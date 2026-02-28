import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
import warnings
import os
import json

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

CLASS_NAMES = ['fmd', 'healthy', 'lumpy skin', 'mastitis']
IMG_SIZE    = (224, 224)

@st.cache_resource
def load_model():
    from tensorflow.keras.optimizers import Adam

    with open('cattle_final_archi.json', 'r') as f:
        config = json.load(f)

    model = tf.keras.Model.from_config(config)
    model.load_weights('cattle_final.weights.h5')
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

st.set_page_config(
    page_title="LivestockAI Kenya",
    page_icon="🐄",
    layout="centered",
)

st.title("LivestockAI Kenya")
st.caption("AI-Powered Cattle Disease Detection")

# ─────────────────────────────────────
# LOAD MODEL INTO APP
# ─────────────────────────────────────
with st.spinner('Loading AI model...'):
    try:
        model = load_model()
        st.success('✅ Model loaded successfully — 93.5% validation accuracy')
    except Exception as e:
        st.error(f'❌ Could not load model: {e}')
        st.stop()

# ─────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────
def predict(image):
    img = image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]) * 100, preds

# ─────────────────────────────────────
# IMAGE UPLOAD
# ─────────────────────────────────────
st.markdown("---")
st.markdown("### 📷 Upload a Cattle Image")
st.caption("Supported formats: JPG, PNG, JPEG")

uploaded = st.file_uploader(
    "Choose an image",
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed'
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Analysing image...'):
        predicted_class, confidence, all_preds = predict(image)

    st.success(f'✅ Prediction: **{predicted_class.upper()}**')
    st.info(f'📊 Confidence: **{confidence:.1f}%**')