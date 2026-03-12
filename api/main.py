import io
import os
from datetime import datetime
from typing import List

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fpdf import FPDF
import tensorflow as tf

# -------------------------------------------------------------------
# Model + core config
# -------------------------------------------------------------------

MODEL_PATH = "cattle_disease_model.h5"

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["fmd", "healthy", "lumpy skin", "mastitis"]

DISEASE_INFO = {
    "fmd": {
        "full_name":    "Foot and Mouth Disease",
        "emoji":        "🟠",
        "severity":     "Urgent",
        "requires_vet": True,
        "what_you_see": "Blisters on the mouth, tongue, and hooves. The animal may be drooling heavily or struggling to walk.",
        "what_to_do":   "Separate this animal from the rest of the herd right away. Do not move animals off the farm. Call your vet or the nearest livestock office today — FMD spreads very fast.",
        "urgency_msg":  "Act today. Every hour matters.",
    },
    "healthy": {
        "full_name":    "No Disease Detected",
        "emoji":        "🟢",
        "severity":     "All Clear",
        "requires_vet": False,
        "what_you_see": "The animal shows no visible signs of disease.",
        "what_to_do":   "Your animal looks healthy. Keep up regular check-ups, ensure clean water and feed, and stay on your vaccination schedule.",
        "urgency_msg":  "Continue routine care.",
    },
    "lumpy skin": {
        "full_name":    "Lumpy Skin Disease",
        "emoji":        "🔴",
        "severity":     "Urgent",
        "requires_vet": True,
        "what_you_see": "Round raised lumps or nodules appearing across the skin. The animal may have a fever and reduced milk output.",
        "what_to_do":   "Separate the animal from the herd immediately. LSD spreads through insect bites — treat the whole herd with insect repellent. Contact your vet to vaccinate at-risk animals.",
        "urgency_msg":  "Isolate today. Protect the herd.",
    },
    "mastitis": {
        "full_name":    "Mastitis",
        "emoji":        "🟡",
        "severity":     "Needs Attention",
        "requires_vet": True,
        "what_you_see": "The udder looks swollen or feels warm and painful. Milk may appear watery, lumpy, or discoloured.",
        "what_to_do":   "Contact your vet for antibiotic treatment. Milk the affected quarters separately and discard that milk. Wash hands and equipment between animals.",
        "urgency_msg":  "Book a vet visit soon.",
    },
}


def safe(text):
    return (
        str(text)
        .replace("\u2014", "-").replace("\u2013", "-")
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2192", "->").replace("\u2022", "-")
    )

# -------------------------------------------------------------------
# PDF builder (from Streamlit app)
# -------------------------------------------------------------------

def build_pdf(results):
    PAGE_W = 210
    PAGE_H = 297
    M = 12
    CONT_W = PAGE_W - M * 2
    IMG_W = 68
    COL2_X = M + IMG_W + 6
    COL2_W = PAGE_W - COL2_X - M
    FOOTER_Y = PAGE_H - 11
    MAX_Y = PAGE_H - 20

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for rec_idx, r in enumerate(results):
        pdf.add_page()
        info = DISEASE_INFO[r["disease"]]

        # Header
        pdf.set_fill_color(31, 41, 55)
        pdf.rect(0, 0, PAGE_W, 22, "F")
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(M, 5)
        pdf.cell(CONT_W, 7, safe("Cattle Health Check  -  Disease Report"))
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(156, 163, 175)
        pdf.set_xy(M, 14)
        pdf.cell(
            CONT_W,
            5,
            safe(
                f'Date: {datetime.now().strftime("%d %B %Y  %I:%M %p")}  |  '
                f'Animal {rec_idx + 1} of {len(results)}  |  Photo: {r["filename"]}'
            ),
        )

        y = 26

        # Image
        img_path = os.path.join(base_dir, f"_tmp_img_{rec_idx}.jpg")
        pil_img = Image.open(io.BytesIO(r["image_bytes"])).convert("RGB")
        w, h = pil_img.size
        px = int(IMG_W * 3.7795)  # mm to px approx
        if w > px:
            pil_img = pil_img.resize((px, int(h * px / w)), Image.LANCZOS)
        img_h_mm = IMG_W * (pil_img.size[1] / pil_img.size[0])
        pil_img.save(img_path, format="JPEG", quality=55, optimize=True)
        pdf.image(img_path, x=M, y=y, w=IMG_W)
        os.remove(img_path)

        # Right column
        ry = y
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(31, 41, 55)
        pdf.set_xy(COL2_X, ry)
        pdf.multi_cell(COL2_W, 6.5, safe(info["full_name"]))
        ry = pdf.get_y()
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(107, 114, 128)
        pdf.set_xy(COL2_X, ry)
        pdf.cell(COL2_W, 5, safe(f'Confidence: {r["confidence"]:.0f}%'))
        ry += 6
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(21, 128, 61)
        pdf.set_xy(COL2_X, ry)
        pdf.cell(COL2_W, 5, safe(info["urgency_msg"]))
        ry += 7

        for lbl, val in [
            ("Vet Required", "Yes" if info["requires_vet"] else "No"),
            ("Status", info["severity"]),
        ]:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(107, 114, 128)
            pdf.set_xy(COL2_X, ry)
            pdf.cell(28, 5, safe(lbl + ":"), ln=False)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(55, 65, 81)
            pdf.cell(COL2_W - 28, 5, safe(val))
            ry += 5.5

        y = max(y + img_h_mm, ry) + 5
        pdf.set_draw_color(209, 213, 219)
        pdf.line(M, y, PAGE_W - M, y)
        y += 4

        # What you see
        if y < MAX_Y:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y)
            pdf.cell(CONT_W, 5, "What You Are Seeing:")
            y += 5.5
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(75, 85, 99)
            pdf.set_xy(M, y)
            pdf.multi_cell(CONT_W, 4.5, safe(info["what_you_see"]))
            y = pdf.get_y() + 3

        # What to do
        if y < MAX_Y:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y)
            pdf.cell(CONT_W, 5, "What To Do:")
            y += 5.5
            pdf.set_fill_color(249, 250, 251)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(55, 65, 81)
            pdf.set_xy(M, y)
            pdf.multi_cell(CONT_W, 4.5, safe(info["what_to_do"]), fill=True)
            y = pdf.get_y() + 5

        # Analysis breakdown bars
        LABEL_W = 78
        PCT_W = 16
        BAR_X = M + LABEL_W + PCT_W
        BAR_W = CONT_W - LABEL_W - PCT_W
        if y < MAX_Y:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y)
            pdf.cell(CONT_W, 5, "Analysis Breakdown:")
            y += 6
            for cname, prob in zip(CLASS_NAMES, r["all_preds"]):
                if y >= MAX_Y:
                    break
                pct = float(prob) * 100
                is_top = cname == r["disease"]
                fill_w = BAR_W * (pct / 100.0)

                pdf.set_font("Helvetica", "B" if is_top else "", 7.5)
                pdf.set_text_color(31, 41, 55)
                pdf.set_xy(M, y)
                pdf.cell(
                    LABEL_W,
                    5,
                    safe(("> " if is_top else "   ") + DISEASE_INFO[cname]["full_name"]),
                    ln=False,
                )
                pdf.set_font("Helvetica", "B" if is_top else "", 7.5)
                pdf.set_text_color(21, 128, 61)
                pdf.cell(PCT_W, 5, f"{pct:.0f}%", ln=False)

                bar_y = y + 1.2
                pdf.set_fill_color(229, 231, 235)
                pdf.rect(BAR_X, bar_y, BAR_W, 2.8, "F")
                if fill_w > 0.1:
                    if is_top:
                        pdf.set_fill_color(21, 128, 61)
                    else:
                        pdf.set_fill_color(156, 163, 175)
                    pdf.rect(BAR_X, bar_y, fill_w, 2.8, "F")

                y += 6.5

        # Footer
        pdf.set_xy(M, FOOTER_Y)
        pdf.set_font("Helvetica", "I", 6)
        pdf.set_text_color(140, 140, 140)
        pdf.cell(
            CONT_W,
            4,
            safe(
                "AI-assisted only. Does NOT replace a veterinary diagnosis. "
                "Always consult a licensed vet.  |  Cattle Health Check Kenya"
            ),
            align="C",
        )

    return bytes(pdf.output())

# -------------------------------------------------------------------
# Prediction helper
# -------------------------------------------------------------------

def predict_single(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    pred = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx], float(pred[idx]) * 100.0, pred

# -------------------------------------------------------------------
# FastAPI app + endpoints
# -------------------------------------------------------------------

app = FastAPI(title="Cattle Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single-image prediction, JSON response + recommendations."""
    image_bytes = await file.read()
    disease, confidence, all_preds = predict_single(image_bytes)

    info = DISEASE_INFO[disease]

    return {
        "predicted_class": disease,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(p) for i, p in enumerate(all_preds)
        },
        "info": {
            "full_name": info["full_name"],
            "what_you_see": info["what_you_see"],
            "what_to_do": info["what_to_do"],
            "urgency": info["urgency_msg"],
            "severity": info["severity"],
            "requires_vet": info["requires_vet"],
        },
    }

@app.post("/analyze_batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Multi-image analysis (1–5 images) via a single 'files' field.
    Returns a downloadable PDF report summarizing predictions + recommendations.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="You can upload at most 5 images.")

    results = []
    for idx, file in enumerate(files, start=1):
        image_bytes = await file.read()
        disease, confidence, all_preds = predict_single(image_bytes)
        results.append(
            {
                "index": idx,
                "filename": file.filename,
                "disease": disease,
                "confidence": confidence,
                "all_preds": all_preds,
                "image_bytes": image_bytes,
            }
        )

    pdf_bytes = build_pdf(results)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="classification_report.pdf"'
        },
    )