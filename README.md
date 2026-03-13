# Cattle Health Check
### AI-Powered Cattle Disease Detection for Kenyan Farmers

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Accuracy](https://img.shields.io/badge/Accuracy-93.5%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning web application that helps Kenyan cattle farmers identify common cattle diseases from a photo. Upload an image of your animal and receive an instant diagnosis, severity assessment, and recommended action — in seconds.

---

## Detectable Conditions

| Condition | Severity | Action |
|-----------|----------|--------|
| Foot and Mouth Disease (FMD) | 🔴 Urgent | Isolate immediately, call vet today |
| Lumpy Skin Disease (LSD) | 🔴 Urgent | Isolate, treat herd with repellent |
| Mastitis | 🟡 Needs Attention | Contact vet for antibiotic treatment |
| Healthy Cattle | 🟢 All Clear | Continue routine care |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Architecture | EfficientNetB0 (Transfer Learning) |
| Training Strategy | Two-phase: frozen base then fine-tuning |
| Training Images | 2,800 (700 per class, balanced) |
| Validation Accuracy | **93.5%** |
| Macro F1 Score | **0.92** |
| Framework | TensorFlow 2.10 / Keras |

---

## Prerequisites

Before you begin, make sure the following are installed on your machine:

- [Python 3.9 or 3.10](https://www.python.org/downloads/) — TensorFlow 2.10 does **not** support Python 3.11+
- [Git](https://git-scm.com/downloads) — includes Git Bash for Windows users

---

## Getting Started

### 1. Clone the Repository

Right-click on the folder where you want to save the project and select **"Git Bash Here"**, then run:

```bash
git clone https://github.com/Ashipunaz/cattle-disease-detection.git
cd cattle-disease-detection
```

---

### 2. Create and Activate a Virtual Environment

**Windows (Git Bash):**
```bash
python -m venv venv
source venv/Scripts/activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> You will know the environment is active when you see `(venv)` at the start of your terminal prompt.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Backend API (FastAPI)

This project exposes a production-ready HTTP API for model inference.

- API implementation: `api/main.py` (FastAPI + TensorFlow)
- Deployed base URL (Render): `https://cattle-disease-detection-q8x2.onrender.com`
- Interactive docs: `https://cattle-disease-detection-q8x2.onrender.com/docs`

See `API_CONSUMER_GUIDE.md` for full endpoint details and examples.

#### Run the API locally

From the project root with the virtual environment active:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Local docs: `http://127.0.0.1:8000/docs`

#### Docker (for local or other hosts)

```bash
docker build -t cattle-api .
docker run -p 8000:8000 cattle-api
```

### 5. Open in Jupyter Notebook *(optional — for exploring the model)*


Right-click on the project folder, select **"Git Bash Here"**, then run:

```bash
jupyter notebook
```

Jupyter will open in your browser automatically.

---

### 6. Run the Streamlit App

Navigate into the app folder and start the application:

```bash
cd cattle_streamlit
streamlit run main.py
```

The app will open in your browser at:
```
http://localhost:8501
```

---

## Deployment and Security

- The Render deployment reads its configuration from environment variables:
  - `CORS_ALLOW_ORIGINS` – comma-separated list of allowed origins.
  - `CORS_ALLOW_CREDENTIALS` – usually `false` unless you control the frontend.
  - `MAX_UPLOAD_BYTES`, `MAX_IMAGE_PIXELS` – upload and image safety limits.
  - `RATE_LIMIT_PER_MINUTE` – basic per-IP rate limiting.
  - `ADMIN_API_KEY` – optional; protects `/models/set-active` when enabled.
- Do **not** hard-code API keys or secrets in notebooks or source files. Use env vars instead.



---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure venv is active, then run `pip install -r requirements.txt` |
| `Could not load the AI model` | Check both model files are in the project root folder |
| `streamlit: command not found` | Run `pip install streamlit` with venv active |
| `jupyter: command not found` | Run `pip install jupyter` with venv active |
| Slow first startup | Normal — TensorFlow loads the model into memory on first run |
| Python version error | Use Python 3.9 or 3.10 only. Check with `python --version` |

---

## Disclaimer

> This application provides **AI-assisted triage only**. Results do not constitute a certified veterinary diagnosis. Always consult a licensed veterinary officer for confirmation and treatment. The developers accept no liability for decisions made solely on the basis of this tool's output.

---

## Author

**Ashipunaz** · [github.com/Ashipunaz](https://github.com/Ashipunaz)