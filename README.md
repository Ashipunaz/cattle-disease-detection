---
title: Cattle Health Check
emoji: 🐄
colorFrom: green
colorTo: darkgreen
sdk: streamlit
sdk_version: "1.26.0"
python_version: "3.10"
app_file: main.py
pinned: false
---

# 🐄 Cattle Health Check

**Cattle Health Check** is a Streamlit app that uses AI to detect common cattle diseases from uploaded photos. It currently supports:

- **Foot and Mouth Disease (FMD)** 🟠
- **Lumpy Skin Disease (LSD)** 🔴
- **Mastitis** 🟡
- **Healthy Cattle** 🟢

## Features

1. Upload up to 5 photos of your cattle at a time.
2. Get instant predictions with confidence percentages.
3. Detailed guidance on what to do next, including vet advice.
4. Download a PDF report for each session.
5. Session history to track past checks.

## How It Works

1. Upload clear photos of your animals.
2. The AI model predicts the condition of each animal.
3. You see confidence levels, disease severity, and recommended actions.
4. Download the PDF report for record-keeping or sharing with your vet.

## Technical Details

- **Framework:** Streamlit
- **Model:** TensorFlow CNN trained on 2,800 cattle images.
- **Input size:** 224x224 pixels
- **Max images per check:** 5
- **Disease classes:** FMD, Lumpy Skin, Mastitis, Healthy
- **PDF generation:** FPDF

## Notes

- The tool is AI-assisted only and does **not replace a licensed veterinary officer**.
- Photos are not stored or shared; session history is kept only while the browser is open.

Check the configuration reference at [Hugging Face Spaces Config](https://huggingface.co/docs/hub/spaces-config-reference)