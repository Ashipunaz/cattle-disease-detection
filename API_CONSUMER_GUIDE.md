# Cattle Disease Detection API – Consumer Guide

This document explains how to consume the deployed Cattle Disease Detection API from frontend or backend clients.

---

## Base URL

Production base URL:

- `https://cattle-disease-detection-q8x2.onrender.com`

Interactive documentation (Swagger UI):

- `GET /docs` → `https://cattle-disease-detection-q8x2.onrender.com/docs`

---

## 1. Health and Metadata

### `GET /`

Returns basic metadata and links to docs and health.

### `GET /health`

Use this for uptime checks and version info.

**Sample response:**

```json
{
  "status": "ok",
  "active_model_version": "v1",
  "available_versions": ["v1"],
  "build": "78c54442",
  "rate_limit_per_minute": 60
}
```

---

## 2. Model Versions

### `GET /models`

Lists available model versions and which one is currently active.

**Sample response:**

```json
{
  "active_model_version": "v1",
  "available_versions": {
    "v1": "cattle_disease_model.h5"
  }
}
```

### `POST /models/set-active` (admin only)

Changes the active model version (for example, from `v1` to `v2`).

- This endpoint is **protected** by an admin key.
- It is only available if the server is configured with `ADMIN_API_KEY`.
- Requests must include header: `X-Admin-Key: <ADMIN_API_KEY>`.

**Example:**

```bash
curl -X POST "https://cattle-disease-detection-q8x2.onrender.com/models/set-active?version=v1" \
  -H "X-Admin-Key: <ADMIN_API_KEY>"
```

**Sample response:**

```json
{ "active_model_version": "v1" }
```

If `ADMIN_API_KEY` is not set, this route returns `404` and should be treated as disabled.

---

## 3. Predict Disease From an Image

### `POST /predict`

Classifies a single cattle image.

**Request:**

- Method: `POST`
- URL: `https://cattle-disease-detection-q8x2.onrender.com/predict`
- Headers:
  - `Accept: application/json`
  - `Content-Type: multipart/form-data`
- Body (multipart form-data):
  - `file`: image file (JPEG, PNG, or WebP).
- Optional query parameter:
  - `version`: model version to use (e.g. `v1`). If omitted, the active version is used.

**cURL example:**

```bash
curl -X POST "https://cattle-disease-detection-q8x2.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/cattle_photo.jpg;type=image/jpeg"
```

**Sample response:**

```json
{
  "model_version": "v1",
  "predicted_class": "lumpy skin",
  "confidence": 99,
  "probabilities": {
    "fmd": 0.0000053,
    "healthy": 0.0000677,
    "lumpy skin": 0.9999043,
    "mastitis": 0.0000227
  },
  "info": {
    "full_name": "Lumpy Skin Disease",
    "what_you_see": "Round raised lumps or nodules appearing across the skin. The animal may have a fever and reduced milk output.",
    "what_to_do": "Separate the animal from the herd immediately. LSD spreads through insect bites — treat the whole herd with insect repellent. Contact your vet to vaccinate at-risk animals.",
    "urgency": "Isolate today. Protect the herd.",
    "severity": "Urgent",
    "requires_vet": true
  }
}
```

**Notes for applications:**

- `confidence` is an **integer percentage** (e.g. `99` → “99% sure”).
- `probabilities` are raw probabilities (approximate sum = 1.0).
- `info` contains the text your UI should surface to end users.
- For batch analysis (1–5 images), call `/predict` multiple times and aggregate the results in the client; you can generate PDFs on the frontend if needed.

---

## 4. Limits and Errors

Server-side protections are applied for robustness:

- **Max file size**: `MAX_UPLOAD_BYTES` (default ~5 MB).
- **Max image resolution**: `MAX_IMAGE_PIXELS` (default ~20 megapixels).
- **Allowed media types**: `image/jpeg`, `image/png`, `image/webp`.
- **Rate limiting**: `RATE_LIMIT_PER_MINUTE` requests per minute per IP (best-effort).

Common error codes:

- `400 Bad Request` – invalid or corrupted image.
- `413 Payload Too Large` – file or image too large.
- `415 Unsupported Media Type` – unsupported content type.
- `429 Too Many Requests` – rate limit exceeded; try again after the indicated delay.
- `500 Internal Server Error` – unexpected issue (e.g. missing model file).

---

## 5. Typical Flow (1–5 Images)

1. User selects up to 5 image files.
2. For each image:
   - Send `POST /predict` with that image as `file`.
   - Collect `predicted_class`, `confidence`, `probabilities`, and `info`.
3. Display results in your UI:
   - Show the predicted disease and confidence.
   - Show the recommended actions and urgency text from `info`.
4. (Optional) Build a summary PDF on the frontend (for example, using a JavaScript PDF library) using the returned data.



