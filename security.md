# Security Audit Findings (cattle-disease-detection)

Audit scope: repository contents under `D:/cattle-disease-detection` with focus on secret exposure, API hardening, and input validation.

## High severity

### 1) Exposed API key committed to git history

- **Where**: `Untitled.ipynb`
- **What**: A Roboflow API key was found hardcoded in a notebook (redacted in this report).
- **Risk**: Anyone with repo access (or anyone the notebook is shared with) can use the key to access the Roboflow account/project, incur costs, exfiltrate data, or modify resources (depending on key privileges).
- **Fix (required)**:
  - **Revoke/rotate** the Roboflow key immediately in Roboflow.
  - Remove the key from the notebook (replace with an environment variable).
  - If this repo is public or the key was ever pushed, **rewrite git history** (e.g. `git filter-repo` / BFG) and force-push (coordinate with collaborators).
  - Add notebook outputs/temporary files to `.gitignore` where appropriate.

## Medium severity

### 2) Unauthenticated “admin” endpoint allows changing active model

- **Where**: `api/main.py` → `POST /models/set-active`
- **Risk**: Anyone on the internet can switch the active model version, which can:
  - cause incorrect/unstable behavior,
  - break client expectations,
  - enable abuse (flip-flopping versions as a DoS vector).
- **Fix** (choose one):
  - **Remove this endpoint in production**, and set `MODEL_VERSION` as a Render environment variable instead.
  - Or protect it with an **admin API key**:
    - Require a header like `X-Admin-Key` and compare to `ADMIN_API_KEY` stored in environment variables.
  - Also consider rate-limiting on this endpoint.

### 3) Overly permissive CORS configuration

- **Where**: `api/main.py` uses:
  - `allow_origins=["*"]`
  - `allow_credentials=True`
- **Risk**: With credentials enabled, wildcard origins are unsafe; browsers may block it inconsistently and it increases cross-site abuse risk.
- **Fix**:
  - Set `allow_origins` to an explicit list of your frontend domains (e.g. `https://your-frontend.com`).
  - If you truly need wildcard origins, set `allow_credentials=False` and use token-based auth for protected operations.

### 4) No request size limits / upload constraints (DoS risk)

- **Where**: `POST /predict` in `api/main.py` accepts `UploadFile` and reads the whole file into memory.
- **Risk**: Attackers can upload very large files or non-images to exhaust memory/CPU (TensorFlow inference + PIL decode) and degrade service.
- **Fix**:
  - Enforce a **max upload size** (at reverse proxy / Render and at app level).
  - Validate `content_type` and reject non-image types.
  - Consider limiting pixel dimensions after decode (e.g., reject images above a max resolution).
  - Add basic rate limiting (per IP) if this is internet-facing.

## Low severity / hardening recommendations

### 5) Model versioning map is static and file paths are on disk

- **Where**: `MODEL_PATHS` in `api/main.py`
- **Risk**: Minimal. However, missing model file errors may reveal internal file paths depending on error messages.
- **Fix**:
  - Keep error messages generic for clients; log the detailed path server-side only.

### 6) Notebooks in repo may include sensitive outputs

- **Where**: `Untitled.ipynb`, `training_model_improved.ipynb`
- **Risk**: Notebooks can accidentally capture tokens, local paths, dataset references, or secrets in output cells.
- **Fix**:
  - Clear outputs before committing or use tools to strip outputs in CI.
  - Add a policy: never commit credentials in notebooks; always use env vars.

## Suggested immediate action plan (recommended order)

1. **Rotate Roboflow key** and remove it from `Untitled.ipynb`; rewrite git history if the repo is/was public.
2. **Lock down** `POST /models/set-active` (remove or protect with `ADMIN_API_KEY`).
3. **Fix CORS** to explicit origins and align credentials usage.
4. Add **upload validation** (type, max size, max pixels) and basic rate limiting.

## Notes

- This audit did not find other common cloud keys (AWS/GCP/Stripe/etc.) in the scanned files besides the Roboflow key.
- If you want, I can provide a hardened `api/main.py` patch implementing:
  - `ADMIN_API_KEY` auth for `/models/set-active`
  - `MAX_UPLOAD_BYTES` guard
  - image content-type checks
  - CORS origin allowlist via env var

