# ── BASE IMAGE ───────────────────────────────────────────────────────────────
# We start from an official Python image, not a blank Linux image.
# python:3.12-slim is the minimal Python image — it has Python installed
# but strips out everything else to keep the image small (~130MB vs ~900MB
# for the full image). Smaller image = faster to build, push, and pull.
FROM python:3.12-slim

# ── WORKING DIRECTORY ────────────────────────────────────────────────────────
# Sets the working directory inside the container.
# All subsequent commands (COPY, RUN, CMD) execute relative to this path.
# /app is a convention — nothing special about the name.
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=300
# ── INSTALL DEPENDENCIES FIRST ───────────────────────────────────────────────
# We copy requirements.txt and install BEFORE copying the rest of the code.
# Why this order matters: Docker builds in layers and caches each layer.
# If we copied all code first, then requirements.txt, any code change would
# invalidate the cache and force a full pip install every time.
# By copying requirements.txt first, pip install is only re-run when
# requirements.txt actually changes — not on every code edit.
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# ── COPY APPLICATION CODE ────────────────────────────────────────────────────
# Copy source code and the files the API needs at runtime.
# We copy only what's needed — not data/, not .venv/, not mlruns/.
# .dockerignore handles exclusions (like .gitignore but for Docker).
COPY src/api/main.py .
COPY models/champion.pkl     models/champion.pkl
COPY models/preprocessor.pkl models/preprocessor.pkl
COPY models/metadata.json    models/metadata.json

# ── EXPOSE PORT ───────────────────────────────────────────────────────────────
# Tells Docker that this container listens on port 8000.
# This is documentation — it doesn't actually open the port.
# The actual port binding happens at runtime with -p 8000:8000.
EXPOSE 8000

# ── START COMMAND ────────────────────────────────────────────────────────────
# CMD is what runs when the container starts.
# uvicorn is the ASGI server that serves FastAPI.
# main:app means "in main.py, find the object called app"
# --host 0.0.0.0 means accept connections from any IP, not just localhost.
#   Without this, the API would only be reachable from inside the container.
# --port 8000 matches the EXPOSE above.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]