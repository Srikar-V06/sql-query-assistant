# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Hugging Face Spaces runs as a non-root user; set a predictable home
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create the HF-expected non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY --chown=user app.py         .
COPY --chown=user environment.py .
COPY --chown=user database.py    .
COPY --chown=user graders.py     .
COPY --chown=user models.py      .
COPY --chown=user tasks.py       .
COPY --chown=user inference.py   .
COPY --chown=user openenv.yaml   .

# ── Switch to non-root user ────────────────────────────────────────────────────
USER user

# ── Expose port (HF Spaces expects 7860) ──────────────────────────────────────
EXPOSE 7860

# ── Launch ────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]