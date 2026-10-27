# StadiumIQ — container image for Docker-based hosts (Hugging Face Spaces, Render, Fly.io, ...).
# Synthetic data and models are generated at build time, so the container starts instantly.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

# XGBoost needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (required by Hugging Face Spaces, good practice elsewhere).
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/stadiumiq

COPY --chown=app:app requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
USER app
RUN python -m src.bootstrap

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')"
# Hosts inject $PORT (e.g. Render); Hugging Face Spaces uses 7860 (set PORT=7860 there).
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
