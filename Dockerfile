# Інтелектуальна система предиктивної діагностики — Streamlit-додаток у контейнері
# Збірка:  docker build -t predictive-maintenance-ai4i:2.0 .
# Запуск:  docker run --rm -p 8501:8501 predictive-maintenance-ai4i:2.0
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=Europe/Kyiv

# libgomp1 — OpenMP-рантайм, потрібний LightGBM; tzdata — коректний час у протоколах
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Спочатку залежності — шар кешується, доки не змінено requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Запуск від непривілейованого користувача
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
