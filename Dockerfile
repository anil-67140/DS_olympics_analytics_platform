# Reproducible runtime for the API and dashboard.
#   docker build -t olympics-platform .
#   docker run -p 8000:8000 -v $PWD/workspace:/app/workspace olympics-platform            # API
#   docker run -p 7860:7860 -v $PWD/workspace:/app/workspace olympics-platform dashboard   # dashboard
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OLYMPICS_ROOT=/app/workspace PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install dependencies first (cached layer). Training-only extras are skipped, but the boosting libraries stay:
# the champion model may be XGBoost/LightGBM/CatBoost and must be loadable here.
COPY requirements.txt .
RUN grep -v -E "^(shap|evidently|kaggle|mlflow)" requirements.txt > requirements-serve.txt \
    && pip install -r requirements-serve.txt

COPY params.yaml pyproject.toml ./
COPY src ./src
COPY apps ./apps
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh && useradd -m app && mkdir -p /app/workspace && chown -R app /app
USER app

EXPOSE 8000 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health' if os.environ.get('SERVICE','api')=='api' else f'http://127.0.0.1:{p}/').status==200 else 1)"

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["api"]
