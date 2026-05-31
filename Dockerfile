# ============================================================
# DOCKERFILE — API Prédiction VBG
# Image légère Python 3.11 slim
# ============================================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Landry — ENSPY Yaoundé <contact@enspy.cm>"
LABEL version="1.0.0"
LABEL description="API Prédiction des Violences Basées sur le Genre"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY app/ ./app/
COPY run.py .

# Copie du modèle (doit exister avant le build)
COPY model_vbg.pkl .
COPY model_metadata.json .

# Création du répertoire logs
RUN mkdir -p logs

# Exposition du port
EXPOSE 8000

# Vérification santé
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Commande de démarrage
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
