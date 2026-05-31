# ============================================================
# APPLICATION FASTAPI PRINCIPALE — API VBG
# Prédiction des Violences Basées sur le Genre
# ============================================================

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.routers import prediction, health
from app.services.prediction_service import model_service
from app.middleware.logging_middleware import LoggingMiddleware

# ============================================================
# CONFIGURATION DES LOGS
# ============================================================

settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()  # Supprime le handler par défaut

# Log console — format coloré
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level=settings.LOG_LEVEL,
    colorize=True,
)

# Log fichier — format JSON pour parsing
logger.add(
    settings.LOG_DIR / settings.LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    level=settings.LOG_LEVEL,
    rotation="10 MB",     # Nouveau fichier tous les 10 Mo
    retention="30 days",  # Conservation 30 jours
    compression="zip",    # Compression des anciens logs
)


# ============================================================
# LIFESPAN — Démarrage et Arrêt
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application.
    Charge le modèle au démarrage, libère les ressources à l'arrêt.
    """
    # Démarrage
    logger.info("=" * 60)
    logger.info("DÉMARRAGE — API Prédiction VBG")
    logger.info(f"Version : {settings.API_VERSION}")
    logger.info(f"Host    : {settings.HOST}:{settings.PORT}")
    logger.info("=" * 60)

    try:
        model_service.charger_modele()
        logger.success("API prête à recevoir des requêtes")
    except Exception as e:
        logger.error(f"Échec du chargement du modèle : {e}")
        logger.warning(
            "L'API démarre sans modèle — "
            "les endpoints de prédiction retourneront 503"
        )

    yield  # Application en cours d'exécution

    # Arrêt
    logger.info("ARRÊT — Nettoyage des ressources")
    logger.info("API arrêtée proprement")


# ============================================================
# CRÉATION DE L'APPLICATION
# ============================================================

app = FastAPI(
    title       = settings.API_TITLE,
    description = settings.API_DESCRIPTION,
    version     = settings.API_VERSION,
    lifespan    = lifespan,

    # Documentation
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    openapi_url = "/openapi.json",

    # Métadonnées
    contact={
        "name" : "Landry — ENSPY Yaoundé",
        "email": "contact@enspy.cm",
    },
    license_info={
        "name": "Usage Académique — ENSPY 2024",
    },
)

# ============================================================
# MIDDLEWARES
# ============================================================

# Logging automatique
app.add_middleware(LoggingMiddleware)

# CORS — Autoriser les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ============================================================
# ROUTERS
# ============================================================

app.include_router(health.router)
app.include_router(
    prediction.router,
    prefix=settings.API_PREFIX
)

# ============================================================
# GESTIONNAIRE D'ERREURS GLOBAL
# ============================================================
@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse("dashboard.html")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Erreur non gérée : {type(exc).__name__} — {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "code"   : 500,
            "message": "Erreur interne du serveur",
            "detail" : str(exc),
        }
    )
