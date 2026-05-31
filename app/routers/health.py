# ============================================================
# ROUTER — Health Check & Statut Système
# ============================================================

from fastapi import APIRouter
from datetime import datetime
from app.models.schemas import HealthResponse
from app.services.prediction_service import model_service
from app.config import settings

router = APIRouter(tags=["Système"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de l'état de l'API"
)
async def health_check() -> HealthResponse:
    """
    Endpoint de santé — vérifie que l'API et le modèle
    sont opérationnels.
    """
    return HealthResponse(
        status        = "OK" if model_service.est_charge else "DÉGRADÉ",
        version       = settings.API_VERSION,
        modele_charge = model_service.est_charge,
        timestamp     = datetime.now().isoformat(),
    )


@router.get("/", summary="Accueil", include_in_schema=False)
async def root():
    return {
        "message"     : "API Prédiction VBG — Violences Basées sur le Genre",
        "version"     : settings.API_VERSION,
        "docs"        : "/docs",
        "redoc"       : "/redoc",
        "health"      : "/health",
        "avertissement": (
            "⚠️ Usage académique et de recherche uniquement. "
            "Résultats à valider par des professionnels qualifiés."
        )
    }
