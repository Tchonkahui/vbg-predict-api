# ============================================================
# ROUTER — Endpoints de Prédiction VBG
# ============================================================

from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse
from loguru import logger
from datetime import datetime

from app.models.schemas import (
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput,
    ErrorResponse
)
from app.services.prediction_service import model_service

router = APIRouter(
    prefix="/predict",
    tags=["Prédiction VBG"],
    responses={
        422: {"model": ErrorResponse, "description": "Données invalides"},
        500: {"model": ErrorResponse, "description": "Erreur serveur"},
    }
)

# ----------------------------------------------------------
# AVERTISSEMENT ÉTHIQUE — Affiché sur chaque endpoint
# ----------------------------------------------------------

ETHICAL_WARNING = (
    "⚠️ USAGE RESPONSABLE UNIQUEMENT : "
    "Ce résultat est un indicateur statistique. "
    "Toute décision doit impliquer un professionnel qualifié. "
    "Interdit pour tout usage discriminatoire."
)


# ----------------------------------------------------------
# ENDPOINT 1 — Prédiction individuelle
# ----------------------------------------------------------

@router.post(
    "/individuelle",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Prédiction individuelle du risque de VBG",
    description=(
        "## Prédiction individuelle\n\n"
        "Calcule la probabilité de risque de violence domestique "
        "à partir du profil socio-démographique d'une personne.\n\n"
        "### ⚠️ Avertissements éthiques\n"
        "- Usage **académique et de recherche** uniquement\n"
        "- Ne **jamais** utiliser pour décisions automatisées\n"
        "- Validation par professionnel **obligatoire**\n"
        "- Consentement éclairé requis si données réelles\n\n"
        "### Interprétation des niveaux de risque\n"
        "| Niveau | Probabilité | Action recommandée |\n"
        "|--------|-------------|-------------------|\n"
        "| Faible | < 20% | Prévention générale |\n"
        "| Modéré | 20-40% | Suivi recommandé |\n"
        "| Élevé | 40-65% | Intervention prioritaire |\n"
        "| Très Élevé | > 65% | Intervention immédiate |"
    )
)
async def prediction_individuelle(
    input_data: PredictionInput,
    request: Request
) -> PredictionOutput:
    """
    Endpoint de prédiction individuelle.

    Retourne :
    - Prédiction binaire (0/1)
    - Probabilité de risque
    - Niveau de risque catégorisé
    - Facteurs de risque identifiés
    - Recommandation professionnelle
    - Avertissements éthiques
    """
    logger.info(
        f"[PREDICT-INDIVIDUELLE] "
        f"IP={request.client.host} | "
        f"âge={input_data.age} | "
        f"emploi={input_data.employment}"
    )

    try:
        result = model_service.predire(input_data)
        logger.success(
            f"[PREDICT-INDIVIDUELLE] OK — "
            f"niveau={result.niveau_risque} | "
            f"proba={result.probabilite_risque}"
        )
        return result

    except RuntimeError as e:
        logger.error(f"[PREDICT-INDIVIDUELLE] Modèle non chargé : {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modèle non disponible : {str(e)}"
        )
    except ValueError as e:
        logger.warning(f"[PREDICT-INDIVIDUELLE] Données invalides : {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Données invalides : {str(e)}"
        )
    except Exception as e:
        logger.error(f"[PREDICT-INDIVIDUELLE] Erreur interne : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur. Contactez l'administrateur."
        )


# ----------------------------------------------------------
# ENDPOINT 2 — Prédiction par lot
# ----------------------------------------------------------

@router.post(
    "/batch",
    response_model=BatchPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Prédiction par lot (usage agrégé uniquement)",
    description=(
        "## Prédiction par lot\n\n"
        "Calcule les probabilités de risque pour un groupe d'observations.\n\n"
        "### ⚠️ Usage agrégé uniquement\n"
        "Les résultats batch sont destinés à des analyses statistiques "
        "de groupe, **jamais** à un ciblage individuel automatique.\n\n"
        "- Maximum **100 observations** par requête\n"
        "- Résultats à interpréter au niveau collectif\n"
        "- Taux global calculé sur l'ensemble du batch"
    )
)
async def prediction_batch(
    batch_input: BatchPredictionInput,
    request: Request
) -> BatchPredictionOutput:
    """Endpoint de prédiction par lot — usage agrégé et statistique."""

    logger.info(
        f"[PREDICT-BATCH] "
        f"IP={request.client.host} | "
        f"n={len(batch_input.observations)}"
    )

    try:
        result = model_service.predire_batch(batch_input)
        logger.success(
            f"[PREDICT-BATCH] OK — "
            f"{result.n_risque_detecte}/{result.n_observations} risques | "
            f"taux={result.taux_risque_global}"
        )
        return result

    except Exception as e:
        logger.error(f"[PREDICT-BATCH] Erreur : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ----------------------------------------------------------
# ENDPOINT 3 — Informations sur le modèle
# ----------------------------------------------------------

@router.get(
    "/info-modele",
    summary="Informations et limites du modèle",
    description="Retourne les métadonnées, performances et limites du modèle"
)
async def info_modele():
    """Retourne les informations complètes sur le modèle déployé."""
    return {
        "modele"      : model_service.metadata,
        "avertissement": ETHICAL_WARNING,
        "limites"     : [
            "Entraîné sur 347 observations — généralisation limitée",
            "Dataset non représentatif de la population camerounaise",
            "Variables manquantes : alcool, antécédents familiaux, réseau social",
            "Biais de déclaration : sous-déclaration probable des violences",
            "Données transversales : pas de causalité établie",
            "Performances variables selon les sous-groupes socio-démographiques",
        ],
        "usage_responsable": [
            "Résultats agrégés uniquement — jamais de ciblage individuel automatique",
            "Validation humaine obligatoire avant toute décision",
            "Consentement éclairé requis si données individuelles réelles",
            "Audit des biais recommandé tous les 6 mois",
            "Ne pas utiliser dans des contextes juridiques ou pénaux",
        ],
        "timestamp"   : datetime.now().isoformat(),
    }
