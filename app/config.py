# ============================================================
# CONFIGURATION CENTRALISÉE — API VBG
# ============================================================

from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Paramètres globaux de l'application"""

    # Informations API
    API_TITLE       : str = "API Prédiction — Violences Basées sur le Genre"
    API_DESCRIPTION : str = (
        "## ⚠️ Avertissement Éthique\n\n"
        "Cette API est un **outil d'aide à la décision uniquement**.\n\n"
        "- Les prédictions ne remplacent **jamais** l'évaluation humaine\n"
        "- Résultats à usage **collectif et agrégé** uniquement\n"
        "- Tout usage discriminatoire ou stigmatisant est **strictement interdit**\n"
        "- Validation obligatoire par un professionnel qualifié\n\n"
        "---\n\n"
        "**Développé par** : Landry — ENSPY Yaoundé, Cameroun\n\n"
        "**Contact** : [email de contact]\n\n"
        "**Version** : 1.0.0"
    )
    API_VERSION     : str = "1.0.0"
    API_PREFIX      : str = "/api/v1"

    # Serveur
    HOST            : str = "0.0.0.0"
    PORT            : int = 8000
    DEBUG           : bool = False
    RELOAD          : bool = False

    # Modèle
    MODEL_PATH      : Path = BASE_DIR / "model_vbg.pkl"
    METADATA_PATH   : Path = BASE_DIR / "model_metadata.json"

    # Logs
    LOG_DIR         : Path = BASE_DIR / "logs"
    LOG_LEVEL       : str = "INFO"
    LOG_FILE        : str = "api_vbg.log"

    # Sécurité & Rate limiting
    MAX_BATCH_SIZE  : int = 100
    RATE_LIMIT      : int = 60   # requêtes par minute

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
