# ============================================================
# SCHÉMAS PYDANTIC — Validation des Données
# Entrées, Sorties et Erreurs de l'API VBG
# ============================================================

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
from enum import Enum
from datetime import datetime


# ============================================================
# ÉNUMÉRATIONS — Valeurs autorisées
# ============================================================

class EducationLevel(str, Enum):
    NO_EDUCATION = "no education"
    PRIMARY      = "primary"
    SECONDARY    = "secondary"
    TERTIARY     = "tertiary"


class EmploymentStatus(str, Enum):
    EMPLOYED      = "employed"
    SELF_EMPLOYED = "self-employed"
    UNEMPLOYED    = "unemployed"
    STUDENT       = "student"


class MaritalStatus(str, Enum):
    MARRIED   = "married"
    UNMARRIED = "unmarried"


# ============================================================
# SCHÉMA D'ENTRÉE — Prédiction individuelle
# ============================================================

class PredictionInput(BaseModel):
    """
    Données socio-démographiques pour la prédiction du risque de VBG.

    ⚠️ AVERTISSEMENT : Ces données sont sensibles.
    Leur collecte et traitement doivent respecter le consentement
    éclairé de la personne concernée.
    """

    age: int = Field(
        ...,
        ge=15,
        le=80,
        description="Âge de la personne (15-80 ans)",
        example=28
    )

    education: EducationLevel = Field(
        ...,
        description="Niveau d'éducation",
        example="secondary"
    )

    employment: EmploymentStatus = Field(
        ...,
        description="Statut professionnel actuel",
        example="unemployed"
    )

    income: float = Field(
        ...,
        ge=0,
        description="Revenu mensuel en unité locale (0 si sans revenu)",
        example=0.0
    )

    marital_status: MaritalStatus = Field(
        ...,
        description="Statut matrimonial",
        example="married"
    )

    # Champ optionnel pour contexte
    identifiant: Optional[str] = Field(
        default=None,
        description="Identifiant anonymisé (facultatif, pour suivi)",
        example="CASE_001"
    )

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 15:
            raise ValueError(
                "L'âge minimum est 15 ans (population cible de l'étude)"
            )
        if v > 80:
            raise ValueError(
                "L'âge maximum est 80 ans (hors plage du modèle)"
            )
        return v

    @field_validator('income')
    @classmethod
    def validate_income(cls, v):
        if v < 0:
            raise ValueError("Le revenu ne peut pas être négatif")
        if v > 1_000_000:
            raise ValueError(
                "Revenu hors plage du modèle (max: 1,000,000)"
            )
        return v

    @model_validator(mode='after')
    def check_consistency(self):
        """Vérification de la cohérence des données"""
        if (self.employment == EmploymentStatus.STUDENT
                and self.age > 40):
            pass  # Avertissement non bloquant — étudiant > 40 ans possible
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "age"           : 27,
                "education"     : "primary",
                "employment"    : "unemployed",
                "income"        : 0.0,
                "marital_status": "married",
                "identifiant"   : "CASE_001"
            }
        }


# ============================================================
# SCHÉMA D'ENTRÉE — Prédiction par lot (batch)
# ============================================================

class BatchPredictionInput(BaseModel):
    """Prédictions multiples — usage agrégé et statistique uniquement"""

    observations: List[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Liste d'observations (max 100 par requête)"
    )

    @field_validator('observations')
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError(
                "Taille maximale d'un batch : 100 observations. "
                "Découpez votre requête en plusieurs appels."
            )
        return v


# ============================================================
# SCHÉMA DE SORTIE — Réponse prédiction individuelle
# ============================================================

class RiskLevel(str, Enum):
    FAIBLE    = "Faible"
    MODERE    = "Modéré"
    ELEVE     = "Élevé"
    TRES_ELEVE= "Très Élevé"


class PredictionOutput(BaseModel):
    """
    Résultat de prédiction avec avertissements éthiques intégrés.
    """

    # Identifiant
    identifiant         : Optional[str] = None

    # Prédiction
    prediction          : int   = Field(
        ..., description="0 = risque non détecté | 1 = risque détecté"
    )
    probabilite_risque  : float = Field(
        ..., description="Probabilité de risque (0.0 à 1.0)"
    )
    niveau_risque       : RiskLevel = Field(
        ..., description="Niveau de risque catégorisé"
    )
    pourcentage_risque  : str   = Field(
        ..., description="Probabilité en pourcentage"
    )

    # Métadonnées
    timestamp           : str   = Field(
        ..., description="Horodatage de la prédiction"
    )
    version_modele      : str   = Field(
        ..., description="Version du modèle utilisé"
    )

    # Avertissements éthiques — toujours présents
    avertissement_ethique: str = Field(
        default=(
            "⚠️ Cette prédiction est un indicateur statistique, "
            "pas un diagnostic. Elle doit être interprétée par "
            "un professionnel qualifié et ne peut pas fonder "
            "seule une décision."
        )
    )
    recommandation       : str = Field(
        default="Consulter un travailleur social ou professionnel de santé"
    )

    # Facteurs de risque détectés
    facteurs_risque      : List[str] = Field(
        default_factory=list,
        description="Facteurs de risque identifiés dans le profil"
    )


# ============================================================
# SCHÉMA DE SORTIE — Réponse batch
# ============================================================

class BatchPredictionOutput(BaseModel):
    """Résultats agrégés pour prédiction par lot"""

    n_observations      : int
    n_risque_detecte    : int
    taux_risque_global  : str
    predictions         : List[PredictionOutput]
    timestamp           : str
    avertissement_global: str = (
        "⚠️ Ces résultats sont à usage AGRÉGÉ uniquement. "
        "Aucun ciblage individuel automatique n'est autorisé."
    )


# ============================================================
# SCHÉMA DE SANTÉ — Health Check
# ============================================================

class HealthResponse(BaseModel):
    status          : str
    version         : str
    modele_charge   : bool
    timestamp       : str
    avertissement   : str = (
        "API à usage académique et de recherche. "
        "Déploiement en production requiert validation éthique."
    )


# ============================================================
# SCHÉMA D'ERREUR
# ============================================================

class ErrorResponse(BaseModel):
    code        : int
    message     : str
    detail      : Optional[str] = None
    timestamp   : str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
