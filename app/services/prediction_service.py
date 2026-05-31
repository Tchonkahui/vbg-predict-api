# ============================================================
# SERVICE DE PRÉDICTION — Logique Métier
# Chargement modèle, feature engineering, prédiction
# ============================================================

import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from loguru import logger

from app.config import settings
from app.models.schemas import (
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput,
    RiskLevel
)


class ModelService:
    """
    Service central de prédiction VBG.
    Encapsule le chargement du modèle, le feature engineering
    et la génération des prédictions avec avertissements éthiques.
    """

    def __init__(self):
        self.model    = None
        self.metadata = {}
        self._charge  = False

    # ----------------------------------------------------------
    # CHARGEMENT DU MODÈLE
    # ----------------------------------------------------------

    def charger_modele(self) -> None:
        """
        Charge le modèle ML et ses métadonnées au démarrage de l'API.
        Lève une exception si le modèle est introuvable.
        """
        try:
            logger.info(f"Chargement du modèle depuis : {settings.MODEL_PATH}")

            if not settings.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Modèle introuvable : {settings.MODEL_PATH}\n"
                    "Placez le fichier model_vbg.pkl dans le répertoire racine."
                )

            self.model = joblib.load(settings.MODEL_PATH)
            logger.success("Modèle ML chargé avec succès")

            if settings.METADATA_PATH.exists():
                with open(settings.METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"Métadonnées chargées — "
                    f"Modèle : {self.metadata.get('nom_modele', 'inconnu')} | "
                    f"Version : {self.metadata.get('version', '?')}"
                )
            else:
                logger.warning(
                    "Fichier de métadonnées introuvable — "
                    "valeurs par défaut utilisées"
                )
                self.metadata = {
                    "nom_modele": "Modèle VBG",
                    "version"   : "1.0.0"
                }

            self._charge = True

        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")
            raise

    @property
    def est_charge(self) -> bool:
        return self._charge

    # ----------------------------------------------------------
    # FEATURE ENGINEERING — Réplication du pipeline EDA
    # ----------------------------------------------------------

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le même feature engineering que lors de l'entraînement.
        CRITIQUE : doit être identique au code de preprocessing.
        """
        df = df.copy()

        # Nettoyage
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip().str.lower()

        # Correction orthographe connue
        df['marital_status'] = df['marital_status'].replace(
            'unmarred', 'unmarried'
        )

        # Variables dérivées income
        df['has_income']  = (df['income'] > 0).astype(int)
        df['income_log']  = np.log1p(df['income'])

        # Groupes d'âge
        df['age_group'] = pd.cut(
            df['age'],
            bins=[15, 25, 35, 45, 100],
            labels=['15-24', '25-34', '35-44', '45-60'],
            include_lowest=True
        ).astype(str)

        # Score de vulnérabilité
        edu_risk = {
            'no education': 2, 'primary': 1,
            'secondary': 0,    'tertiary': 0
        }
        df['edu_risk']  = df['education'].map(edu_risk).fillna(0)
        df['emp_risk']  = (df['employment'] == 'unemployed').astype(int)
        df['mar_risk']  = (df['marital_status'] == 'married').astype(int)
        df['inc_risk']  = (df['income'] == 0).astype(int)
        df['age_risk']  = df['age'].apply(lambda x: 1 if 25<=x<=35 else 0)

        df['vulnerability_score'] = (
            df['edu_risk'] + df['emp_risk'] + df['mar_risk']
            + df['inc_risk'] + df['age_risk']
        )

        # Interaction
        df['edu_x_emp'] = df['education'] + '_' + df['employment']

        return df

    def _identifier_facteurs_risque(
        self, row: pd.Series
    ) -> List[str]:
        """
        Identifie les facteurs de risque présents dans un profil.
        Retournés en langage clair pour les professionnels.
        """
        facteurs = []

        if row.get('employment') == 'unemployed':
            facteurs.append(
                "Sans emploi — dépendance économique (facteur principal)"
            )
        if row.get('income', 0) == 0:
            facteurs.append("Absence de revenu propre")
        if row.get('marital_status') == 'married':
            facteurs.append("Statut marital : mariée (cohabitation avec partenaire)")
        if row.get('education') in ['no education', 'primary']:
            facteurs.append(
                f"Niveau d'éducation faible : {row.get('education')}"
            )
        if 25 <= row.get('age', 0) <= 35:
            facteurs.append(
                f"Tranche d'âge à risque : {row.get('age')} ans (25-35 ans)"
            )

        return facteurs

    def _categoriser_risque(self, proba: float) -> Tuple[RiskLevel, str]:
        """
        Catégorise le niveau de risque et génère une recommandation.
        Seuils définis selon le contexte VBG et l'usage décideurs.
        """
        if proba < 0.20:
            return (
                RiskLevel.FAIBLE,
                "Prévention générale — sensibilisation aux ressources disponibles"
            )
        elif proba < 0.40:
            return (
                RiskLevel.MODERE,
                "Suivi recommandé — orienter vers services sociaux de proximité"
            )
        elif proba < 0.65:
            return (
                RiskLevel.ELEVE,
                "Intervention prioritaire — évaluation par travailleur social requise"
            )
        else:
            return (
                RiskLevel.TRES_ELEVE,
                "Intervention immédiate — référer vers structures de protection spécialisées"
            )

    # ----------------------------------------------------------
    # PRÉDICTION INDIVIDUELLE
    # ----------------------------------------------------------

    def predire(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Génère une prédiction individuelle avec tous les avertissements.
        """
        if not self._charge:
            raise RuntimeError(
                "Modèle non chargé. Appelez charger_modele() d'abord."
            )

        logger.info(
            f"Prédiction individuelle — "
            f"âge={input_data.age} | "
            f"emploi={input_data.employment} | "
            f"id={input_data.identifiant}"
        )

        try:
            # Construction du DataFrame
            df_input = pd.DataFrame([{
                'age'           : input_data.age,
                'education'     : input_data.education.value,
                'employment'    : input_data.employment.value,
                'income'        : input_data.income,
                'marital_status': input_data.marital_status.value,
            }])

            # Feature engineering
            df_fe = self._feature_engineering(df_input)

            # Colonnes attendues par le modèle
            features = [
                'age', 'income_log', 'vulnerability_score',
                'has_income', 'emp_risk', 'mar_risk',
                'inc_risk', 'age_risk', 'edu_risk',
                'education', 'employment', 'marital_status',
                'age_group', 'edu_x_emp'
            ]
            X = df_fe[features]

            # Prédiction
            prediction = int(self.model.predict(X)[0])
            proba      = float(self.model.predict_proba(X)[0][1])

            # Catégorisation et recommandation
            niveau_risque, recommandation = self._categoriser_risque(proba)

            # Facteurs de risque
            facteurs = self._identifier_facteurs_risque(df_input.iloc[0])

            logger.info(
                f"Prédiction terminée — "
                f"proba={proba:.3f} | niveau={niveau_risque}"
            )

            return PredictionOutput(
                identifiant         = input_data.identifiant,
                prediction          = prediction,
                probabilite_risque  = round(proba, 4),
                niveau_risque       = niveau_risque,
                pourcentage_risque  = f"{proba*100:.1f}%",
                timestamp           = datetime.now().isoformat(),
                version_modele      = self.metadata.get("version", "1.0.0"),
                recommandation      = recommandation,
                facteurs_risque     = facteurs,
            )

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction : {e}")
            raise

    # ----------------------------------------------------------
    # PRÉDICTION PAR LOT
    # ----------------------------------------------------------

    def predire_batch(
        self, batch_input: BatchPredictionInput
    ) -> BatchPredictionOutput:
        """
        Prédictions sur plusieurs observations.
        Usage agrégé uniquement — rapport de groupe.
        """
        logger.info(
            f"Prédiction batch — "
            f"{len(batch_input.observations)} observations"
        )

        predictions = [
            self.predire(obs)
            for obs in batch_input.observations
        ]

        n_risque = sum(p.prediction for p in predictions)
        taux     = n_risque / len(predictions) * 100

        logger.info(
            f"Batch terminé — "
            f"{n_risque}/{len(predictions)} risques détectés "
            f"({taux:.1f}%)"
        )

        return BatchPredictionOutput(
            n_observations     = len(predictions),
            n_risque_detecte   = n_risque,
            taux_risque_global = f"{taux:.1f}%",
            predictions        = predictions,
            timestamp          = datetime.now().isoformat(),
        )


# Instance singleton — partagée dans toute l'application
model_service = ModelService()
