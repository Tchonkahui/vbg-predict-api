# ============================================================
# TESTS — API Prédiction VBG
# Couverture : endpoints, validation, cas limites, éthique
# Usage : pytest tests/ -v
# ============================================================

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Ajout du chemin racine
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)

# ============================================================
# DONNÉES DE TEST
# ============================================================

PROFIL_VALIDE = {
    "age"           : 27,
    "education"     : "primary",
    "employment"    : "unemployed",
    "income"        : 0.0,
    "marital_status": "married",
    "identifiant"   : "TEST_001"
}

PROFIL_RISQUE_FAIBLE = {
    "age"           : 45,
    "education"     : "tertiary",
    "employment"    : "employed",
    "income"        : 5000.0,
    "marital_status": "unmarried",
}

PROFIL_INVALIDE_AGE = {
    "age"           : 5,  # Trop jeune
    "education"     : "primary",
    "employment"    : "unemployed",
    "income"        : 0.0,
    "marital_status": "married",
}

PROFIL_INVALIDE_EDUCATION = {
    "age"           : 25,
    "education"     : "phd",  # Valeur non autorisée
    "employment"    : "unemployed",
    "income"        : 0.0,
    "marital_status": "married",
}

PROFIL_REVENU_NEGATIF = {
    "age"           : 25,
    "education"     : "primary",
    "employment"    : "unemployed",
    "income"        : -100.0,   # Invalide
    "marital_status": "married",
}


# ============================================================
# TESTS — HEALTH CHECK
# ============================================================

class TestHealthCheck:

    def test_health_endpoint_existe(self):
        """L'endpoint /health doit être accessible"""
        response = client.get("/health")
        assert response.status_code in [200, 503]

    def test_health_retourne_json(self):
        """La réponse doit être du JSON valide"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data

    def test_root_endpoint(self):
        """L'endpoint racine doit retourner les infos de base"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "avertissement" in data


# ============================================================
# TESTS — VALIDATION DES ENTRÉES
# ============================================================

class TestValidationEntrees:

    def test_age_trop_jeune_rejete(self):
        """Un âge < 15 ans doit être rejeté"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_INVALIDE_AGE
        )
        assert response.status_code == 422

    def test_education_invalide_rejetee(self):
        """Une valeur d'éducation non autorisée doit être rejetée"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_INVALIDE_EDUCATION
        )
        assert response.status_code == 422

    def test_revenu_negatif_rejete(self):
        """Un revenu négatif doit être rejeté"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_REVENU_NEGATIF
        )
        assert response.status_code == 422

    def test_champs_obligatoires(self):
        """Une requête vide doit être rejetée"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json={}
        )
        assert response.status_code == 422

    def test_age_limite_superieure(self):
        """Un âge > 80 ans doit être rejeté"""
        profil = PROFIL_VALIDE.copy()
        profil["age"] = 99
        response = client.post(
            "/api/v1/predict/individuelle",
            json=profil
        )
        assert response.status_code == 422

    def test_tous_niveaux_education_valides(self):
        """Tous les niveaux d'éducation valides doivent être acceptés"""
        niveaux = ["no education", "primary", "secondary", "tertiary"]
        for niveau in niveaux:
            profil = PROFIL_VALIDE.copy()
            profil["education"] = niveau
            response = client.post(
                "/api/v1/predict/individuelle",
                json=profil
            )
            # 200 si modèle chargé, 503 si pas de modèle — les deux sont OK
            assert response.status_code in [200, 503], (
                f"Échec pour éducation='{niveau}' : {response.status_code}"
            )

    def test_tous_statuts_emploi_valides(self):
        """Tous les statuts d'emploi valides doivent être acceptés"""
        statuts = ["employed", "self-employed", "unemployed", "student"]
        for statut in statuts:
            profil = PROFIL_VALIDE.copy()
            profil["employment"] = statut
            response = client.post(
                "/api/v1/predict/individuelle",
                json=profil
            )
            assert response.status_code in [200, 503]


# ============================================================
# TESTS — STRUCTURE DES RÉPONSES
# ============================================================

class TestStructureReponses:

    def test_reponse_contient_avertissement_ethique(self):
        """Toute réponse de prédiction doit contenir un avertissement"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_VALIDE
        )
        if response.status_code == 200:
            data = response.json()
            assert "avertissement_ethique" in data
            assert len(data["avertissement_ethique"]) > 0

    def test_reponse_contient_recommandation(self):
        """Toute prédiction doit inclure une recommandation professionnelle"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_VALIDE
        )
        if response.status_code == 200:
            data = response.json()
            assert "recommandation" in data
            assert len(data["recommandation"]) > 0

    def test_reponse_contient_facteurs_risque(self):
        """La réponse doit lister les facteurs de risque identifiés"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_VALIDE
        )
        if response.status_code == 200:
            data = response.json()
            assert "facteurs_risque" in data
            assert isinstance(data["facteurs_risque"], list)

    def test_probabilite_entre_0_et_1(self):
        """La probabilité prédite doit être entre 0 et 1"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_VALIDE
        )
        if response.status_code == 200:
            data = response.json()
            proba = data["probabilite_risque"]
            assert 0.0 <= proba <= 1.0

    def test_prediction_binaire(self):
        """La prédiction doit être 0 ou 1"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_VALIDE
        )
        if response.status_code == 200:
            data = response.json()
            assert data["prediction"] in [0, 1]

    def test_niveau_risque_valide(self):
        """Le niveau de risque doit être une valeur autorisée"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_VALIDE
        )
        if response.status_code == 200:
            data = response.json()
            niveaux_valides = ["Faible", "Modéré", "Élevé", "Très Élevé"]
            assert data["niveau_risque"] in niveaux_valides


# ============================================================
# TESTS — PRÉDICTION PAR LOT
# ============================================================

class TestBatchPrediction:

    def test_batch_valide(self):
        """Un batch valide doit être traité"""
        batch = {
            "observations": [PROFIL_VALIDE, PROFIL_RISQUE_FAIBLE]
        }
        response = client.post(
            "/api/v1/predict/batch",
            json=batch
        )
        assert response.status_code in [200, 503]

    def test_batch_vide_rejete(self):
        """Un batch vide doit être rejeté"""
        batch = {"observations": []}
        response = client.post(
            "/api/v1/predict/batch",
            json=batch
        )
        assert response.status_code == 422

    def test_batch_trop_grand_rejete(self):
        """Un batch de plus de 100 observations doit être rejeté"""
        batch = {
            "observations": [PROFIL_VALIDE] * 101
        }
        response = client.post(
            "/api/v1/predict/batch",
            json=batch
        )
        assert response.status_code == 422

    def test_batch_reponse_contient_avertissement_global(self):
        """La réponse batch doit contenir l'avertissement d'usage agrégé"""
        batch = {"observations": [PROFIL_VALIDE]}
        response = client.post(
            "/api/v1/predict/batch",
            json=batch
        )
        if response.status_code == 200:
            data = response.json()
            assert "avertissement_global" in data


# ============================================================
# TESTS — ÉTHIQUE ET SÉCURITÉ
# ============================================================

class TestEthiqueSecurite:

    def test_info_modele_contient_limites(self):
        """L'endpoint info-modèle doit exposer les limites"""
        response = client.get("/api/v1/predict/info-modele")
        assert response.status_code == 200
        data = response.json()
        assert "limites" in data
        assert len(data["limites"]) > 0

    def test_info_modele_contient_usage_responsable(self):
        """L'endpoint info-modèle doit exposer les règles d'usage"""
        response = client.get("/api/v1/predict/info-modele")
        assert response.status_code == 200
        data = response.json()
        assert "usage_responsable" in data
        assert len(data["usage_responsable"]) > 0

    def test_avertissement_present_dans_toutes_predictions(self):
        """L'avertissement éthique doit être présent dans chaque réponse"""
        response = client.post(
            "/api/v1/predict/individuelle",
            json=PROFIL_VALIDE
        )
        if response.status_code == 200:
            data = response.json()
            assert "avertissement_ethique" in data
            # L'avertissement ne doit pas être vide
            assert data["avertissement_ethique"] != ""


# ============================================================
# LANCEMENT DIRECT DES TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
