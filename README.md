# 🔍 API Prédiction — Violences Basées sur le Genre (VBG)

> **Projet académique** — École Nationale Supérieure Polytechnique de Yaoundé (ENSPY)  
> Data Science 3ème année | Analyse prédictive des facteurs socio-démographiques associés aux VBG

---

## ⚠️ Avertissement Éthique Important

> Ce projet est développé à des fins **académiques et de recherche uniquement**.  
> Les prédictions générées sont des **indicateurs statistiques**, pas des diagnostics.  
> Tout usage discriminatoire, stigmatisant ou de ciblage individuel automatique est **strictement interdit**.  
> Toute prédiction doit être **validée par un professionnel qualifié**.

---

## 📋 Description

API REST développée avec **FastAPI** permettant de prédire la probabilité de risque de violence domestique à partir de facteurs socio-démographiques.

**Objectif** : Orienter les politiques publiques de prévention vers les populations les plus vulnérables.

---

## 🚀 Installation et Lancement

### Prérequis
- Python 3.11+
- pip

### Installation locale

```bash
# Cloner le projet
git clone https://github.com/votre-username/vbg-prediction-api.git
cd vbg-prediction-api

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt

# Placer le modèle dans le répertoire racine
# (télécharger model_vbg.pkl depuis Colab)
cp /chemin/vers/model_vbg.pkl .
cp /chemin/vers/model_metadata.json .

# Lancer l'API
python run.py
```

### Avec Docker

```bash
# Build de l'image
docker build -t vbg-api:1.0.0 .

# Lancement du conteneur
docker run -d \
  --name vbg-api \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  vbg-api:1.0.0

# Vérification
curl http://localhost:8000/health
```

---

## 📡 Endpoints Disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET`   | `/`      | Accueil et informations |
| `GET`   | `/health` | État de l'API et du modèle |
| `GET`   | `/docs`  | Documentation interactive (Swagger) |
| `GET`   | `/redoc` | Documentation (ReDoc) |
| `POST`  | `/api/v1/predict/individuelle` | Prédiction individuelle |
| `POST`  | `/api/v1/predict/batch` | Prédictions par lot |
| `GET`   | `/api/v1/predict/info-modele` | Métadonnées et limites |

---

## 💡 Exemples d'Utilisation

### Prédiction individuelle

```bash
curl -X POST "http://localhost:8000/api/v1/predict/individuelle" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 27,
    "education": "primary",
    "employment": "unemployed",
    "income": 0.0,
    "marital_status": "married",
    "identifiant": "CASE_001"
  }'
```

**Réponse :**
```json
{
  "identifiant": "CASE_001",
  "prediction": 1,
  "probabilite_risque": 0.6842,
  "niveau_risque": "Élevé",
  "pourcentage_risque": "68.4%",
  "recommandation": "Intervention prioritaire — évaluation par travailleur social requise",
  "facteurs_risque": [
    "Sans emploi — dépendance économique (facteur principal)",
    "Absence de revenu propre",
    "Statut marital : mariée",
    "Tranche d'âge à risque : 27 ans (25-35 ans)"
  ],
  "avertissement_ethique": "⚠️ Cette prédiction est un indicateur statistique, pas un diagnostic...",
  "timestamp": "2024-11-15T14:32:01",
  "version_modele": "1.0.0"
}
```

### Prédiction batch

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [
      {"age": 27, "education": "primary", "employment": "unemployed",
       "income": 0.0, "marital_status": "married"},
      {"age": 45, "education": "tertiary", "employment": "employed",
       "income": 5000.0, "marital_status": "unmarried"}
    ]
  }'
```

---

## 🏗️ Structure du Projet

```
vbg_api/
├── app/
│   ├── __init__.py
│   ├── main.py               # Application FastAPI principale
│   ├── config.py             # Configuration centralisée
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Schémas Pydantic (validation I/O)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── prediction.py     # Endpoints de prédiction
│   │   └── health.py         # Health check
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py  # Logique métier + ML
│   └── middleware/
│       ├── __init__.py
│       └── logging_middleware.py  # Logging des requêtes
├── tests/
│   └── test_api.py           # Tests pytest complets
├── logs/                     # Logs générés automatiquement
├── docs/                     # Documentation additionnelle
├── model_vbg.pkl             # Modèle ML (non versionné)
├── model_metadata.json       # Métadonnées du modèle
├── requirements.txt          # Dépendances Python
├── Dockerfile                # Conteneurisation
├── .env.example              # Variables d'environnement exemple
└── README.md                 # Ce fichier
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec couverture de code
pytest tests/ -v --cov=app --cov-report=html

# Tests spécifiques
pytest tests/test_api.py::TestValidationEntrees -v
pytest tests/test_api.py::TestEthiqueSecurite -v
```

---

## 📊 Modèle ML

| Paramètre | Valeur |
|-----------|--------|
| Dataset | Domestic Violence — Kaggle (347 obs.) |
| Features | Age, Éducation, Emploi, Revenu, Statut marital |
| Variable cible | Violence (Oui/Non) |
| Validation | StratifiedKFold (k=5) |
| Métrique prioritaire | Recall (minimiser faux négatifs) |
| Déséquilibre | Traité par SMOTE (ratio 0.6) |

---

## ⚖️ Considérations Éthiques

1. **Corrélation ≠ Causalité** — Les facteurs identifiés sont associés statistiquement, non causalement
2. **Biais de déclaration** — Les VBG sont sous-déclarées ; les résultats sous-estiment probablement la prévalence
3. **Usage collectif uniquement** — Jamais de ciblage individuel automatique
4. **Validation humaine** — Toute prédiction doit être validée par un professionnel
5. **Audit régulier** — Révision des biais recommandée tous les 6 mois

---

## 📚 Références

- OMS (2021). *Violence against women prevalence estimates*
- Lundberg & Lee (2017). *A unified approach to interpreting model predictions*. NeurIPS
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR

---

## 👨‍💻 Auteur

**Landry** — Étudiant 3ème année Data Science  
École Nationale Supérieure Polytechnique de Yaoundé (ENSPY)  
Cameroun — 2024

---

## 📄 Licence

Usage académique uniquement — ENSPY 2024  
Toute utilisation commerciale ou en production requiert une validation éthique institutionnelle.
