#!/usr/bin/env python3
# ============================================================
# POINT D'ENTRÉE — Lancement de l'API VBG
# Usage : python run.py
# ============================================================

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host    = settings.HOST,
        port    = settings.PORT,
        reload  = settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
