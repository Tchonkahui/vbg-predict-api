# ============================================================
# MIDDLEWARE — Logging & Monitoring des Requêtes
# ============================================================

import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de logging automatique de toutes les requêtes.
    Enregistre : méthode, path, durée, status code, IP.
    Les données personnelles ne sont jamais loggées.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        # Log de la requête entrante
        logger.info(
            f"→ {request.method} {request.url.path} | "
            f"IP={request.client.host}"
        )

        # Traitement de la requête
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Erreur non gérée : {e}")
            raise

        # Durée de traitement
        duree = round((time.time() - start_time) * 1000, 2)

        # Log de la réponse
        log_fn = logger.success if response.status_code < 400 else logger.warning
        log_fn(
            f"← {request.method} {request.url.path} | "
            f"status={response.status_code} | "
            f"durée={duree}ms"
        )

        # Ajout des headers de réponse
        response.headers["X-Process-Time"] = f"{duree}ms"
        response.headers["X-API-Version"]  = "1.0.0"

        return response
