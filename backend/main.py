"""
Groundtruth — main.py

FastAPI application entry point for the Groundtruth backend.

Sets up:
  - CORS middleware (permissive for hackathon — all origins allowed)
  - Application lifespan: startup check for football API availability
  - Health check endpoint: GET /health
  - Router registration: /matches and /ask

Run with:
    uvicorn backend.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env before importing services (services read env vars at import time)
load_dotenv()

from backend.routers import chat, matches  # noqa: E402 — must be after load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Application lifespan — runs startup/shutdown logic
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks on application start.

    Currently:
    - Checks that football-data.org World Cup competition data is accessible
      on the free tier. Logs a clear warning if not.
    - Logs the rules file availability (rules_loader loads on first call).

    This is a non-blocking check — the app starts even if the API is
    unreachable, so developers can work with mock data.
    """
    logger.info("Groundtruth backend starting...")

    # Startup: check football API
    football_api_key = os.getenv("FOOTBALL_API_KEY")
    if football_api_key:
        try:
            from backend.services.football_api import FootballAPIClient
            client = FootballAPIClient()
            available = client.check_competition_available()
            if not available:
                logger.warning(
                    "⚠️  World Cup competition data is NOT available on the free tier. "
                    "Check your football-data.org subscription. "
                    "The /matches endpoint will return a 503 until this is resolved."
                )
            else:
                logger.info("✅ Football API: World Cup data is accessible.")
        except Exception as e:
            logger.warning("Football API startup check failed: %s", e)
    else:
        logger.warning(
            "FOOTBALL_API_KEY not set — football API calls will fail. "
            "Set it in .env to enable real match data."
        )

    # Startup: pre-load rules cache
    try:
        from backend.services.rules_loader import load_rules, list_rules
        load_rules()
        rule_names = list_rules()
        logger.info(
            "✅ Rules loaded: %d rules available (%s ... )",
            len(rule_names),
            ", ".join(rule_names[:3]),
        )
    except Exception as e:
        logger.warning("Rules loader startup check failed: %s", e)

    logger.info("Groundtruth backend ready.")
    yield
    logger.info("Groundtruth backend shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Groundtruth API",
    description=(
        "Agentic AI backend for Groundtruth — explains World Cup football moments "
        "to fans using IBM Granite + Langflow. "
        "Built for the IBM SkillsBuild AI Builders Challenge 2026."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for hackathon (Streamlit runs on a different port)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(matches.router)
app.include_router(chat.router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Simple liveness probe. Returns 200 when the server is running.

    Used by Render's free tier health checks and by the Streamlit
    frontend to confirm the backend is reachable before making API calls.
    """
    return {"status": "ok", "version": "0.1.0"}
