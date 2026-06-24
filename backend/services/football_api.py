"""
Groundtruth — football_api.py

Client for the football-data.org REST API (v4), with in-memory caching
to respect the free tier's 10 calls/minute rate limit.

Endpoints used:
  - GET /v4/competitions/WC/matches  → list of World Cup matches
  - GET /v4/matches/{match_id}       → full match detail (events, goals, cards)
  - GET /v4/persons/{player_id}/matches → player match history + stats

Cache strategy:
  - Key: "{endpoint_name}:{identifier}"
  - TTL: 300 seconds (5 minutes)
  - Storage: in-process dict (_cache) — no Redis, no external dependencies
  - Cache is shared across requests within the same process

All public methods raise HTTPException(503) on API failure rather than
letting the error propagate silently into the agent pipeline.
"""

import logging
import os
import time
from typing import Any

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from backend.models.schemas import MatchInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.football-data.org/v4"
CACHE_TTL = 300  # seconds
WORLD_CUP_CODE = "WC"

# ---------------------------------------------------------------------------
# In-memory cache (module-level singleton, shared across requests)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[Any, float]] = {}


def _get_cached(key: str) -> Any | None:
    """Return cached data for key if it exists and hasn't expired."""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL:
            logger.debug("Cache HIT: %s", key)
            return data
        logger.debug("Cache EXPIRED: %s", key)
    return None


def _set_cached(key: str, data: Any) -> None:
    """Store data in the cache with the current timestamp."""
    _cache[key] = (data, time.time())
    logger.debug("Cache SET: %s", key)


def _get_api_key() -> str:
    """Retrieve the football-data.org API key from the environment.

    Raises:
        HTTPException(503): If the key is not configured.
    """
    key = os.getenv("FOOTBALL_API_KEY")
    if not key:
        logger.error(
            "FOOTBALL_API_KEY environment variable is not set. "
            "Football API calls will fail."
        )
        raise HTTPException(
            status_code=503,
            detail="Football API is not configured. Set FOOTBALL_API_KEY in .env.",
        )
    return key


# ---------------------------------------------------------------------------
# FootballAPIClient
# ---------------------------------------------------------------------------


class FootballAPIClient:
    """Thin async client for the football-data.org v4 API.

    All methods are synchronous (using httpx.Client) so they can be called
    directly from FastAPI path operations without needing async wrappers.
    For Phase 2, these can be converted to async (httpx.AsyncClient) to
    improve throughput.
    """

    def __init__(self) -> None:
        self._headers = {"X-Auth-Token": _get_api_key()}

    def _get(self, endpoint: str, cache_key: str) -> dict:
        """Make a GET request to the football-data.org API.

        Checks the in-memory cache before making a network request.
        On any HTTP or network error, raises HTTPException(503) so the
        API always returns a structured error to the Streamlit frontend.

        Args:
            endpoint: URL path relative to BASE_URL (e.g. "/competitions/WC/matches").
            cache_key: Unique key for this request in the cache.

        Returns:
            dict: Parsed JSON response body.

        Raises:
            HTTPException(503): On API failure or network error.
        """
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"{BASE_URL}{endpoint}"
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self._headers)
                response.raise_for_status()
                data = response.json()
                _set_cached(cache_key, data)
                return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "Football API returned %d for %s: %s",
                e.response.status_code,
                url,
                e.response.text[:200],
            )
            raise HTTPException(
                status_code=503,
                detail=f"Football API error {e.response.status_code}: {e.response.text[:100]}",
            ) from e

        except httpx.RequestError as e:
            logger.error("Football API network error for %s: %s", url, e)
            raise HTTPException(
                status_code=503,
                detail=f"Could not reach the football data API: {str(e)[:100]}",
            ) from e

    def check_competition_available(self) -> bool:
        """Startup check — confirm the World Cup competition is accessible.

        Called during application startup (lifespan event). Logs a clear
        warning if the free tier does not cover the World Cup 2026 data.

        Returns:
            bool: True if the competition endpoint responds with valid data.
        """
        try:
            data = self._get(
                f"/competitions/{WORLD_CUP_CODE}",
                cache_key=f"competition:{WORLD_CUP_CODE}",
            )
            competition_name = data.get("name", "unknown")
            logger.info(
                "Football API: competition '%s' is accessible (code=%s).",
                competition_name,
                WORLD_CUP_CODE,
            )
            return True
        except HTTPException:
            logger.warning(
                "Football API: competition '%s' is NOT accessible on the free tier. "
                "Match data will be unavailable. Check your subscription or API key.",
                WORLD_CUP_CODE,
            )
            return False

    def get_matches(self) -> list[MatchInfo]:
        """Fetch all World Cup matches for the UI dropdown.

        Returns:
            list[MatchInfo]: Lightweight match summaries with id, teams, date, status.

        Raises:
            HTTPException(503): If the API is unavailable.
        """
        data = self._get(
            f"/competitions/{WORLD_CUP_CODE}/matches",
            cache_key=f"matches:{WORLD_CUP_CODE}",
        )

        matches = []
        for raw in data.get("matches", []):
            try:
                # Knockout stage matches have null team names until opponents are decided.
                # Use "TBD" so Pydantic doesn't reject the record — these still show
                # in the dropdown so fans can select the match slot.
                home_name = (raw.get("homeTeam") or {}).get("name") or "TBD"
                away_name = (raw.get("awayTeam") or {}).get("name") or "TBD"

                matches.append(
                    MatchInfo(
                        id=str(raw["id"]),
                        home_team=home_name,
                        away_team=away_name,
                        date=raw["utcDate"][:10],  # ISO date only
                        status=raw["status"],
                    )
                )
            except (KeyError, TypeError, ValidationError) as e:
                logger.warning("Skipping malformed match record: %s — %s", raw.get("id"), e)

        logger.info("Fetched %d matches from football-data.org.", len(matches))
        return matches

    def get_match_events(self, match_id: str) -> dict:
        """Fetch full match detail for a specific match.

        This is the data source for the `get_match_events` agent tool.
        The returned dict contains goals, bookings, substitutions, lineups,
        and referee information exactly as returned by the API.

        Args:
            match_id: football-data.org numeric match ID (as string).

        Returns:
            dict: Full match detail JSON.

        Raises:
            HTTPException(503): If the API is unavailable.
        """
        return self._get(
            f"/matches/{match_id}",
            cache_key=f"match_events:{match_id}",
        )

    def get_player_stats(self, player_id: str) -> dict:
        """Fetch match history and stats for a specific player.

        This is the data source for the `get_player_stats` agent tool.

        Args:
            player_id: football-data.org numeric person/player ID (as string).

        Returns:
            dict: Player match history JSON with goals, assists, cards per match.

        Raises:
            HTTPException(503): If the API is unavailable.
        """
        return self._get(
            f"/persons/{player_id}/matches",
            cache_key=f"player_stats:{player_id}",
        )


# ---------------------------------------------------------------------------
# Module-level client instance (created lazily to allow env var loading)
# ---------------------------------------------------------------------------

_client_instance: FootballAPIClient | None = None


def get_football_client() -> FootballAPIClient:
    """Return the shared FootballAPIClient instance.

    Raises HTTPException(503) if FOOTBALL_API_KEY is not set.
    This is intentional — every API call fails fast rather than silently.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = FootballAPIClient()
    return _client_instance
