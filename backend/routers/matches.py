"""
Groundtruth — matches.py router

Exposes GET /matches, which returns a list of World Cup matches from
the football-data.org API for the Streamlit match selector dropdown.

The response is a list of MatchInfo objects (id, home_team, away_team,
date, status). The Streamlit UI renders these as dropdown options so the
user can select the match they're watching before asking a question.

If the football API is unavailable (no key set, rate limited, or the
free tier doesn't cover World Cup 2026), a 503 error is returned with
a descriptive message — never a silent 500.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import MatchInfo
from backend.services.football_api import get_football_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("", response_model=list[MatchInfo])
async def list_matches() -> list[MatchInfo]:
    """Return all World Cup 2026 matches for the UI dropdown.

    Fetches from football-data.org with a 5-minute in-memory cache.
    Includes all matches regardless of status (scheduled, live, finished).

    Returns:
        list[MatchInfo]: Sorted list of match summaries.

    Raises:
        HTTPException(503): If football-data.org is unreachable or returns an error.
        HTTPException(503): If FOOTBALL_API_KEY is not configured.
    """
    try:
        client = get_football_client()
        matches = client.get_matches()
        logger.info("GET /matches → returning %d matches", len(matches))
        return matches
    except HTTPException:
        # Re-raise HTTPExceptions from the football API client unchanged
        raise
    except Exception as e:
        logger.error("Unexpected error in GET /matches: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch match list: {str(e)[:200]}",
        ) from e
