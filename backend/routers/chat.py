"""
Groundtruth — chat.py router

Exposes POST /ask — the main agentic endpoint.

Request: { question: str, match_id: str | None }
Response: { answer: str, agent_trace: { tool_used, was_revised, revision_reason } }

In Phase 1 (current), this calls langflow_client.run_agent() which returns
a mock response if Langflow is not configured. The endpoint never returns a
500 — if Langflow is unreachable, a mock AskResponse is returned so the
Streamlit UI can always render something.

In Phase 2, the same endpoint will return real IBM Granite responses from the
Langflow pipeline — no changes needed to this router.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import AskRequest, AskResponse
from backend.services import langflow_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["chat"])


@router.post("", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """Send a football question to the Langflow agentic pipeline.

    The agent autonomously:
    1. Selects a tool (get_match_events / get_player_stats / get_rule_definition)
    2. Fetches grounded data from the selected source
    3. Generates a plain-English explanation using IBM Granite
    4. Runs a self-critique pass to ensure the answer explains WHY, not just WHAT
    5. Returns the final answer with AgentTrace metadata

    In dev mode (Langflow not configured), returns a realistic mock response
    with tool_used="mock" so the Streamlit UI can always render the AgentTrace.

    Args:
        request: AskRequest with question (required) and match_id (optional).

    Returns:
        AskResponse: Answer + AgentTrace (tool_used, was_revised, revision_reason).

    Raises:
        HTTPException(422): If the request body fails Pydantic validation
                            (handled automatically by FastAPI).
    """
    logger.info(
        "POST /ask — question: '%s...', match_id: %s",
        request.question[:50],
        request.match_id,
    )

    try:
        response = langflow_client.run_agent(
            question=request.question,
            match_id=request.match_id,
        )
        logger.info(
            "POST /ask → tool_used=%s, was_revised=%s",
            response.agent_trace.tool_used,
            response.agent_trace.was_revised,
        )
        return response

    except HTTPException:
        raise

    except Exception as e:
        # Unexpected errors should never reach the user as a 500 —
        # return a graceful error message instead.
        logger.error("Unexpected error in POST /ask: %s", e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"The agent encountered an unexpected error: {str(e)[:200]}",
        ) from e
