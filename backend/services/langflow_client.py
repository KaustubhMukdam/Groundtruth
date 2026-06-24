"""
Groundtruth — langflow_client.py

Thin wrapper that calls the Langflow agentic pipeline via its REST endpoint.
Langflow exposes every pipeline as a REST API once running — this module
sends the user question and match_id to that endpoint and parses the
structured response back into an AskResponse.

Phase 1 (current): Returns a realistic mock response when Langflow is not
running (LANGFLOW_URL not set or connection refused). This keeps the
backend fully testable without Langflow installed.

Phase 2: Replace the mock branch with real Langflow API calls once the
pipeline is built and the flow ID is known.
"""

import logging
import os

import httpx

from backend.models.schemas import AgentTrace, AskResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGFLOW_DEFAULT_TIMEOUT = 30.0  # seconds — 3 Granite calls per request ~5–7s


# ---------------------------------------------------------------------------
# Mock response (dev mode / Phase 1)
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = AskResponse(
    answer=(
        "This is a placeholder answer from Groundtruth in development mode. "
        "Langflow is not connected yet — this mock confirms the full request/response "
        "pipeline is working. In Phase 2, IBM Granite will autonomously select a tool, "
        "fetch real match data or rule definitions, generate a grounded explanation, "
        "and run a self-critique pass before returning the final answer here."
    ),
    agent_trace=AgentTrace(
        tool_used="mock",
        was_revised=False,
        revision_reason=None,
    ),
)


# ---------------------------------------------------------------------------
# Langflow client
# ---------------------------------------------------------------------------


def _get_langflow_config() -> tuple[str | None, str | None]:
    """Read Langflow connection config from environment variables.

    Returns:
        tuple[str | None, str | None]: (base_url, flow_id) — either or
        both may be None if not configured.
    """
    return os.getenv("LANGFLOW_URL"), os.getenv("LANGFLOW_FLOW_ID")


def _parse_langflow_response(data: dict) -> AskResponse:
    """Parse the Langflow REST response into an AskResponse.

    Langflow returns a nested JSON structure that changes slightly between
    versions. This function extracts the relevant fields defensively.

    Expected Langflow v1 response structure (approximate):
    {
        "outputs": [
            {
                "outputs": [
                    {
                        "results": {
                            "message": { "text": "..." }
                        }
                    }
                ]
            }
        ],
        "session_id": "..."
    }

    TODO (Phase 2): Update this parser once the actual Langflow pipeline
    response shape is confirmed. The agent pipeline will need to return
    tool_used and was_revised in a structured way — either as part of the
    message text (JSON-encoded) or as separate flow variables.

    Args:
        data: Raw Langflow API response as a dict.

    Returns:
        AskResponse: Parsed response with AgentTrace metadata.
    """
    try:
        # Attempt to extract answer from standard Langflow v1 output structure
        outputs = data.get("outputs", [])
        if outputs:
            inner_outputs = outputs[0].get("outputs", [])
            if inner_outputs:
                results = inner_outputs[0].get("results", {})
                message = results.get("message", {})
                answer_text = message.get("text", "")

                if answer_text:
                    # TODO (Phase 2): Parse tool_used and was_revised from
                    # Langflow output variables once the pipeline is wired up.
                    # For now, we return sensible defaults.
                    return AskResponse(
                        answer=answer_text,
                        agent_trace=AgentTrace(
                            tool_used="langflow",  # will be specific in Phase 2
                            was_revised=False,       # will come from pipeline in Phase 2
                            revision_reason=None,
                        ),
                    )
    except (KeyError, IndexError, TypeError) as e:
        logger.warning("Unexpected Langflow response structure: %s — data: %s", e, str(data)[:200])

    # Fallback if response structure is unexpected
    logger.warning("Could not parse Langflow response cleanly — returning raw output.")
    return AskResponse(
        answer=str(data),
        agent_trace=AgentTrace(
            tool_used="langflow_unparsed",
            was_revised=False,
            revision_reason=None,
        ),
    )


def run_agent(question: str, match_id: str | None = None) -> AskResponse:
    """Send a question to the Langflow agentic pipeline and return the response.

    This is the single entry point called by the /ask router. It handles:
    - Dev mode (Langflow not configured): returns _MOCK_RESPONSE immediately
    - Connected mode: POSTs to the Langflow flow endpoint and parses the result
    - Connection failure: logs the error, falls back to mock response

    Args:
        question: The football question from the fan.
        match_id: Optional football-data.org match ID for match-specific questions.

    Returns:
        AskResponse: Full agent response including answer and AgentTrace.
    """
    langflow_url, flow_id = _get_langflow_config()

    # ── Priority 1: Langflow connected mode ────────────────────────────────
    # If Langflow URL and Flow ID are configured, use the visual pipeline.
    if langflow_url and flow_id and "your_langflow_flow_id_here" not in flow_id:
        endpoint = f"{langflow_url.rstrip('/')}/api/v1/run/{flow_id}"
        payload = {
            "input_value": question,
            "output_type": "chat",
            "input_type": "chat",
            "tweaks": {"match_id": match_id or ""},
        }
        try:
            logger.info("Calling Langflow: POST %s (match_id=%s)", endpoint, match_id)
            with httpx.Client(timeout=LANGFLOW_DEFAULT_TIMEOUT) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                logger.info("Langflow responded successfully.")
                return _parse_langflow_response(data)
        except httpx.ConnectError:
            logger.warning(
                "Langflow is not reachable at %s. Falling back to Python agent.", endpoint
            )
        except httpx.HTTPStatusError as e:
            logger.error("Langflow HTTP %d. Falling back to Python agent.", e.response.status_code)
        except httpx.RequestError as e:
            logger.error("Langflow request error: %s. Falling back to Python agent.", e)

    # ── Priority 2: Direct Python Granite agent ────────────────────────────
    # When Langflow is not running (or Phase 2 building), use the Python
    # implementation of the same pipeline. Requires IBM_API_KEY + IBM_PROJECT_ID.
    from backend.services import granite_agent

    if granite_agent._is_granite_configured():
        logger.info(
            "Langflow not configured or unreachable. "
            "Running Granite pipeline directly via Python agent."
        )
        try:
            return granite_agent.run_pipeline(question=question, match_id=match_id)
        except Exception as e:
            logger.error(
                "Python Granite agent failed: %s — falling back to mock.", e, exc_info=True
            )

    # ── Priority 3: Mock mode ──────────────────────────────────────────────
    # Only reached when NEITHER Langflow NOR IBM credentials are configured.
    # Useful for frontend/backend integration testing without any AI keys.
    logger.warning(
        "IBM_API_KEY not set and Langflow not configured. "
        "Returning mock response. Set credentials in .env to get real answers."
    )
    return _MOCK_RESPONSE
