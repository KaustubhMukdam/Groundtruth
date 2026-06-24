"""
Groundtruth — granite_agent.py

The core agentic pipeline powered by IBM Granite via watsonx.ai.

Implements the 3-step pipeline described in architecture.md:
  Step 1 — Tool Selection:    Granite decides which tool to call (temp=0.1)
  Step 2 — Tool Execution:    Fetch data from the selected source
  Step 3 — Explanation:       Granite generates plain-English answer (temp=0.3)
  Step 4 — Self-Critique:     Granite checks WHY vs WHAT, revises if needed (temp=0.2)

This module runs completely independently of Langflow — it is the
Python-first implementation of the agent. The Langflow pipeline in
/langflow/Groundtruth_agent_flow.json replicates this same logic
visually for the demo video.

Usage:
    from backend.services.granite_agent import run_pipeline
    response = run_pipeline(question="Why was that offside?", match_id="537391")
"""

import logging
import os
import re

from backend.models.schemas import AgentTrace, AskResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRANITE_MODEL_ID = "ibm/granite-4-h-small"  # Confirmed via watsonx.ai API supported models list
AVAILABLE_TOOLS = ["get_match_events", "get_player_stats", "get_rule_definition"]

# Similarity threshold for detecting a meaningful revision in self-critique.
# If the revised answer differs from the draft by fewer than this many chars, 
# we treat it as unchanged (minor rewording doesn't count as "revised").
REVISION_MIN_DIFF = 30

# ---------------------------------------------------------------------------
# Prompt templates (from architecture.md + model_card.md)
# ---------------------------------------------------------------------------

TOOL_SELECTION_PROMPT = """You are a football analysis assistant helping fans understand the World Cup.

A fan asked: "{question}"

You must pick the SINGLE BEST tool to answer this question.

Available tools:
- get_match_events: For questions about specific match moments — goals, cards, VAR decisions, substitutions, what happened during a specific match, or a player's role IN a specific match
- get_player_stats: For questions about a player's overall tournament stats, their total goals/assists, or comparing a player across matches
- get_rule_definition: For questions about football rules, tactics, formations, referee decisions in general, or processes like offside, VAR, handball, high press

Respond with ONLY the tool name. One line. No explanation. Choose from:
get_match_events
get_player_stats
get_rule_definition

Tool:"""

EXPLANATION_PROMPT = """You are Groundtruth — a football expert who explains the game to fans in plain English.

A fan asked: "{question}"

Here is the relevant data:
---
{tool_output}
---

Write a clear explanation for a passionate fan who just watched this moment live.
- Focus on WHY it happened, not just WHAT happened.
- Use plain language — explain any technical term you use.
- Be warm and engaging, like you're talking to a friend after the match.
- Keep it to 3–6 sentences. Be specific, not generic.
- If the data doesn't contain enough specifics, say what you know from the rule/context.

Explanation:"""

SELF_CRITIQUE_PROMPT = """You just generated this explanation for a football fan:
---
{previous_answer}
---
Check it against this standard:
1. Does it explain WHY the decision was made? (not just WHAT happened)
2. Is it grounded in specific facts from the data? (not generic football knowledge)
3. Would a fan who didn't watch the match understand it?

If ALL THREE are YES: return the explanation EXACTLY as written above. No changes.
If ANY is NO: rewrite it to fix the specific gap. Keep the warm, fan-friendly tone.

Return ONLY the final explanation. No preamble. No "Here is the revised version:" prefix.

Final explanation:"""

# ---------------------------------------------------------------------------
# Watsonx.ai model builder
# ---------------------------------------------------------------------------


def _is_granite_configured() -> bool:
    """Check whether IBM API credentials are present in the environment."""
    return bool(os.getenv("IBM_API_KEY") and os.getenv("IBM_PROJECT_ID"))


def _build_model(temperature: float, max_new_tokens: int):
    """Create and return a watsonx.ai ModelInference instance.

    Args:
        temperature: Sampling temperature (0.1 for deterministic, 0.3 for natural language).
        max_new_tokens: Maximum tokens in the generated response.

    Returns:
        ModelInference: Ready-to-call IBM Granite model instance.

    Raises:
        RuntimeError: If IBM credentials are not set in the environment.
    """
    api_key = os.getenv("IBM_API_KEY")
    project_id = os.getenv("IBM_PROJECT_ID")
    ibm_url = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key or not project_id:
        raise RuntimeError(
            "IBM_API_KEY and IBM_PROJECT_ID must be set in .env to run the Granite agent."
        )

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    except ImportError as e:
        raise RuntimeError(
            "ibm-watsonx-ai package is not installed. Run: pip install ibm-watsonx-ai==1.0.10"
        ) from e

    model = ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials={"apikey": api_key, "url": ibm_url},
        project_id=project_id,
        params={
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.TEMPERATURE: temperature,
            GenParams.STOP_SEQUENCES: ["\n\n"],  # Stop at double newline to prevent rambling
        },
    )
    return model


# ---------------------------------------------------------------------------
# Step 1: Tool Selection
# ---------------------------------------------------------------------------


def select_tool(question: str, match_id: str | None) -> str:
    """Use Granite to determine which of the 3 tools best answers the question.

    Uses temperature=0.1 for near-deterministic tool routing.
    Falls back to a keyword-based heuristic if Granite returns an unrecognised value.

    Args:
        question: The fan's question in plain English.
        match_id: Optional match ID (if set, biases toward get_match_events for ambiguous questions).

    Returns:
        str: One of 'get_match_events', 'get_player_stats', 'get_rule_definition'.
    """
    logger.info("Step 1 — Tool selection for: '%s...'", question[:60])

    prompt = TOOL_SELECTION_PROMPT.format(question=question)

    try:
        model = _build_model(temperature=0.1, max_new_tokens=20)
        raw_output = model.generate_text(prompt=prompt).strip().lower()
        logger.info("Granite tool selection raw output: '%s'", raw_output)

        # Extract tool name from output (Granite may add extra words)
        for tool in AVAILABLE_TOOLS:
            if tool in raw_output:
                logger.info("Tool selected by Granite: %s", tool)
                return tool

    except Exception as e:
        logger.warning("Granite tool selection failed: %s — falling back to heuristic.", e)

    # Heuristic fallback: keyword matching
    return _heuristic_tool_selection(question, match_id)


def _heuristic_tool_selection(question: str, match_id: str | None) -> str:
    """Keyword-based tool selection fallback when Granite is unavailable.

    This is used when the IBM API call fails or returns an unrecognised value.
    It is NOT the primary path — Granite is always tried first.

    Args:
        question: The fan's question.
        match_id: If set, biases toward match events for ambiguous questions.

    Returns:
        str: Best-guess tool name based on question keywords.
    """
    q = question.lower()

    # Rule/tactic keywords → get_rule_definition
    rule_keywords = [
        "what is", "explain", "rule", "law", "tactic", "formation",
        "offside", "var", "handball", "yellow card", "red card", "penalty",
        "high press", "pressing", "false nine", "offside trap", "gegenpressing",
        "parking the bus", "4-3-3", "4-4-2", "3-5-2", "foul", "booking",
    ]
    if any(kw in q for kw in rule_keywords):
        return "get_rule_definition"

    # Player-specific keywords → get_player_stats
    player_keywords = [
        "how many goals", "how many assists", "player", "scored",
        "most goals", "top scorer", "hat trick",
    ]
    if any(kw in q for kw in player_keywords):
        return "get_player_stats"

    # If a match is selected, default to match events
    if match_id:
        return "get_match_events"

    return "get_rule_definition"


# ---------------------------------------------------------------------------
# Step 2: Tool Execution
# ---------------------------------------------------------------------------


def execute_tool(tool_name: str, question: str, match_id: str | None) -> str:
    """Execute the selected tool and return formatted context for Granite.

    Each tool returns a plain-text string that Granite receives as context.
    This keeps the tool outputs consistent and readable for the LLM.

    Args:
        tool_name: One of the 3 tool names.
        question: The original fan question (used for rule keyword extraction).
        match_id: football-data.org match ID (needed for match-based tools).

    Returns:
        str: Formatted context string (facts, rule text, or stats).
    """
    logger.info("Step 2 — Executing tool: %s (match_id=%s)", tool_name, match_id)

    if tool_name == "get_rule_definition":
        return _tool_get_rule_definition(question)

    if tool_name == "get_match_events":
        return _tool_get_match_events(match_id)

    if tool_name == "get_player_stats":
        # For Phase 2, player stats are derived from match events context.
        # The free tier doesn't support player search by name.
        # Phase 3 can add player search if player_id is known from the match data.
        return _tool_get_match_events(match_id)

    return "No relevant data found for this question."


def _tool_get_rule_definition(question: str) -> str:
    """Fetch rule/tactic definition from local football_rules.json."""
    from backend.services.rules_loader import get_full_rule, load_rules

    # Extract likely rule keyword from question
    rules = load_rules()
    question_lower = question.lower()

    best_match = None
    best_key = None

    # Try each rule key against the question
    for key in rules:
        normalised_key = key.replace("_", " ").replace("-", " ")
        if normalised_key in question_lower or key.lower() in question_lower:
            best_match = rules[key]
            best_key = key
            break

    # Fallback: try individual words in question
    if not best_match:
        for key in rules:
            normalised_key = key.replace("_", " ")
            for word in question_lower.split():
                if word in normalised_key or normalised_key in word:
                    best_match = rules[key]
                    best_key = key
                    break
            if best_match:
                break

    if best_match:
        logger.info("Rule found: %s", best_key)
        parts = [
            f"Rule: {best_key.replace('_', ' ').upper()}",
            f"Definition: {best_match.get('short_definition', '')}",
        ]
        if best_match.get("when_VAR_reviews") and "not applicable" not in best_match["when_VAR_reviews"].lower():
            parts.append(f"VAR involvement: {best_match['when_VAR_reviews']}")
        if best_match.get("common_misconceptions"):
            misconceptions = " | ".join(best_match["common_misconceptions"][:2])
            parts.append(f"Common misconceptions: {misconceptions}")
        parts.append(f"Fan explanation: {best_match.get('plain_english', '')}")
        return "\n".join(parts)

    logger.warning("No rule found for question: '%s'", question[:60])
    return (
        "No specific rule found for this question. "
        "Provide a general football explanation based on your knowledge."
    )


def _tool_get_match_events(match_id: str | None) -> str:
    """Fetch match events from football-data.org and format for Granite."""
    if not match_id:
        return (
            "No specific match was selected. "
            "Please answer the question using general football knowledge, "
            "and note that the fan should select a match from the dropdown for specific event details."
        )

    try:
        from backend.services.football_api import get_football_client
        client = get_football_client()
        data = client.get_match_events(match_id)
        return _format_match_data(data)
    except Exception as e:
        logger.error("Failed to fetch match events for %s: %s", match_id, e)
        return f"Could not retrieve match data (ID: {match_id}). Error: {str(e)[:100]}"


def _format_match_data(data: dict) -> str:
    """Convert raw football-data.org match JSON into readable text for Granite.

    Granite receives this as context — it needs to be clear and structured,
    not raw JSON (which wastes tokens and confuses the model).
    """
    home = (data.get("homeTeam") or {}).get("name", "Home Team")
    away = (data.get("awayTeam") or {}).get("name", "Away Team")

    score = data.get("score", {})
    full_time = score.get("fullTime", {})
    home_score = full_time.get("home", "?")
    away_score = full_time.get("away", "?")

    match_date = (data.get("utcDate") or "")[:10]
    status = data.get("status", "UNKNOWN")

    lines = [
        f"Match: {home} vs {away}",
        f"Date: {match_date}",
        f"Status: {status}",
        f"Score: {home} {home_score} – {away_score} {away}",
        "",
    ]

    # Goals
    goals = data.get("goals", [])
    if goals:
        lines.append("Goals:")
        for g in goals:
            scorer = (g.get("scorer") or {}).get("name", "Unknown")
            team = (g.get("team") or {}).get("name", "")
            minute = g.get("minute", "?")
            goal_type = g.get("type", "REGULAR")
            assist = (g.get("assist") or {}).get("name")
            line = f"  {minute}' — {scorer} ({team})"
            if goal_type != "REGULAR":
                line += f" [{goal_type}]"
            if assist:
                line += f" (assist: {assist})"
            lines.append(line)
        lines.append("")

    # Bookings / Cards
    bookings = data.get("bookings", [])
    if bookings:
        lines.append("Bookings:")
        for b in bookings:
            player = (b.get("player") or {}).get("name", "Unknown")
            team = (b.get("team") or {}).get("name", "")
            minute = b.get("minute", "?")
            card = b.get("card", "YELLOW")
            lines.append(f"  {minute}' — {player} ({team}) [{card} CARD]")
        lines.append("")

    # Substitutions
    subs = data.get("substitutions", [])
    if subs:
        lines.append("Substitutions:")
        for s in subs[:6]:  # Limit to first 6 to save tokens
            player_out = (s.get("playerOut") or {}).get("name", "?")
            player_in = (s.get("playerIn") or {}).get("name", "?")
            team = (s.get("team") or {}).get("name", "")
            minute = s.get("minute", "?")
            lines.append(f"  {minute}' — {player_out} → {player_in} ({team})")
        lines.append("")

    # Referee
    referees = data.get("referees", [])
    if referees:
        ref_name = referees[0].get("name", "Unknown")
        lines.append(f"Referee: {ref_name}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3: Explanation Generation
# ---------------------------------------------------------------------------


def generate_explanation(question: str, tool_name: str, tool_output: str) -> str:
    """Use Granite to generate a plain-English explanation grounded in tool output.

    Uses temperature=0.3 for natural, fan-friendly language while staying factual.

    Args:
        question: The original fan question.
        tool_name: Which tool was used (for context in the prompt).
        tool_output: The formatted text from tool execution.

    Returns:
        str: Draft plain-English explanation.
    """
    logger.info("Step 3 — Generating explanation...")

    prompt = EXPLANATION_PROMPT.format(
        question=question,
        tool_output=tool_output,
    )

    try:
        model = _build_model(temperature=0.3, max_new_tokens=400)
        explanation = model.generate_text(prompt=prompt).strip()
        logger.info("Explanation generated (%d chars).", len(explanation))
        return explanation
    except Exception as e:
        logger.error("Granite explanation generation failed: %s", e)
        # Return the tool output directly as a fallback — not ideal but better than empty
        return (
            f"Here's what I found: {tool_output[:500]}\n\n"
            f"(Note: The AI explanation step encountered an error: {str(e)[:100]})"
        )


# ---------------------------------------------------------------------------
# Step 4: Self-Critique
# ---------------------------------------------------------------------------


def self_critique(draft_explanation: str) -> tuple[str, bool, str | None]:
    """Use Granite to review and potentially revise the draft explanation.

    This is the key differentiating feature of the Groundtruth agent.
    Granite checks its own output against the WHY/grounded/fan-understandable
    checklist defined in architecture.md.

    Uses temperature=0.2 for consistent, structured output.

    Args:
        draft_explanation: The explanation from Step 3.

    Returns:
        tuple[str, bool, str | None]:
            - final_answer: The (possibly revised) explanation
            - was_revised: True if the answer was meaningfully changed
            - revision_reason: Brief note on what was improved (if revised)
    """
    logger.info("Step 4 — Self-critique pass...")

    prompt = SELF_CRITIQUE_PROMPT.format(previous_answer=draft_explanation)

    try:
        model = _build_model(temperature=0.2, max_new_tokens=400)
        revised = model.generate_text(prompt=prompt).strip()
        logger.info("Self-critique complete. Revised length: %d chars.", len(revised))

        # Handle suspicious / empty responses immediately
        if len(revised) < 30:
            logger.warning("Self-critique returned suspiciously short or empty string. Rejecting revision.")
            return draft_explanation, False, None

        # Detect meaningful revision: significant character difference
        was_revised = abs(len(revised) - len(draft_explanation)) > REVISION_MIN_DIFF or (
            revised.lower() != draft_explanation.lower()
            and _text_similarity_low(revised, draft_explanation)
        )

        revision_reason = None
        if was_revised:
            revision_reason = "Added more specific WHY explanation and grounded it in match context."
            logger.info("Self-critique produced a meaningful revision.")
        else:
            logger.info("Self-critique: original answer passed the WHY/grounded/clarity checks.")
            revised = draft_explanation

        return revised, was_revised, revision_reason

    except Exception as e:
        logger.error("Granite self-critique failed: %s — returning original.", e)
        return draft_explanation, False, None


def _text_similarity_low(text_a: str, text_b: str) -> bool:
    """Rough check: returns True if the two texts share fewer than 70% of words.

    Used to confirm a self-critique revision is substantive, not just minor
    rephrasing that happens to be a different length.
    """
    words_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    words_b = set(re.findall(r"\b\w+\b", text_b.lower()))
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap < 0.70  # Meaningfully different if < 70% word overlap


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(question: str, match_id: str | None = None) -> AskResponse:
    """Run the full Groundtruth agentic pipeline for a fan's question.

    This is the single entry point called by langflow_client.run_agent() when
    Langflow is not configured AND IBM credentials are available.

    The 4-step pipeline:
    1. Granite selects which tool to call
    2. Tool fetches relevant data (match events / player stats / rule definition)
    3. Granite generates a plain-English explanation grounded in the data
    4. Granite critiques its own answer; revises if WHY explanation is missing

    Args:
        question: The fan's football question in plain English.
        match_id: Optional football-data.org match ID for match-specific questions.

    Returns:
        AskResponse: Final answer + AgentTrace (tool_used, was_revised, revision_reason).
    """
    logger.info(
        "=== Granite pipeline START | question='%s...' | match_id=%s ===",
        question[:50],
        match_id,
    )

    # ── Step 1: Tool Selection ─────────────────────────────────────────────
    tool_name = select_tool(question=question, match_id=match_id)

    # ── Step 2: Tool Execution ─────────────────────────────────────────────
    tool_output = execute_tool(tool_name=tool_name, question=question, match_id=match_id)

    # ── Step 3: Explanation Generation ─────────────────────────────────────
    draft_explanation = generate_explanation(
        question=question,
        tool_name=tool_name,
        tool_output=tool_output,
    )

    # ── Step 4: Self-Critique ──────────────────────────────────────────────
    final_answer, was_revised, revision_reason = self_critique(
        draft_explanation=draft_explanation
    )

    logger.info(
        "=== Granite pipeline END | tool=%s | was_revised=%s ===",
        tool_name,
        was_revised,
    )

    return AskResponse(
        answer=final_answer,
        agent_trace=AgentTrace(
            tool_used=tool_name,
            was_revised=was_revised,
            revision_reason=revision_reason,
        ),
    )
