"""
Groundtruth — rules_loader.py

Loads and serves football rule definitions from the local
data/football_rules.json file. This is the data backend for the
`get_rule_definition` tool in the Langflow agentic pipeline.

Rules are loaded once at module import time and cached in memory —
there is no need to re-read the file on each request. Football rules
don't change during the tournament.

Usage:
    from backend.services.rules_loader import get_rule, list_rules
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution — works regardless of the working directory the server is
# started from, as long as the project root contains data/football_rules.json
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RULES_FILE = _PROJECT_ROOT / "data" / "football_rules.json"

# ---------------------------------------------------------------------------
# In-memory cache — loaded once at import time
# ---------------------------------------------------------------------------

_rules_cache: dict | None = None


def load_rules() -> dict:
    """Load the full football_rules.json into memory and return it.

    Subsequent calls return the cached dict without re-reading the file.
    If the file is missing or malformed, logs an error and returns an
    empty dict so the rest of the system degrades gracefully.

    Returns:
        dict: Full rules dictionary keyed by rule name.
    """
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    if not _RULES_FILE.exists():
        logger.error(
            "football_rules.json not found at %s. Rule lookups will return None.",
            _RULES_FILE,
        )
        _rules_cache = {}
        return _rules_cache

    try:
        with open(_RULES_FILE, "r", encoding="utf-8") as f:
            _rules_cache = json.load(f)
        logger.info(
            "Loaded %d rules from %s", len(_rules_cache), _RULES_FILE
        )
    except json.JSONDecodeError as e:
        logger.error("Failed to parse football_rules.json: %s", e)
        _rules_cache = {}

    return _rules_cache


def get_rule(keyword: str) -> str | None:
    """Return the plain_english explanation for a given rule keyword.

    Performs a case-insensitive exact-key lookup first, then falls back
    to a substring search across all rule keys. This means the agent can
    pass 'Offside' or 'offside rule' and still get a match.

    Args:
        keyword: The rule name or partial name to look up (e.g. 'offside',
                 'yellow_card', 'high press', 'VAR').

    Returns:
        str: The plain_english field for the matched rule.
        None: If no match is found.
    """
    rules = load_rules()
    if not rules:
        return None

    # 1. Exact key match (case-insensitive, normalise spaces→underscores)
    normalised = keyword.lower().strip().replace(" ", "_").replace("-", "_")
    if normalised in rules:
        return rules[normalised].get("plain_english")

    # 2. Substring match — check if keyword appears in any rule key
    for key, definition in rules.items():
        if normalised in key or keyword.lower() in key:
            logger.debug("Rule '%s' matched via substring to key '%s'", keyword, key)
            return definition.get("plain_english")

    logger.warning("No rule found for keyword: '%s'", keyword)
    return None


def get_full_rule(keyword: str) -> dict | None:
    """Return the full rule definition dict for a given keyword.

    Unlike get_rule() which returns only the plain_english field, this
    returns the entire definition including short_definition,
    when_VAR_reviews, and common_misconceptions. Useful for the agent
    when it needs richer context to craft a nuanced explanation.

    Args:
        keyword: The rule name or partial name to look up.

    Returns:
        dict: Full rule definition dict.
        None: If no match is found.
    """
    rules = load_rules()
    if not rules:
        return None

    normalised = keyword.lower().strip().replace(" ", "_").replace("-", "_")
    if normalised in rules:
        return rules[normalised]

    for key, definition in rules.items():
        if normalised in key or keyword.lower() in key:
            return definition

    return None


def list_rules() -> list[str]:
    """Return all available rule keywords from the rules file.

    Used by the agent's tool selection prompt to know what rules are
    available for lookup, and exposed via the API for debugging.

    Returns:
        list[str]: Sorted list of all rule keys.
    """
    rules = load_rules()
    return sorted(rules.keys())
