"""
Groundtruth — frontend/app.py

Streamlit chat UI for the Groundtruth football explainability agent.

Features:
  - Match selector dropdown (fetches from GET /matches on the FastAPI backend)
  - Chat input + scrollable message history (persisted in st.session_state)
  - For each AI response, shows agent trace metadata:
      🔧 Tool used: {tool_used}
      ✏️  Answer revised: Yes / No
  - Graceful handling when the backend is unreachable (friendly error, no crash)

Run with:
    streamlit run frontend/app.py

The backend URL is read from BACKEND_URL in the environment (defaults to
http://localhost:8000 for local development).
"""

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PAGE_TITLE = "Groundtruth ⚽"
PAGE_ICON = "⚽"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": str, "content": str, "trace": dict|None}

if "selected_match" not in st.session_state:
    st.session_state.selected_match = None  # MatchInfo dict or None

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def fetch_matches() -> list[dict]:
    """Fetch the list of World Cup matches from the FastAPI backend.

    Returns:
        list[dict]: Match dicts with id, home_team, away_team, date, status.
        Empty list on error (error is surfaced via st.warning).
    """
    try:
        response = requests.get(f"{BACKEND_URL}/matches", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        st.warning(
            f"⚠️ Cannot reach the backend at `{BACKEND_URL}`. "
            "Is the FastAPI server running? (`uvicorn backend.main:app --reload`)"
        )
        return []
    except requests.HTTPError as e:
        st.warning(f"⚠️ Backend returned an error: {e.response.status_code} — {e.response.text[:200]}")
        return []
    except Exception as e:
        st.warning(f"⚠️ Unexpected error fetching matches: {e}")
        return []


def ask_agent(question: str, match_id: str | None) -> dict | None:
    """Send a question to the FastAPI /ask endpoint.

    Args:
        question: The fan's football question.
        match_id: Optional match ID from the dropdown selection.

    Returns:
        dict: AskResponse JSON with 'answer' and 'agent_trace' keys.
        None: On error (error is surfaced via st.error).
    """
    payload = {"question": question, "match_id": match_id}
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json=payload,
            timeout=120,  # 3 Granite calls = up to ~7s; 30s leaves plenty of margin
        )
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        st.error(
            f"❌ Cannot reach the backend at `{BACKEND_URL}`. "
            "Please start the FastAPI server and refresh."
        )
        return None
    except requests.HTTPError as e:
        st.error(
            f"❌ The agent returned an error ({e.response.status_code}): "
            f"{e.response.text[:300]}"
        )
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def format_match_label(match: dict) -> str:
    """Format a match dict into a human-readable dropdown label."""
    return f"{match['home_team']} vs {match['away_team']} ({match['date']})"


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Groundtruth ⚽")
st.caption(
    "Ask anything about a World Cup match — rules, tactics, referee decisions. "
    "Powered by IBM Granite + Langflow."
)

st.divider()

# ---------------------------------------------------------------------------
# Match selector
# ---------------------------------------------------------------------------

st.subheader("Select a match (optional)")
st.caption("Selecting a match helps the agent answer questions about specific moments.")

matches = fetch_matches()

match_options = ["No specific match — general question"] + [
    format_match_label(m) for m in matches
]
selected_label = st.selectbox("Match", match_options, key="match_selector")

# Resolve the selected match_id
if selected_label == match_options[0]:
    selected_match_id = None
else:
    # Find the match dict that corresponds to the selected label
    selected_idx = match_options.index(selected_label) - 1  # offset for the "no match" option
    selected_match_id = matches[selected_idx]["id"] if matches else None

st.divider()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

st.subheader("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            trace = msg["trace"]
            tool = trace.get("tool_used", "unknown")
            revised = trace.get("was_revised", False)
            revision_reason = trace.get("revision_reason")

            st.caption(
                f"🔧 **Tool used:** `{tool}`   "
                f"✏️ **Answer revised:** {'Yes' if revised else 'No'}"
            )
            if revised and revision_reason:
                with st.expander("Why was the answer revised?"):
                    st.write(revision_reason)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

question = st.chat_input(
    "Ask a football question... (e.g. 'Why was that offside?', 'What is a high press?')"
)

if question:
    # Append user message to history
    st.session_state.messages.append(
        {"role": "user", "content": question, "trace": None}
    )

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Call the agent
    with st.chat_message("assistant"):
        with st.spinner("Groundtruth is thinking... ⚽"):
            result = ask_agent(question=question, match_id=selected_match_id)

        if result:
            answer = result.get("answer", "Sorry, I didn't get a response.")
            trace = result.get("agent_trace", {})

            st.markdown(answer)

            tool = trace.get("tool_used", "unknown")
            revised = trace.get("was_revised", False)
            revision_reason = trace.get("revision_reason")

            st.caption(
                f"🔧 **Tool used:** `{tool}`   "
                f"✏️ **Answer revised:** {'Yes' if revised else 'No'}"
            )
            if revised and revision_reason:
                with st.expander("Why was the answer revised?"):
                    st.write(revision_reason)

            # Save to session history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )
        else:
            # ask_agent already surfaced an st.error — don't add to history
            pass

# ---------------------------------------------------------------------------
# Sidebar: about panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("About Groundtruth")
    st.markdown(
        """
        **Groundtruth** is an agentic AI system that explains World Cup 
        football moments to fans in plain English.

        **How it works:**
        1. You ask a question
        2. IBM Granite picks the right data tool
        3. Real match data is fetched
        4. Granite explains the WHY
        5. A self-critique pass checks the answer
        6. You get a grounded explanation

        **Built for:** IBM SkillsBuild AI Builders Challenge 2026

        **IBM tools:** Granite `granite-4-h-small` · Langflow
        """
    )
    st.divider()
    st.caption(f"Backend: `{BACKEND_URL}`")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
