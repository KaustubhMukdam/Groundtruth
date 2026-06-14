# Groundtruth: AI Agent for Football Fans

> Built for the IBM SkillsBuild AI Builders Challenge

---

## What is this?

Ever watched a World Cup match and thought, *why was that offside? What exactly is a high press? Why did the ref pull out a yellow card for that challenge?*

Groundtruth is an **AI agent** that answers those questions in plain English. You ask about a match, tactics, referee decisions, player stats, team strategy, and the agent autonomously decides which data to fetch, grounds its answer in real match facts, then checks itself to make sure it explained *why*, not just *what*.

It's not a stats dashboard. It's not a prediction tool. It's an **explainability agent**, built specifically for fans who want to understand the game, not just watch it.

<!-- ---

## Demo

> 📹 [Watch the 3-minute demo →](#) *(updated after submission)*

**Try asking:**
- "Why was Mbappé's goal disallowed in the 67th minute?"
- "How did Morocco's defensive block work against Spain?"
- "What is an offside trap and did India use it?"
- "Why did the referee show a straight red card there?" -->

---

## How the agent works

Groundtruth is a 3-step agentic pipeline, not a chatbot with a fixed answer path:

```
You ask a question
        ↓
IBM Granite, Tool Selector
"Which tool should I call for this question?"
    ├── get_match_events   (goals, cards, VAR decisions, timeline)
    ├── get_player_stats   (goals, assists, form, match history)
    └── get_rule_definition (offside law, VAR protocol, card rules)
        ↓
Tool is called → real match data returned
        ↓
IBM Granite, Explanation Generator
Generates plain-English explanation grounded in the tool output
        ↓
IBM Granite, Self-Critique
"Did I explain WHY, or just WHAT? If just WHAT, revise."
        ↓
Final answer → You
```

The key difference from a chatbot: **Granite decides what to do next**. It picks the tool. It checks its own work. If the explanation only describes what happened without explaining why, it rewrites itself. That's the agentic behaviour.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | IBM Granite (`granite-13b-instruct-v2`) | Required for challenge + excellent at structured instruction-following; reliable for self-critique |
| Agent pipeline | Langflow | Visual agentic pipeline, tool nodes, decision routing, exportable JSON for judges |
| Football data | football-data.org API (free) | Live + historical World Cup match data, clean REST API |
| Rules data | Local JSON (`football_rules.json`) | FIFA laws don't change mid-tournament; local = zero latency |
| Backend | FastAPI (Python 3.11) | Async, lightweight, automatic OpenAPI docs |
| Frontend | Streamlit | Fastest way to ship a working chat UI solo in a hackathon |
| Deployment | Render (free tier) | Simple Python deployment, no Docker needed |

---

## IBM tools used

- **IBM Granite**, `granite-13b-instruct-v2` via watsonx.ai. Used in 3 places: tool selection, explanation generation, and self-critique. Chosen for its reliable instruction-following behaviour on structured prompts.
- **Langflow**, Visual agentic pipeline that orchestrates all 3 Granite calls + tool nodes. The full pipeline is exported as `/langflow/Groundtruth_agent_flow.json`, importable directly.

---

## Running it locally

**You'll need:**
- Python 3.11+
- IBM watsonx.ai API key + Project ID (free via IBM SkillsBuild)
- football-data.org API key (free registration)
- Langflow installed locally

```bash
# Clone the repo
git clone https://github.com/KaustubhMukdam/Groundtruth
cd Groundtruth

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Fill in IBM_API_KEY, IBM_PROJECT_ID, FOOTBALL_API_KEY, LANGFLOW_URL in .env

# Start Langflow (in a separate terminal)
langflow run
# Import Groundtruth_agent_flow.json from the Langflow UI
# Copy the pipeline's API endpoint URL → paste as LANGFLOW_URL in .env

# Start the backend
uvicorn backend.main:app --reload

# Start the frontend
streamlit run frontend/app.py
```

---

## Project structure

```
Groundtruth/
├── backend/
│   ├── main.py                      # FastAPI entry point
│   ├── routers/
│   │   ├── chat.py                  # POST /ask, main agent endpoint
│   │   └── matches.py               # GET /matches, match list for UI
│   ├── services/
│   │   ├── langflow_client.py       # Calls Langflow pipeline via REST
│   │   ├── football_api.py          # football-data.org client + cache
│   │   └── rules_loader.py          # Loads local football_rules.json
│   └── models/
│       └── schemas.py               # AskRequest, AskResponse, AgentTrace
├── frontend/
│   └── app.py                       # Streamlit chat UI
├── langflow/
│   └── Groundtruth_agent_flow.json    # Full Langflow pipeline, import this
├── data/
│   └── football_rules.json          # FIFA rules + tactic definitions
├── docs/                            # Full project documentation
└── tests/
```

---

## What the self-critique actually does

This is the part I'm most proud of. After Granite generates an explanation, it runs a second pass with this check:

> *"Does this explain WHY the decision was made? Is it grounded in specific match facts? Would a fan who didn't watch the match understand it? If any answer is NO, rewrite."*

On tactic and referee questions, this revision step measurably improves the output. The before/after examples are in `docs/experiment_log.md`.

---

## Honest limitations

- football-data.org free tier has a 10 calls/minute rate limit, first-time requests for uncached matches will be slightly slower
- No VAR frame-by-frame detail in the API, offside VAR explanations describe the process correctly but can't show the exact millimetre measurement
- 3 Granite calls per question = 5–7 second response time on the free tier. Noted in demo video.
- Render free tier sleeps after 15 minutes, first request after sleep takes ~30 seconds

---

## What I learned building this

First time with IBM's AI stack. Langflow was surprisingly fast to pick up, the visual pipeline makes it easy to see exactly where data is flowing and which Granite call is doing what. The self-critique prompt took the most iteration, getting Granite to check against "WHY vs WHAT" reliably required about 8 prompt versions.

I'm also a big fan of the Streamlit chat UI, it's so fast and easy to prototype a working chatbot in a hackathon.

---

## About

The `/docs` folder has every planning and design document created for this project: PRD, architecture, experiment log, evaluation rubric, prompt iteration notes. Built following my own developer documentation system.

If you're a judge: the demo video shows the complete agent flow including the tool-selection step and self-critique revision. The Langflow pipeline JSON is in `/langflow/` and imports in under 2 minutes.

---

*IBM SkillsBuild AI Builders Challenge · "AI Inside the Match" · June 2026 · India*