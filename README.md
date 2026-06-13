# Groundtruth — AI Match Explainer for Football Fans

> Built for the IBM SkillsBuild AI Builders Challenge · June 2026  
> Solo project · Built by a CS/DS student in 17 days between a Kaggle course and a deadline

---

## What is this?

Ever watched a World Cup match and thought — *why was that offside? What exactly is a high press? Why did the ref give a yellow card for that?*

Groundtruth is a conversational AI that answers those questions in plain English. You ask about a match — tactics, referee decisions, player stats, team strategy — and it explains it back to you like a knowledgeable friend who actually understands football.

It's not a stats dashboard. It's not a prediction tool. It's an **explainer** — built specifically for fans who want to understand the game better, not just watch it.

<!-- ---

## Demo

> 📹 [Watch the 3-minute demo →](#) *(link updated after submission)*

**Example questions you can ask:**
- "Why was Mbappé's goal disallowed in the 67th minute?"
- "How did Morocco's defensive block work against Spain?"
- "What does 'pressing triggers' mean and did India use them?"
- "Explain the offside trap India used in the second half" -->

---

## How it works

```
You ask a question
        ↓
Langflow pipeline routes it
        ↓
Football Data API pulls match context (live + historical)
        ↓
IBM Granite LLM generates a plain-English explanation
        ↓
You get an answer that actually makes sense
```

The key thing: every answer includes **why** — not just what happened, but the tactical or rule-based reasoning behind it. That's the "explainability" part. IBM Granite is specifically good at this kind of structured, grounded reasoning.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | IBM Granite (via watsonx.ai) | Required for challenge + good at structured explanations |
| Pipeline | Langflow | Visual agent pipeline — easy to reason about and show in demo |
| Football data | football-data.org API (free) | Live + historical World Cup match data |
| Backend | FastAPI (Python) | Lightweight, async, great for AI APIs |
| Frontend | Streamlit | Fastest way to build a clean chat UI solo |
| Deployment | Render (free tier) | Simple deployment, stays free |

---

## Running it locally

**You'll need:**
- Python 3.11+
- IBM watsonx.ai API key (free tier available via IBM SkillsBuild)
- football-data.org API key (free)
- Langflow installed

```bash
# Clone the repo
git clone https://github.com/KaustubhMukdam/Groundtruth
cd Groundtruth

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# → Fill in your IBM_API_KEY and FOOTBALL_API_KEY in .env

# Start Langflow
langflow run

# In a separate terminal, start the backend
uvicorn backend.main:app --reload

# Open the UI
streamlit run frontend/app.py
```

That's it. No Docker, no complex setup. It should work on any machine.

---

## Project structure

```
Groundtruth/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── routers/
│   │   └── chat.py          # /ask endpoint
│   ├── services/
│   │   ├── granite.py       # IBM Granite integration
│   │   ├── football_api.py  # football-data.org client
│   │   └── explainer.py     # Core explanation logic
│   └── models/
│       └── schemas.py       # Pydantic request/response models
├── frontend/
│   └── app.py               # Streamlit chat interface
├── langflow/
│   └── Groundtruth_flow.json  # Exportable Langflow pipeline
├── docs/                    # Full project documentation
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## IBM tools used

- **IBM Granite** — the core LLM for generating explanations. Specifically using `granite-13b-instruct` via the watsonx.ai API. Chosen because it's strong at instruction-following and produces structured, factual explanations rather than hallucinated summaries.
- **Langflow** — the visual pipeline that orchestrates the full flow: user input → context fetch → prompt construction → Granite → response. The Langflow JSON export is in `/langflow/` — you can import it directly.

---

## What I learned building this

This was my first time using IBM's AI stack. Langflow took about 2 hours to get comfortable with — the visual flow builder is genuinely useful for understanding data flow. Granite's instruction-following is solid, especially when you give it structured context (match data + user question) rather than open-ended prompts.

The hardest part was prompt engineering for explainability — getting Granite to explain *why* something happened, not just describe it. The solution was structuring the context as: `[Rule/Tactic name] + [What happened in the match] + [Explain to a fan who's curious but not expert]`.

---

## Challenges and honest limitations

- The football-data.org free tier has rate limits — the app will be slow under heavy traffic
- Granite explanations are only as good as the match data fed to it — if the API doesn't have granular event data, the explanation is general
- Offside explanations are particularly hard — VAR decisions involve mm-level precision that no public API captures

---

## What's next (post-challenge)

- Add support for player comparison ("Was Neymar's passing better than in 2022?")
- WhatsApp integration so fans can ask from their phones during a match
- Hindi/regional language support for Indian fans

---

## About

Built solo over ~2 weeks while simultaneously completing the Google x Kaggle 5-Day AI Agents course. Most of the backend was written between 9pm and midnight. The Langflow pipeline was designed on paper first, which saved a lot of debugging time.

If you're a judge: thank you for your time. The demo video shows the full working flow — please watch it before reading the code.

If you're a student looking at this after the challenge: the `/docs` folder has every design and planning document I created. Steal the system, it works.

---

*IBM SkillsBuild AI Builders Challenge · June 2026 · India*
