# 🔎 VeriSearch — Multi-Agent Research Assistant

A multi-agent AI system that researches any question on the web and writes a cited report, with a built-in Critic agent that checks every claim before the report is delivered.

**Live demo:** [add your Streamlit link here]

## How it works

Four specialized agents collaborate through a **LangGraph** workflow:

| Agent | Role |
|---|---|
| **Planner** | Breaks the question into focused, searchable sub-questions |
| **Researcher** | Searches the web (Tavily) for each sub-question and collects unique sources |
| **Writer** | Drafts a structured Markdown report, citing every claim as [n] |
| **Critic** | Checks the draft against the sources for uncited or unsupported claims, missing coverage, and invalid citation numbers, then sends it back for revision if needed |

```
START → Planner → Researcher → Writer → Critic ──approved──→ END
                                  ↑          │
                                  └─revise───┘  (max 2 revisions)
```

The Critic combines **LLM-as-judge** review with a **rule-based check** that catches citation numbers pointing to no real source.

## Tech stack

Python · LangGraph · LangChain · Groq (GPT-OSS-120B) · Tavily Search API · Pydantic structured outputs · Streamlit

## Run it locally

```bash
git clone https://github.com/poojan3108/verisearch.git
cd verisearch
pip install -r requirements.txt
cp .env.example .env      # then add your GROQ_API_KEY and TAVILY_API_KEY
streamlit run app.py
```

## Project structure

```
verisearch/
├── agent/
│   ├── config.py   # model + API clients + settings
│   ├── state.py    # shared state passed between agents
│   ├── nodes.py    # Planner, Researcher, Writer, Critic
│   └── graph.py    # LangGraph workflow with the review loop
├── app.py          # Streamlit web app
├── research_demo.ipynb
└── test_setup.py   # checks your API keys work
```