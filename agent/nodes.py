"""The four agents: Planner, Researcher, Writer, Critic."""
import re
from datetime import date

from pydantic import BaseModel, Field

from agent.config import (
    EXCLUDED_DOMAINS, MAX_SUB_QUESTIONS, RESULTS_PER_QUERY, SEARCH_DEPTH,
    get_llm, get_search_client,
)
from agent.state import ResearchState


# ---------- Structured outputs ----------
class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description="Short web search queries (under 20 words each) that together answer the main question"
    )


class Review(BaseModel):
    approved: bool = Field(description="True only if the report needs no changes")
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems to fix; empty if approved",
    )


# ---------- Helper: retry structured calls ----------
def structured_call(schema, prompt: str, temperature: float = 0.2, retries: int = 3):
    """Groq sometimes rejects a structured-output call; retry before giving up."""
    last_error = None
    for _ in range(retries):
        try:
            return get_llm(temperature).with_structured_output(schema).invoke(prompt)
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise last_error


# ---------- 1. Planner ----------
def planner(state: ResearchState) -> dict:
    plan = structured_call(Plan, 
        f"You are a research planner. Break this question into at most "
        f"{MAX_SUB_QUESTIONS} short, focused sub-questions (under 20 words each) that can "
        f"each be answered with a web search. Cover different angles (facts, recent developments, "
        f"pros/cons, numbers). Today's date is {date.today():%B %d, %Y}; for questions "
        f"about recent or latest developments, focus on the most recent period.\n\n"
        f"Question: {state['question']}"
    )
    subs = plan.sub_questions[:MAX_SUB_QUESTIONS]
    return {
        "sub_questions": subs,
        "revisions": 0,
        "log": [f"Planner: split the question into {len(subs)} sub-questions"],
    }


# ---------- 2. Researcher ----------
def researcher(state: ResearchState) -> dict:
    client = get_search_client()
    sources, seen_urls = [], set()
    for sub_q in state["sub_questions"]:
        try:
            results = client.search(
                sub_q[:390],
                max_results=RESULTS_PER_QUERY,
                search_depth=SEARCH_DEPTH,
                exclude_domains=EXCLUDED_DOMAINS,
            )
        except Exception:  # noqa: BLE001  (one failed search shouldn't stop the run)
            continue
        for r in results.get("results", []):
            if r["url"] in seen_urls:
                continue  # skip duplicate pages
            seen_urls.add(r["url"])
            sources.append({
                "id": len(sources) + 1,
                "title": r.get("title", "Untitled"),
                "url": r["url"],
                "content": r.get("content", "")[:1500],
            })
    return {
        "sources": sources,
        "log": [f"Researcher: collected {len(sources)} unique sources"],
    }


# ---------- 3. Writer ----------
def _format_sources(sources: list[dict]) -> str:
    return "\n\n".join(
        f"[{s['id']}] {s['title']}\n{s['content']}" for s in sources
    )


def normalize_citations(text: str) -> str:
    """Models sometimes cite as 【1】 or ［1］; convert everything to [1]."""
    return re.sub(r"[【［]\s*(\d+)\s*[】］]", r"[\1]", text)


def writer(state: ResearchState) -> dict:
    feedback = ""
    if state.get("issues"):
        feedback = (
            "\n\nA reviewer found these problems in your previous draft. Fix all of them:\n"
            + "\n".join(f"- {i}" for i in state["issues"])
            + f"\n\nPrevious draft:\n{state['draft']}"
        )
    prompt = (
        "You are a research writer. Write a clear, well-structured report in Markdown "
        "that answers the question using ONLY the numbered sources below.\n"
        "Rules:\n"
        "- Cite every factual claim with its source number in plain square brackets, e.g. [2].\n"
        "- If sources give conflicting numbers, state that they differ and cite each.\n"
        "- Only use source numbers that exist below. Never invent facts.\n"
        "- Use short sections with ## headings and end with a '## Key Takeaways' section.\n"
        "- Do NOT add a references list; it is added automatically.\n"
        f"- Today's date is {date.today():%B %d, %Y}.\n\n"
        f"Question: {state['question']}\n\n"
        f"Sub-questions to cover:\n" + "\n".join(f"- {q}" for q in state["sub_questions"])
        + f"\n\nSources:\n{_format_sources(state['sources'])}"
        + feedback
    )
    draft = normalize_citations(get_llm(temperature=0.3).invoke(prompt).content)
    revisions = state.get("revisions", 0) + (1 if state.get("issues") else 0)
    action = "revised the report" if state.get("issues") else "wrote the first draft"
    return {"draft": draft, "revisions": revisions, "log": [f"Writer: {action}"]}


# ---------- 4. Critic ----------
def find_invalid_citations(draft: str, num_sources: int) -> list[int]:
    """Rule-based check: citation numbers that point to no real source."""
    cited = {int(n) for n in re.findall(r"[\[【［]\s*(\d+)\s*[\]】］]", draft)}
    return sorted(n for n in cited if n < 1 or n > num_sources)


def critic(state: ResearchState) -> dict:
    try:
        review = structured_call(Review, temperature=0, prompt=(
        "You are a strict research reviewer. Check the report against the sources.\n"
        "Reject it if: a factual claim has no citation, a claim is not supported by "
        "the cited source, a sub-question is not answered, or conflicting figures "
        "from different sources are presented without noting the disagreement.\n"
        "Approve it if it is accurate, fully cited, and complete.\n\n"
        f"Sub-questions:\n" + "\n".join(f"- {q}" for q in state["sub_questions"])
        + f"\n\nSources:\n{_format_sources(state['sources'])}"
        + f"\n\nReport:\n{state['draft']}"
        ))
    except Exception:  # noqa: BLE001  (keep the report rather than crash)
        return {"approved": True, "issues": [],
                "log": ["Critic: review call failed after retries; kept current draft"]}
    issues = list(review.issues)
    bad = find_invalid_citations(state["draft"], len(state["sources"]))
    if bad:
        issues.append(f"Citations {bad} do not match any source; remove or fix them.")
    approved = review.approved and not bad
    verdict = "approved the report" if approved else f"requested {len(issues)} fixes"
    return {"approved": approved, "issues": issues, "log": [f"Critic: {verdict}"]}