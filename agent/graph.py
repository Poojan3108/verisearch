"""Wires the four agents into a LangGraph workflow with a review loop."""
from langgraph.graph import END, START, StateGraph

from agent.config import MAX_REVISIONS
from agent.nodes import critic, planner, researcher, writer
from agent.state import ResearchState


def route_after_critic(state: ResearchState) -> str:
    if state.get("approved") or state.get("revisions", 0) >= MAX_REVISIONS:
        return "done"
    return "revise"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_node("critic", critic)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"revise": "writer", "done": END})
    return g.compile()


def format_sources(sources: list[dict]) -> str:
    """Markdown reference list appended to the final report."""
    return "\n".join(f"{s['id']}. [{s['title']}]({s['url']})" for s in sources)
