"""The shared memory that all four agents read from and write to."""
import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict, total=False):
    question: str                         # the user's research question
    sub_questions: list[str]              # Planner output
    sources: list[dict]                   # Researcher output: id, title, url, content
    draft: str                            # Writer output
    issues: list[str]                     # Critic feedback for the next revision
    approved: bool                        # Critic verdict
    revisions: int                        # how many times the Writer has rewritten
    log: Annotated[list[str], operator.add]  # step-by-step activity feed for the UI
