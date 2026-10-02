"""Shared settings and clients for every agent."""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"   # Groq production model (Oct 2026)
MAX_SUB_QUESTIONS = 4                # how many angles the Planner researches
RESULTS_PER_QUERY = 3                # web results per sub-question
MAX_REVISIONS = 2                    # Critic can send the draft back this many times
SEARCH_DEPTH = "advanced"            # higher-quality, more relevant results than "basic"

# Low-reliability sources the Researcher should never cite
EXCLUDED_DOMAINS = [
    "youtube.com", "tiktok.com", "instagram.com", "facebook.com", "x.com",
    "twitter.com", "reddit.com", "quora.com", "pinterest.com", "linkedin.com",
]


def get_llm(temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=temperature)


def get_search_client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])