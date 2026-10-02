"""Step 1: confirm the LLM (Groq) and web search (Tavily) both work."""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient

load_dotenv()

# 1. Test the LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
reply = llm.invoke("Reply with exactly: setup works")
print("LLM says:", reply.content)

# 2. Test web search
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
results = tavily.search("latest EV battery breakthroughs", max_results=2)
print("\nSearch results:")
for r in results["results"]:
    print("-", r["title"], "|", r["url"])
