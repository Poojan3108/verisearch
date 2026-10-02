"""Streamlit web app: run with  streamlit run app.py"""
import time

import streamlit as st

from agent.graph import build_graph, format_sources

st.set_page_config(page_title="VeriSearch", page_icon="🔎", layout="wide")
st.title("🔎 VeriSearch")
st.subheader("Multi-Agent Research Assistant")
st.caption("Planner → Researcher → Writer → Critic, built with LangGraph, Groq, and Tavily")

with st.form("research_form"):
    question = st.text_input(
        "What do you want to research?",
        placeholder="e.g. What are the latest breakthroughs in solid-state EV batteries?",
    )
    submitted = st.form_submit_button("Start research", type="primary")

if submitted and not question.strip():
    st.warning("Please type a research question first.")

if submitted and question.strip():
    graph = build_graph()
    final = {}
    start = time.time()

    with st.status("Agents are working...", expanded=True) as status:
        for update in graph.stream({"question": question}, stream_mode="updates"):
            for node, output in update.items():
                final.update(output)
                for line in output.get("log", []):
                    st.write("✅ " + line)
                if node == "planner":
                    for q in output["sub_questions"]:
                        st.write(f"   • {q}")
                if node == "critic" and output.get("issues"):
                    for issue in output["issues"]:
                        st.write(f"   ⚠️ {issue}")
        status.update(label="Research complete", state="complete", expanded=False)

    elapsed = time.time() - start
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sub-questions", len(final.get("sub_questions", [])))
    c2.metric("Sources", len(final.get("sources", [])))
    c3.metric("Revisions", final.get("revisions", 0))
    c4.metric("Time", f"{elapsed:.0f}s")

    if not final.get("approved"):
        st.warning("The Critic still had concerns after the maximum revisions; review citations carefully.")

    report = final["draft"] + "\n\n## Sources\n" + format_sources(final["sources"])
    st.markdown(report)
    st.download_button("Download report (.md)", report, file_name="research_report.md")