"""
Delve LangGraph Workflow
─────────────────────────
Defines the full research pipeline as a LangGraph StateGraph.

Pipeline flow:
  planner -> retrieval -> summarizer -> proposer <-> critic (debate loop) -> cross_paper -> gap_analysis -> paper_architect
"""

import logging

from langgraph.graph import StateGraph, END
from typing import Any

from app.core.state import ResearchState
from app.core.agents import (
    planner_node,
    retrieval_node,
    summarizer_node,
    proposer_node,
    critic_node,
    cross_paper_analyzer_node,
    gap_analysis_node,
    paper_architect_node,
    debate_should_continue,
)

logger = logging.getLogger("delve.graph")


def build_research_graph(checkpointer: Any = None) -> Any:
    """
    Build and compile the Delve research graph.

    Returns a compiled LangGraph that can be invoked with a ResearchState.
    """

    # ── Create the graph ──────────────────────────────────────────────────
    graph = StateGraph(ResearchState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("proposer", proposer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("cross_paper", cross_paper_analyzer_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("paper_architect", paper_architect_node)

    # ── Set entry point ───────────────────────────────────────────────────
    graph.set_entry_point("planner")

    # ── Linear edges ──────────────────────────────────────────────────────
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "summarizer")
    graph.add_edge("summarizer", "proposer")

    # ── Debate loop: proposer → (critic or gap_analysis) ──────────────────
    graph.add_conditional_edges(
        "proposer",
        debate_should_continue,
        {
            "critic": "critic",
            "gap_analysis": "cross_paper",
        },
    )

    # Critic always sends back to proposer for revision
    graph.add_edge("critic", "proposer")

    # ── Cross-paper -> gap_analysis -> paper architect -> END ─────────────
    graph.add_edge("cross_paper", "gap_analysis")
    graph.add_edge("gap_analysis", "paper_architect")
    graph.add_edge("paper_architect", END)

    logger.info("Research graph built successfully")

    # Compile and return
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled
