"""LangGraph StateGraph assembly."""
from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.agents.data_analyst import run as data_analyst_run
from src.agents.feature_ideator import run as feature_ideator_run
from src.agents.feature_coder import run as feature_coder_run
from src.agents.feature_evaluator import run as feature_evaluator_run
from src.agents.output_writer import run as output_writer_run


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("data_analyst", data_analyst_run)
    graph.add_node("feature_ideator", feature_ideator_run)
    graph.add_node("feature_coder", feature_coder_run)
    graph.add_node("feature_evaluator", feature_evaluator_run)
    graph.add_node("output_writer", output_writer_run)

    graph.set_entry_point("data_analyst")
    graph.add_edge("data_analyst", "feature_ideator")
    graph.add_edge("feature_ideator", "feature_coder")
    graph.add_edge("feature_coder", "feature_evaluator")
    graph.add_edge("feature_evaluator", "output_writer")
    graph.add_edge("output_writer", END)

    return graph.compile()
