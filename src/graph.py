"""Сборка LangGraph StateGraph — пайплайн из 3 агентов."""
from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.agents.data_analyst import run as data_analyst_run
from src.agents.feature_engineer import run as feature_engineer_run
from src.agents.evaluator_writer import run as evaluator_writer_run


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("data_analyst", data_analyst_run)
    graph.add_node("feature_engineer", feature_engineer_run)
    graph.add_node("evaluator_writer", evaluator_writer_run)

    graph.set_entry_point("data_analyst")
    graph.add_edge("data_analyst", "feature_engineer")
    graph.add_edge("feature_engineer", "evaluator_writer")
    graph.add_edge("evaluator_writer", END)

    return graph.compile()
