"""
Agent graph implementation.

This module implements a state graph for an agent that can answer questions about
technical documentation and engine manuals. The graph defines the flow of information
through different nodes that process user queries, retrieve relevant information,
and generate appropriate responses.
"""
from langgraph.graph import StateGraph, START, END
from agent.utils.state import AgentState
from agent.utils.nodes import (
    initial_router, 
    specify_file_filter, 
    query_generator, 
    retrieve_files, 
    handle_retrieved_files, 
    answer_question, 
    simple_llm
)


def determine_step_after_initial_router(state: AgentState) -> str:
    """
    Determine the next step after the initial router based on the state.
    
    Args:
        state: The current agent state
        
    Returns:
        The name of the next node to execute
    """
    if not state.initial_router_response:
        return "END"
        
    if state.initial_router_response.next_step == "simple_llm":
        return "simple_llm"
    elif state.initial_router_response.next_step == "retrieve_knowledge" and state.current_file_filter:
        return "query_generator"
    elif state.initial_router_response.next_step == "specify_file_filter" or state.current_file_filter is None:
        return "specify_file_filter"
    elif state.initial_router_response.next_step == "retrieve_knowledge" and not state.current_file_filter:
        return "specify_file_filter"
    else:
        return "END"


def determine_step_after_specify_file_filter(state: AgentState) -> str:
    """
    Determine the next step after specifying a file filter.
    
    Args:
        state: The current agent state
        
    Returns:
        The name of the next node to execute
    """
    if state.current_file_filter:
        return "initial_router"
    else:
        return "END"
    

def determine_step_after_handle_retrieved_files(state: AgentState) -> str:
    """
    Determine the next step after handling retrieved files.
    
    Args:
        state: The current agent state
        
    Returns:
        The name of the next node to execute
    """
    if state.final_router_response and state.final_router_response.next_step == "answer_question":
        return "answer_question"
    else:
        return "query_generator"


# Initialize the state graph
graph = StateGraph(AgentState)

# Add all the nodes to the graph
graph.add_node("initial_router", initial_router)
graph.add_node("specify_file_filter", specify_file_filter)
graph.add_node("query_generator", query_generator)
graph.add_node("retrieve_files", retrieve_files)
graph.add_node("handle_retrieved_files", handle_retrieved_files)
graph.add_node("answer_question", answer_question)
graph.add_node("simple_llm", simple_llm)

# Define the edges between nodes
graph.add_edge(START, "initial_router")  

graph.add_conditional_edges(
    "initial_router",
    determine_step_after_initial_router,
    {
        "query_generator": "query_generator",
        "specify_file_filter": "specify_file_filter",
        "simple_llm": "simple_llm",
        "END": END,
    }
)

graph.add_edge("simple_llm", END)

graph.add_conditional_edges(
    "specify_file_filter",
    determine_step_after_specify_file_filter,
    {
        "initial_router": "initial_router",
        "END": END
    }
)

graph.add_edge("query_generator", "retrieve_files")
graph.add_edge("retrieve_files", "handle_retrieved_files")

graph.add_conditional_edges(
    "handle_retrieved_files",
    determine_step_after_handle_retrieved_files,
    {
        "answer_question": "answer_question",
        "query_generator": "query_generator"
    }
)

graph.add_edge("answer_question", END)

# Compilee executable agent
agent = graph.compile()

