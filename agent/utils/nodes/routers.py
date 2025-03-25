"""
Router node implementations.

This module contains the implementation of router nodes used in the agent graph.
Router nodes are responsible for making decisions about the next steps in the workflow.
"""
import logging
from typing import Dict, Any

from agent.utils.state import AgentState, InitialRouterOptions, FinalRouterOptions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from agent.utils.nodes.data import FILE_OPTIONS

# Configure logging
logger = logging.getLogger(__name__)

def initial_router(state: AgentState) -> Dict[str, Any]:
    """
    Initial router node that determines the first step in processing a user query.
    
    This node analyzes the conversation context and current file filter to decide
    whether to retrieve knowledge, specify a file filter, or use a simple LLM response.
    It acts as the main decision point for routing the user's query through the
    appropriate pathway in the graph.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary with updated state fields
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(InitialRouterOptions)
    
    prompt = PromptTemplate.from_template(
        """
        You're acting as a router for a workflow that answers questions about engine manuals and technical documentation.
        Your job is to determine which node should be executed next based on the user's question and conversation context.
        
        This system has access to the following engine manuals:
        {file_options}
        
        You have three possible options:
        
        Option 1 - 'retrieve_knowledge': Use this when the user is asking a specific technical question about engine manuals AND we already know which manual to search (current_file_filter is set), or when latest message from ai says that we should proceed with retrival.
        
        Option 2 - 'specify_file_filter': Use this when either:
           - We don't yet know which manual to search (current_file_filter is not set)
           - The user is asking about a different manual than the current one
           - The user has explicitly asked to change the manual
        
        Option 3 - 'simple_llm': Use this when there's no specific, clear question to answer, such as greetings, thanks, general chit-chat, or vague inquiries that don't require searching manuals.
        
        Current file filter: {current_file_filter}
        Conversation with user: {conversation}
        
        Examples:
        
        Example 1:
        Current file filter: None
        User: "How do I change the oil in a Caterpillar engine?"
        Decision: 'specify_file_filter' (because we need to confirm which manual to use)
        
        Example 2:
        Current file filter: "caterpillar.pdf"
        User: "What's the recommended maintenance schedule?"
        Decision: 'retrieve_knowledge' (because we know which manual to search)
        
        Example 3:
        Current file filter: "caterpillar.pdf"
        User: "Can you tell me about Waukesha engines instead?"
        Decision: 'specify_file_filter' (because user wants to change manuals)
        
        Example 4:
        Current file filter: "waukesha.pdf"
        User: "Thanks for your help!"
        Decision: 'simple_llm' (because this is general conversation)
        
        Choose the most appropriate option based on the current conversation context.
        """
    )
    
    answer = llm.invoke(prompt.format(
        conversation=state.messages, 
        current_file_filter=state.current_file_filter,
        file_options=FILE_OPTIONS
    ))
    
    logger.info(f"Initial router decision: {answer.next_step}")
    
    return {"initial_router_response": answer, "retrieved_files": []}

def handle_retrieved_files(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates if retrieved documents contain sufficient information to answer the query.
    
    This node analyzes the retrieved documents and decides whether to
    answer the question or retrieve more information, with precise identification
    of a single missing information element to focus the next search iteration.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary with updated state fields
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(FinalRouterOptions)
    
    prompt = PromptTemplate.from_template(
        """
        Analyze if the retrieved documents have enough information to answer the user's question about engine manuals.
        
        User conversation:
        {conversation}
        
        Retrieved documents:
        {retrieved_files}
        
        Your task:
        1. Identify what information is needed to answer the user's latest question
        2. Check if this information exists in the retrieved documents
        3. Make one of two decisions:
           - If ALL necessary information is present: choose 'answer_question'
           - If ANY critical information is missing: choose 'retrieve_knowledge'
        
        If choosing 'retrieve_knowledge', specify ONLY ONE specific missing piece of information.
        Be precise - mention one technical term, part number, procedure, or specification that is still needed.
        
        Provide your decision without lengthy explanation.
        """
    )
    
    conversation = "\n".join([msg.content for msg in state.messages])
    response = llm.invoke(prompt.format(
        retrieved_files=state.retrieved_files, 
        conversation=conversation
    ))

    # Update requery counter
    requeries = state.number_of_requeries
    if response.next_step == "retrieve_knowledge":
        requeries += 1
    
    # If number of requeries exceeds 4, force answer_question
    if requeries > 4:
        logger.warning("Max requeries reached. Forcing answer_question.")
        response.next_step = "answer_question"
    
    logger.info(f"Handle retrieved files decision: {response.next_step}, requeries: {requeries}")
    
    return {"final_router_response": response, "number_of_requeries": requeries} 