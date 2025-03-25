"""
Query generation node implementations.

This module contains nodes responsible for generating optimized queries
for the retrieval system based on conversation context and user questions.
"""
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from agent.utils.state import AgentState

# Configure logging
logger = logging.getLogger(__name__)

def query_generator(state: AgentState) -> Dict[str, Any]:
    """
    Generates an optimized query for the retrieval system based on conversation context.
    
    This node reformulates the user's question into a more effective search query
    for the vector database by extracting key technical terms and focusing on
    specific information needs.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary with updated state fields
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    lacking_info = ""
    if state.final_router_response and state.final_router_response.lacking_informations:
        lacking_info = state.final_router_response.lacking_informations
    
    # If we have specific lacking information, we'll focus only on that
    if lacking_info:
        prompt = PromptTemplate.from_template(
            """
            You are a query generator for a Retrieval-Augmented Generation (RAG) system that searches engine manuals.
            
            Your task is to generate a precise, specific query that will retrieve the most relevant information from technical documents.
            
            We are specifically looking for information about: {lacking_info}
            
            Examples of good queries:
            - Original question: "How often should I change the oil in a Waukesha VGF engine?"
              Good query: "Waukesha VGF engine oil change interval maintenance schedule"
            
            - Original question: "What are the cooling system specifications for a Caterpillar 3500?"
              Good query: "Caterpillar 3500 cooling system specifications temperature pressure coolant"
            
            Create a search query that:
            1. Focuses ONLY on finding this specific information with extreme specificity
            2. Includes specific technical terms, part numbers, model names, and exact keywords mentioned
            3. Uses technical terminology exactly as it appears in manuals
            4. Is formulated as a precise search query (not a question)
            5. Combines the most important 4-6 technical terms related to the information need
            
            Output a highly focused, technical search query only, without explanation.
            """
        )
        query = llm.invoke(prompt.format(lacking_info=lacking_info))
    else:
        prompt = PromptTemplate.from_template(
            """
            You are a query generator for a Retrieval-Augmented Generation (RAG) system that searches engine manuals.
            
            Your task is to generate a precise, specific query that will retrieve the most relevant information from technical documents.
            
            Examples of good queries:
            - Original question: "How often should I change the oil in a Waukesha VGF engine?"
              Good query: "Waukesha VGF engine oil change interval maintenance schedule"
            
            - Original question: "What are the cooling system specifications for a Caterpillar 3500?"
              Good query: "Caterpillar 3500 cooling system specifications temperature pressure coolant"
            
            - Original question: "How do I troubleshoot low power output on my generator?"
              Good query: "generator troubleshooting low power output diagnosis causes solutions"
            
            Based on the conversation history below, create a search query that:
            1. Focuses on the user's core information need with extreme specificity
            2. Includes specific technical terms, part numbers, model names, and exact keywords mentioned
            3. Prioritizes the most recent question/request
            4. Uses technical terminology exactly as it appears in manuals
            5. Includes specific parameters, measurements, or specifications mentioned
            6. Avoids generic terms and uses domain-specific language
            7. Is formulated as a precise search query (not a question)
            8. Combines the most important 4-6 technical terms related to the information need
            
            Conversation history:
            {conversation}
            
            Output a highly focused, technical search query only, without explanation.
            """
        )
        
        conversation = "\n".join([msg.content for msg in state.messages])
        query = llm.invoke(prompt.format(conversation=conversation))
    
    logger.info(f"Generated query: {query.content}")
    
    # Return updated state field
    return {"messages": query} 