"""
Answer generation node implementations.

This module contains nodes responsible for generating responses to user queries,
including both document-based answers and general responses.
"""
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from agent.utils.state import AgentState

# Configure logging
logger = logging.getLogger(__name__)

def answer_question(state: AgentState) -> Dict[str, Any]:
    """
    Generates a response to the user's query based on retrieved documents.
    
    This node formulates a human-readable answer to the user's question
    using the information gathered from the documents and includes proper
    source citations.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary with updated state fields
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Find the last human message
    last_human_message = None
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    if last_human_message is None:
        logger.warning("No human message found in conversation history")
        return {"messages": [AIMessage(content="I couldn't find your question. Please try asking again.")]}
    
    # Prepare document context with metadata for citations
    formatted_docs = []
    for i, doc in enumerate(state.retrieved_files):
        # Extract metadata for citation
        doc_id = i + 1
        filename = doc.metadata.get("filename", "Unknown source")
        
        # Handle different page formats (single value, list, comma-separated string)
        page_info = doc.metadata.get("pages", doc.metadata.get("page", "Unknown page"))
        if isinstance(page_info, list):
            page_info = ", ".join(map(str, page_info))
        
        # Add heading if available
        heading = doc.metadata.get("headings", "")
        heading_info = f", Section: {heading}" if heading and heading != "Unknown" else ""
        
        # Format document with citation ID
        formatted_doc = f"[{doc_id}] {doc.page_content}\nSource: {filename}, Page(s): {page_info}{heading_info}\n\n"
        formatted_docs.append(formatted_doc)
    
    formatted_context = "\n".join(formatted_docs)
    
    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant that can answer questions about engine manuals and technical documentation.
        
        Context from technical documents:
        {context}
        
        Question:
        {question}
        
        When answering:
        1. Use ONLY information from the provided context
        2. Cite your sources using the reference numbers in square brackets [X]
        3. If multiple sources provide relevant information, cite all of them
        4. If the context doesn't contain the answer, say so clearly
        5. Format your answer in a clear, concise way
        6. Include a "Sources" section at the end listing all references used with their page numbers
        7. Be precise with technical terminology and specifications
        
        Answer the question based on the context provided.
        """
    )
    print("formatted_context", formatted_context)
    
    answer = llm.invoke(prompt.format(
        context=formatted_context, 
        question=last_human_message.content
    ))
    
    logger.info("Generated answer with source citations")
    
    # Return updated state field
    return {"messages": [answer]}


def simple_llm(state: AgentState) -> Dict[str, Any]:
    """
    Provides a simple response for general conversation not related to engine manuals.
    
    This node handles small talk and general queries that don't require
    technical document retrieval. It provides helpful prompts to guide the user
    towards asking questions related to engine documentation.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary with updated state fields
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = PromptTemplate.from_template(
        """
        Your job is to ask questions that will help you understand the user's needs. 
        You're a part of a system that answers questions about engine manuals and technical documentation.
        
        Here are some examples of good questions:
        - "Which specific engine model are you interested in learning about?"
        - "Are you looking for maintenance procedures, troubleshooting guides, or specifications?"
        - "Do you need information about a particular component of the engine?"
        - "What specific problem are you trying to solve with the engine?"
        
        Here's the conversation history:
        {conversation}
        
        Ask a specific, focused question that will help the user find the technical information they need.
        Keep your response brief and professional.
        """
    )
    
    answer = llm.invoke(prompt.format(conversation=state.messages))
    
    logger.info("Generated improved simple LLM response")
    
    return {"messages": [answer]} 