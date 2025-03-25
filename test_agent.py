"""
Test script for the Technical Documentation Agent.

This script demonstrates how to use the agent for querying technical documentation.
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.agent import agent

# Load environment variables
load_dotenv('.env')

def test_agent_with_query(query: str):
    """
    Test the agent with a specific query.
    
    Args:
        query: The question to ask the agent
        
    Returns:
        The agent's response
    """
    print(f"Query: {query}")
    print("Processing...")
    
    # Create a message with the query
    message = HumanMessage(content=query)
    
    # Invoke the agent
    result = agent.invoke({"messages": [message]})
    
    # Extract and print the response
    if result and "messages" in result and len(result["messages"]) > 0:
        response = result["messages"][-1].content
        files = result["retrieved_files"]
        print(f"\nResponse: {response}")
        print(f"\nFiles: {files}")
        print(20*"-")
        return response
    else:
        print("No response received")
        return None

if __name__ == "__main__":
    # Example queries
    example_queries = [
        "What are the maintenance procedures for a Waukesha engine?",
        "Tell me about cooling system issues in Caterpillar engines",
        "Hello, how are you today?"  # Small talk example
    ]
    
    # Test each query
    for i, query in enumerate(example_queries):
        print(f"\n--- Example {i+1} ---")
        test_agent_with_query(query)
        print("-" * 50) 