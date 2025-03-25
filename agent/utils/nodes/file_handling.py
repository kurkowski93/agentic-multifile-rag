"""
File handling node implementations.

This module contains nodes responsible for file selection and retrieval operations,
such as determining which files to use and retrieving content from the vector store.
"""
import logging
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from agent.utils.state import AgentState
from agent.utils.nodes.data import FILE_OPTIONS
from agent.utils.nodes.config import vector_store

# Configure logging
logger = logging.getLogger(__name__)

class FileSpecifierOutput(BaseModel):
    """
    Output model for the file specifier node.
    
    Defines the structure for the file specification response.
    """
    filename: Optional[str] = Field(
        default=None, 
        description="The exact name of the file to retrieve. Leave empty unless you are 100% certain based on explicit mention in the conversation."
    )
    response: str = Field(
        description="A question to the user to specify the file name or information that you selected proper file based on context"
    )


def specify_file_filter(state: AgentState) -> Dict[str, Any]:
    """
    Node that helps determine which file to use for information retrieval.
    
    This node either asks the user to specify a file or determines the appropriate
    file based on conversation context.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary with updated state fields
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(FileSpecifierOutput)
    
    prompt = PromptTemplate.from_template(
        """
        Here's the conversation: {conversation}
   
        Here's the list of possible file filters: {file_options} \n\n 
        Please ask user to specify one of them, or do it yourself if you can figure it out based on provided information.
        
        IMPORTANT: Only set the filename if it was explicitly mentioned in the conversation or if you are 100% certain 
        which file the user needs based on clear context. If there is any ambiguity or the user hasn't clearly indicated 
        which file they need, leave the filename empty and ask the user to specify.
        
        If you decide to select a filename, your response must be in the format: "we're going to work with file: [filename] lets proceed with retrieval"
        """
    )
    
    response = llm.invoke(prompt.format(
        conversation=state.messages, 
        file_options=FILE_OPTIONS
    ))
    
    logger.info(f"File filter specified: {response.filename}")
    
    # Return updated state fields
    return {
        "messages": AIMessage(content=response.response),
        "current_file_filter": response.filename
    }


def retrieve_files(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves relevant documents from the vector store based on the query.
    
    This node executes the search against the vector database using the
    generated query and the specified file filter.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary with updated state fields containing retrieved documents
    """
    logger.info(f"Retrieving files with filter: {state.current_file_filter}")
    
    if not vector_store:
        logger.error("Vector store not initialized. Cannot retrieve files.")
        return {"retrieved_files": []}
    
    try:
        files = vector_store.similarity_search(
            state.messages[-1].content, 
            k=3,   
            filter={
                "must": [
                    {
                        "key": "metadata.filename",
                        "match": {"value": state.current_file_filter}
                    }
                ] 
            }
        )
        
        logger.info(f"Retrieved {len(files)} documents")
        return {"retrieved_files": files + state.retrieved_files}
    except Exception as e:
        logger.error(f"Error retrieving files: {e}")
        return {"retrieved_files": state.retrieved_files} 