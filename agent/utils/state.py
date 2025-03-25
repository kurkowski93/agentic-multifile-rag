from typing import Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class InitialRouterOptions(BaseModel):
    """
    Options for the initial router decision.
    
    This model defines the possible next steps after the initial routing decision.
    """
    next_step: str = Field(
        description="The next step to execute. Has to be one of the following: 'retrieve_knowledge', 'specify_file_filter', 'simple_llm'")  

class FinalRouterOptions(BaseModel):
    """
    Options for the final router decision.
    
    This model defines whether to answer the question with current information
    or retrieve more knowledge.
    """
    next_step: str = Field(description="The next step to execute. Has to be one of the following: 'answer_question', 'retrieve_knowledge'")
    lacking_informations: Optional[str] = Field(None, description="Indicates what specific information the agent is lacking, e.g., 'How to do X?'")

class AgentState(BaseModel):
    """
    Represents the state of the agent throughout the graph execution.
    
    This class maintains the conversation history, routing decisions,
    file filters, and retrieved information during the agent's operation.
    """
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    initial_router_response: Optional[InitialRouterOptions] = Field(default=None)
    final_router_response: Optional[FinalRouterOptions] = Field(default=None)
    current_file_filter: Optional[str] = Field(default=None)
    retrieved_files: List[Document] = Field(default_factory=list)
    number_of_requeries: int = Field(default=0)
    
    class Config:
        arbitrary_types_allowed = True
    
    
