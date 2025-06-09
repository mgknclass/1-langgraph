import operator
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages.base import BaseMessage


class TopicSelectionParser(BaseModel):
    topic: str = Field(description="Topic Selected")
    reason: str = Field(description="Reason for Topic Selection")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
