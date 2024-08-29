from langgraph.graph import add_messages, MessagesState
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(MessagesState):
    pass


class AgentGuardrailBeforeState(MessagesState):
    about_weather: bool


class AgentGuardrailAfterState(MessagesState):
    is_english: bool
