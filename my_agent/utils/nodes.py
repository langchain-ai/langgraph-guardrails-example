from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from typing import TypedDict


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)


about_weather_prompt = """Determine whether the user's most recent question is about weather."""

class AboutWeather(TypedDict):
    """Is the user's question about weather?"""
    about_weather: bool

def weather_guardrail(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": about_weather_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name).with_structured_output(AboutWeather)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"about_weather": response['about_weather']}


responds_in_english_prompt = """Determine whether the final assistant response is in English or not."""


class IsEnglish(TypedDict):
    """Is the final assistant response in English?"""
    is_english: bool


def english_guardrail(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": responds_in_english_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name).with_structured_output(IsEnglish)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"is_english": response['is_english']}