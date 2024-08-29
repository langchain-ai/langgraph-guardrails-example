from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, should_continue, tool_node, weather_guardrail
from my_agent.utils.state import AgentGuardrailBeforeState

# Define logic to determine whether question is about the weather
def is_about_weather(state: AgentGuardrailBeforeState) -> Literal['hardcoded_response', 'agent']:
    if not state['about_weather']:
        return "hardcoded_response"
    else:
        return "agent"

def hardcoded_response(state):
    return {"messages": [{"role": "assistant", "content": "sorry I can only answer questions about weather"}]}


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentGuardrailBeforeState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node(weather_guardrail)
workflow.add_node(hardcoded_response)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("weather_guardrail")

workflow.add_conditional_edges("weather_guardrail", is_about_weather)

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")
workflow.add_edge("hardcoded_response", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
