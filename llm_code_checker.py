from langgraph.graph.state import CompiledStateGraph

COLOR_OK = '\033[92m' # light green
COLOR_DEFAULT = '\033[0m'

import os
COLOR_OK = '\033[92m' # light green
COLOR_DEFAULT = '\033[0m'

# read config
print("importing config/config.ini ", end='')
import configparser
config = configparser.ConfigParser()
config.read("config/config.ini")
ANTHROPIC_API_KEY = config.get("AnthropicAPI", "ANTHROPIC_API_KEY")
OPEN_AI_KEY = config.get("OpenAI", "OPEN_AI_KEY")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

print("set OpenAI & Anthropic API keys ", end='')
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
print(COLOR_OK + "OK" + COLOR_DEFAULT)

from langgraph_reflection import create_reflection_graph
from langgraph_reflection import StateGraph
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict
from openevals.llm import create_llm_as_judge

# Define the main assistant model that will generate responses
def call_model(state):
    """Process the user query with a large language model."""
    model = init_chat_model(model="claude-3-7-sonnet-latest")
    return {"messages": model.invoke(state["messages"])}


# Define a basic graph for the main assistant
assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)

from openevals.code.pyright import create_pyright_evaluator

# Function that validates code using Pyright
def try_running(state: dict) -> dict | None:
    """Attempt to run and analyze the extracted Python code."""
    # Extract code from the conversation
    # code = extract_python_code(state['messages'])
    code = "print \"Hello World!\""
    # code = ""

    # Run Pyright analysis
    evaluator = create_pyright_evaluator()
    result = evaluator(outputs=code)

    if not result['score']:
        # If errors found, return critique for the main agent
        return {
            "messages": [{
                "role": "user",
                "content": f"I ran pyright and found this: {result['comment']}\n\n"
                          "Try to fix it..."
            }]
        }
    # No errors found - return None to indicate success
    return None

# Create graphs with reflection
judge_graph = (StateGraph(MessagesState)
    .add_node(try_running)
    .add_edge(START, "try_running")
    .add_edge("try_running", END)
    .compile()
)

# Create reflection system that combines code generation and validation
reflection_app = create_reflection_graph(assistant_graph, judge_graph).compile()

example_query = [
    {
        "role": "user",
        "content": "Please check the specified code with PyCheck",
    }
]
result = reflection_app.invoke({"messages": example_query})
print("Result: ", result)
