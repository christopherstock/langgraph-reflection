# read config
import configparser
config = configparser.ConfigParser()
config.read("config/config.ini")
print("importing config/config.ini OK")

# set API keys
import os
ANTHROPIC_API_KEY = config.get("AnthropicAPI", "ANTHROPIC_API_KEY")
OPEN_AI_KEY = config.get("OpenAI", "OPEN_AI_KEY")
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
print("set OpenAI & Anthropic API keys OK")

from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph_reflection import create_reflection_graph
import json, os, subprocess, tempfile

def analyze_with_pyright(code_string: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
        temp.write(code_string)
        temp_path = temp.name

    try:
        result = subprocess.run(
            [
                "pyright",
                "--outputjson",
                "--level",
                "error",  # Only report errors, not warnings
                temp_path,
            ],
            capture_output=True,
            text=True,
        )

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse Pyright output",
                "raw_output": result.stdout,
            }
    finally:
        os.unlink(temp_path)

def call_model(state: dict) -> dict:
    model = init_chat_model(model="gpt-4o-mini", openai_api_key = OPEN_AI_KEY)
    return {"messages": model.invoke(state["messages"])}

# Define type classes for code extraction
class ExtractPythonCode(TypedDict):
    """Type class for extracting Python code. The python_code field is the code to be extracted."""
    python_code: str

class NoCode(TypedDict):
    """Type class for indicating no code was found."""
    no_code: bool

# Evaluation prompt for the model
EVALUATION_PROMPT = """The below conversation is you conversing with a user to write some python code. Your final response is the last message in the list.

Sometimes you will respond with code, othertimes with a question.

If there is code - extract it into a single python script using ExtractPythonCode.

If there is no code to extract - call NoCode."""

def try_running(state: dict) -> dict | None:
    """Attempt to run and analyze the extracted Python code.

    Args:
        state: The current conversation state

    Returns:
        dict | None: Updated state with analysis results if code was found
    """
    model = init_chat_model(model="gpt-4o-mini")
    extraction = model.bind_tools([ExtractPythonCode, NoCode])
    er = extraction.invoke(
        [{"role": "system", "content": EVALUATION_PROMPT}] + state["messages"]
    )
    if len(er.tool_calls) == 0:
        return None
    tc = er.tool_calls[0]
    if tc["name"] != "ExtractPythonCode":
        return None

    result = analyze_with_pyright(tc["args"]["python_code"])

    print()
    print("Scan Result: ", result)

    explanation = result["generalDiagnostics"]

    if result["summary"]["errorCount"]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"I ran pyright and found this: {explanation}\n\n"
                               "Try to fix it. Make sure to regenerate the entire code snippet. "
                               "If you are not sure what is wrong, or think there is a mistake, "
                               "you can ask me a question rather than generating code",
                }
            ]
        }

def create_graphs():
    """Create and configure the assistant and judge graphs."""
    # Define the main assistant graph
    assistant_graph = (
        StateGraph(MessagesState)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .add_edge("call_model", END)
        .compile()
    )

    # Define the judge graph for code analysis
    judge_graph = (
        StateGraph(MessagesState)
        .add_node(try_running)
        .add_edge(START, "try_running")
        .add_edge("try_running", END)
        .compile()
    )

    # Create the complete reflection graph
    return create_reflection_graph(assistant_graph, judge_graph).compile()

reflection_app = create_graphs()

if __name__ == "__main__":
    """Run an example query through the reflection system."""
    example_query = [
        {
            "role": "user",
            "content": "Write a LangGraph RAG app",
        }
    ]

    print("Running example with reflection using GPT-4o mini...")
    result = reflection_app.invoke({"messages": example_query})
    print()
    print("Final Result:", result)
