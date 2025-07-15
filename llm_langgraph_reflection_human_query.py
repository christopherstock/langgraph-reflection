import configparser
config = configparser.ConfigParser()
config.read("config/config.ini")
print("importing config/config.ini OK")

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

class ExtractPythonCode(TypedDict):
    python_code: str

class NoCode(TypedDict):
    no_code: bool

EVALUATION_PROMPT = """You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response in the conversation below.

Evaluate the response based on these criteria:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?
6. Content Length - The maximum length of the content may not exceed 1.000 characters

If the response meets ALL criteria satisfactorily, set pass to True.

If you find ANY issues with the response, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

Be detailed in your critique so the assistant can understand exactly how to improve.

<response>
{outputs}
</response>"""

def try_running(state: dict) -> dict | None:
    print()
    print("ℹ️ AI Message from LLM: ")
    print(state["messages"][-1].content)

    print()
    print("ℹ️ Entire State Object: ")
    print(state)


    exit(0)


    model = init_chat_model(model="gpt-4o-mini")
    extraction = model.bind_tools([ExtractPythonCode, NoCode])
    er = extraction.invoke(
        [{"role": "system", "content": EVALUATION_PROMPT}] + state["messages"]
    )
    if len(er.tool_calls) == 0:
        print("Check 1")
        return None
    tc = er.tool_calls[0]
    if tc["name"] != "ExtractPythonCode":
        print("Check 2")
        return None

    code = tc["args"]["python_code"]
    scan_result = analyze_with_pyright(code)

    explanation = scan_result["generalDiagnostics"]

    if scan_result["summary"]["errorCount"]:
        print("Check 3")
        print()
        print("⚠️ Pyright scan found violations:")
        print(scan_result)

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
    else:
        print()
        print("✅️ Pyright approved the code:")
        print(code)

assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)

judge_graph = (
    StateGraph(MessagesState)
    .add_node(try_running)
    .add_edge(START, "try_running")
    .add_edge("try_running", END)
    .compile()
)

reflection_app = create_reflection_graph(assistant_graph, judge_graph).compile()

if __name__ == "__main__":
    example_query = [
        {
            "role": "user",
            "content": "Explain how green energy works and why it's important for our planet.",
        }
    ]

    print("running example with reflection using GPT-4o mini ...")
    result = reflection_app.invoke({"messages": example_query})

    print()
    print("✅ Result accepted")
