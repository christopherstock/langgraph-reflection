import configparser

from openevals import create_llm_as_judge

config = configparser.ConfigParser()
config.read("config/config.ini")
print("importing config/config.ini OK")

import os
ANTHROPIC_API_KEY = config.get("AnthropicAPI", "ANTHROPIC_API_KEY")
OPEN_AI_KEY = config.get("OpenAI", "OPEN_AI_KEY")
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY
print("set OpenAI & Anthropic API keys OK")

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph_reflection import create_reflection_graph

def call_model(state: dict) -> dict:
    model = init_chat_model(model="gpt-4o-mini", openai_api_key = OPEN_AI_KEY)
    return {"messages": model.invoke(state["messages"])}

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

def judge_response(state: dict) -> dict | None:
    print()
    print("ℹ️ AI Message from LLM: ")
    print(state["messages"][-1].content)

    print()
    print("ℹ️ Entire State Object: ")
    print(state)

    evaluator = create_llm_as_judge(
        prompt=EVALUATION_PROMPT,
        model="openai:o3-mini",
        feedback_key="pass",
    )

    print()
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)
    print("ℹ️ Evaluation Result:", eval_result)

    if eval_result["score"]:
        print("✅ Response approved by judge")
        print("Rationale: ", eval_result.get('comment'))
        print("-------------------------")
        return None
    else:
        # Otherwise, return the judge's critique as a new user message
        print("⚠️ Judge requested improvements")
        print("Rationale: ", eval_result.get('comment'))
        print("-------------------------")

        return {
            "messages": [{
                "role": "user",
                "content": eval_result["comment"]
            }]
        }

assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)

judge_graph = (
    StateGraph(MessagesState)
    .add_node(judge_response)
    .add_edge(START, "judge_response")
    .add_edge("judge_response", END)
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
