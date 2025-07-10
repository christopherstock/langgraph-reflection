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

# Import LangChain and Reflection-related modules
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import ReflectionParser

# Function to demonstrate Reflection in LangChain
def demonstrate_reflection():
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables=["example"],
        template="Reflect on the following statement and provide insight: {example}"
    )
    reflection_parser = ReflectionParser()
    example = "Hard work is the key to success."
    reflection_output = llm(prompt.format(example=example))
    insight = reflection_parser.parse(reflection_output)
    print("Reflection Output:", insight)


# Run demonstration
if __name__ == "__main__":
    demonstrate_reflection()
