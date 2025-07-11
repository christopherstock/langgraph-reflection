# read config
import configparser

from langchain_community.llms.openai import OpenAI
from langchain_community.output_parsers import rail_parser
from langchain_core.prompts import PromptTemplate

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

# Reflection demonstration starts here
# Step 1: Instantiate LLM
llm = OpenAI(model="gpt-4", api_key=os.environ["OPENAI_API_KEY"])

# Step 2: Create a simple reflection prompt template
prompt = PromptTemplate(template="Reflect on the following text: '{input_text}'")

# Step 3: Define input and render prompt
input_text = "The code simplifies complex tasks by abstracting functionality."
rendered_prompt = prompt.format(input_text=input_text)

# Step 4: Send request to LLM
response = llm.generate([rendered_prompt])

# Step 5: Parse response
parser = rail_parser.GuardrailsOutputParser.from_rail_string(
    rail_str="<rail><output format='text'></output></rail>", api=llm
)
parsed_output = parser.parse(response.responses[0].text)

# Print results
print(f"Prompt: {rendered_prompt}")
print(f"Response: {response.responses[0].text}")
print(f"Parsed Output: {parsed_output}")
