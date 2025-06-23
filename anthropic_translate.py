COLOR_OK = '\033[92m' # light green
COLOR_DEFAULT = '\033[0m'

import os

# read config
print("importing config/config.ini ", end='')
import configparser
config = configparser.ConfigParser()
config.read("config/config.ini")
ANTHROPIC_API_KEY = config.get("AnthropicAPI", "ANTHROPIC_API_KEY")
print(COLOR_OK + "OK" + COLOR_DEFAULT)

print("set OpenAI key ", end='')
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
print(COLOR_OK + "OK" + COLOR_DEFAULT)

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    # other params...
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    (
        "human",
        "I love programming."
    ),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
