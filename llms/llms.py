# LLM classes
from llms.claude import *
from llms.command import *
from llms.gpt import *
from llms.llama import *

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
META_API_KEY = os.getenv("META_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in the environment variables.")

if not ANTHROPIC_API_KEY:
    raise ValueError("Anthropic API key not found in the environment variables.")


# Mapping of form responses to LLM classes
MODELS = {"Anthropic Claude 3.5 Sonnet": Claude(ANTHROPIC_API_KEY, version="claude-3-5-sonnet-20240620"),
          "Cohere Command R": Command(COHERE_API_KEY, version="command-r-08-2024"),
          "Cohere Command R+": Command(COHERE_API_KEY, version="command-r-plus-08-2024"),
          "OpenAI GPT-4o": GPT(OPENAI_API_KEY, version="gpt-4o"),
          "OpenAI GPT-4o mini": GPT(OPENAI_API_KEY, version="gpt-4o-mini"),
          "Meta LLaMA 3.1 8B": Llama(META_API_KEY, version="llama3.1-8b"),
          "Meta LLaMA 3.1 70B": Llama(META_API_KEY, version="llama3.1-70b"),
          "Meta LLaMA 3.1 405B": Llama(META_API_KEY, version="llama3.1-405b"),
          }
