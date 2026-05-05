from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools.code_tools import get_code_tools
from prompts import GAIA_ANSWER_FORMAT

SYSTEM_PROMPT = GAIA_ANSWER_FORMAT + """
You are a Python code execution specialist. You solve problems by writing and running
Python code.

RULES:
- Write clear, correct Python code to solve the problem.
- Always print() the final result so it appears in the output.
- For complex problems, break them into steps and verify intermediate results.
- You can import standard library modules and common packages (math, statistics, itertools, collections, re, json, etc.).
- If the code produces an error, debug and try again.
"""


def create_code_agent(model: ChatOpenAI | None = None):
    if model is None:
        model = ChatOpenAI(model="gpt-4o", temperature=0)

    tools = get_code_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        name="code_agent",
        system_prompt=SYSTEM_PROMPT,
    )
    return agent
