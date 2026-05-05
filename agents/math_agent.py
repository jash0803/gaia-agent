from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools.math_tools import get_math_tools
from prompts import GAIA_ANSWER_FORMAT

SYSTEM_PROMPT = GAIA_ANSWER_FORMAT + """
You are a mathematics and computation specialist. You solve mathematical problems
with precision and accuracy.

CAPABILITIES:
- Arithmetic, algebra, and equation solving
- Statistics and probability
- Calculus (derivatives, integrals)
- Number theory and combinatorics
- Use the calculator tool for simple expressions
- Use Python code for complex computations

RULES:
- Always verify your calculations by computing them, don't just reason about them.
- Use the calculator or Python REPL to compute results — never do mental math for anything non-trivial.
- Give exact numbers, not approximations, unless asked otherwise.
- For decimal answers, provide reasonable precision.
"""


def create_math_agent(model: ChatOpenAI | None = None):
    if model is None:
        model = ChatOpenAI(model="gpt-4o", temperature=0)

    tools = get_math_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        name="math_agent",
        system_prompt=SYSTEM_PROMPT,
    )
    return agent
