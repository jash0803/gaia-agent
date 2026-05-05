from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from agents.web_research import create_web_research_agent
from agents.code_agent import create_code_agent
from agents.file_agent import create_file_agent
from agents.math_agent import create_math_agent
from prompts import GAIA_ANSWER_FORMAT

SUPERVISOR_PROMPT = GAIA_ANSWER_FORMAT + """
You are a supervisor agent coordinating a team of specialized agents to answer
questions from the GAIA benchmark. Your job is to analyze each question, delegate it to the most
appropriate specialist agent, and then relay their FINAL ANSWER back.

YOUR TEAM:
- web_research_agent: For factual lookups, "who/what/when/where" questions, current events,
  anything requiring internet search, Wikipedia lookup, or YouTube video analysis.
  This agent has a tool that can WATCH YouTube videos using Gemini vision.
- code_agent: For Python programming tasks, algorithm problems, code execution,
  and computational problems best solved by writing code.
- file_agent: For questions with attached files ONLY. This agent downloads and processes
  files (Excel, CSV, PDF, audio, images, text files) from the GAIA dataset.
- math_agent: For pure mathematical problems — arithmetic, algebra, calculus, statistics,
  probability, and number theory.

ROUTING RULES (follow strictly):
1. Route to file_agent ONLY when the question contains the marker text
   "[IMPORTANT CONTEXT: This question has an associated file". If this marker is NOT present,
   do NOT use file_agent — even if the question mentions "attached", "file", "image", or
   "audio" in the question text itself. The marker is the only reliable signal.
2. If the question contains a YouTube URL (youtube.com or youtu.be), route to web_research_agent.
   It has the analyze_youtube_video tool that can watch and understand video content.
3. For questions about facts, people, places, events, or anything requiring web lookup,
   use web_research_agent.
4. For math problems or numerical computation, use math_agent.
5. For code/programming questions (without the file marker), use code_agent.
6. You may call multiple agents sequentially if needed (e.g., download a file then compute).

When relaying the agent's answer, use the FINAL ANSWER template above.
NEVER respond with internal routing like "transfer_to_file_agent" — that is NOT an answer.
NEVER say "I couldn't find", "Please provide", or "I'm unable to" — always give your best answer.
"""


def build_supervisor_graph(model: ChatOpenAI | None = None):
    if model is None:
        model = ChatOpenAI(model="gpt-4o", temperature=0)

    web_agent = create_web_research_agent(model)
    code_agent = create_code_agent(model)
    file_agent = create_file_agent(model)
    math_agent = create_math_agent(model)

    supervisor = create_supervisor(
        agents=[web_agent, code_agent, file_agent, math_agent],
        model=model,
        prompt=SUPERVISOR_PROMPT,
    )

    graph = supervisor.compile()
    return graph
