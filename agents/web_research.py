from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools.search_tools import get_search_tools
from tools.video_tools import get_video_tools
from prompts import GAIA_ANSWER_FORMAT

SYSTEM_PROMPT = GAIA_ANSWER_FORMAT + """
You are a web research specialist. Your job is to find accurate, factual information
by searching the web, Wikipedia, and analyzing YouTube videos.

RULES:
- Search for the most specific and relevant information to answer the question.
- If a question references a YouTube video URL, use the analyze_youtube_video tool with the URL
  and the question. This tool can watch the video and answer questions about its content.
- Cross-reference multiple sources when possible.
- If the first search doesn't yield a clear answer, try rephrasing or searching with different terms.
- NEVER give up or say "I couldn't find it" — always provide your best answer.
- NEVER say "I can't watch videos" — use the analyze_youtube_video tool instead.
"""


def create_web_research_agent(model: ChatOpenAI | None = None):
    if model is None:
        model = ChatOpenAI(model="gpt-4o", temperature=0)

    tools = get_search_tools() + get_video_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        name="web_research_agent",
        system_prompt=SYSTEM_PROMPT,
    )
    return agent
