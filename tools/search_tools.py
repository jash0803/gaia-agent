import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic.
    Args:
        query: The search query for Wikipedia.
    Returns a summary of the most relevant Wikipedia article(s).
    """
    from langchain_community.utilities import WikipediaAPIWrapper

    try:
        wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
        return wrapper.run(query)
    except Exception as e:
        return f"Wikipedia search error: {e}. Try using the Tavily web search instead."


def get_search_tools():
    tavily = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
    )

    return [tavily, search_wikipedia]
