import os
import re
from langchain_core.tools import tool


@tool
def analyze_youtube_video(youtube_url: str, question: str) -> str:
    """Analyze a YouTube video using Gemini 2.5 Pro's native video understanding.
    Gemini can watch the video directly from the URL — no download needed.
    Args:
        youtube_url: The full YouTube URL (e.g. https://www.youtube.com/watch?v=xyz).
        question: The question to answer about the video content.
    Returns the analysis/answer as text.
    """
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY environment variable is not set."

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                {
                    "parts": [
                        {"text": question},
                        {"file_data": {"file_uri": youtube_url}},
                    ]
                }
            ],
        )
        return response.text
    except Exception as e:
        return f"Error analyzing video: {e}"


def get_video_tools():
    return [analyze_youtube_video]
