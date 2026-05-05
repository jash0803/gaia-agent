# GAIA Multi-Agent Evaluation System

A multi-agent system built with **LangGraph** and **LangChain** to tackle the [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard) — a set of real-world questions that test AI assistants on reasoning, tool use, and multimodal understanding.

## How It Works

A **supervisor agent** analyzes each incoming question and delegates it to one of four specialized sub-agents:

| Agent | Responsibility | Tools |
|---|---|---|
| **Web Research** | Factual lookups, current events, YouTube video analysis | Tavily Search, Wikipedia, Gemini 2.5 Pro Video |
| **Code Execution** | Python programming, algorithms, data processing | Python REPL |
| **File Processing** | Excel, CSV, PDF, audio, image analysis | GAIA File Downloader, Pandas, Whisper, GPT-5-mini Vision |
| **Math/Reasoning** | Arithmetic, algebra, calculus, statistics | Calculator, Python REPL |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams and data flow.

## Project Structure

```
├── app.py                  # Gradio UI + submission logic
├── agent.py                # GAIAAgent class (supervisor wrapper)
├── prompts.py              # Shared GAIA answer format prompt
├── agents/
│   ├── supervisor.py       # LangGraph supervisor graph
│   ├── web_research.py     # Web search + video agent
│   ├── code_agent.py       # Code execution agent
│   ├── file_agent.py       # File processing agent
│   └── math_agent.py       # Math/reasoning agent
├── tools/
│   ├── search_tools.py     # Tavily + Wikipedia
│   ├── video_tools.py      # Gemini YouTube video analysis
│   ├── code_tools.py       # Python REPL
│   ├── file_tools.py       # File download, Excel, audio, image, PDF
│   └── math_tools.py       # Calculator + Python REPL
├── requirements.txt
└── test_agent.py           # Local testing script
```

## Setup

### Environment Variables

Set these in a local `.env` file:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT-5-mini for reasoning, vision, and Whisper transcription |
| `TAVILY_API_KEY` | Web search via Tavily |
| `GOOGLE_API_KEY` | Gemini 2.5 Pro for YouTube video analysis |
| `HF_TOKEN` | HuggingFace token for downloading GAIA dataset files |

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_agent.py      # test on a random GAIA question
python app.py             # launch Gradio UI
```

## Scoring

The GAIA benchmark uses **exact match** scoring. The agent uses the official GAIA answer format prompt — reasoning through each question before producing a concise `FINAL ANSWER` (a number, a few words, or a comma-separated list) with no articles, abbreviations, or units unless specified.
