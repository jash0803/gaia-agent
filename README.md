# GAIA Multi-Agent Evaluation System

A multi-agent system built with **LangGraph** and **LangChain** to tackle the [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard) — a set of real-world questions that test AI assistants on reasoning, tool use, and multimodal understanding.

## How It Works

A **supervisor agent** analyzes each incoming question and delegates it to one of four specialized sub-agents:

| Agent | Responsibility | Tools |
|---|---|---|
| **Web Research** | Factual lookups, current events, "who/what/when/where" | Tavily Search, Wikipedia |
| **Code Execution** | Python programming, algorithms, data processing | Python REPL |
| **File Processing** | Excel, CSV, PDF, audio, image analysis | GAIA File Downloader, Pandas, Whisper, GPT-4o Vision |
| **Math/Reasoning** | Arithmetic, algebra, calculus, statistics | Calculator, Python REPL |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams and data flow.

## Project Structure

```
├── app.py                  # Gradio UI + submission logic
├── agent.py                # GAIAAgent class (supervisor wrapper)
├── agents/
│   ├── supervisor.py       # LangGraph supervisor graph
│   ├── web_research.py     # Web search agent
│   ├── code_agent.py       # Code execution agent
│   ├── file_agent.py       # File processing agent
│   └── math_agent.py       # Math/reasoning agent
├── tools/
│   ├── search_tools.py     # Tavily + Wikipedia
│   ├── code_tools.py       # Python REPL
│   ├── file_tools.py       # File download, Excel, audio, image, PDF
│   └── math_tools.py       # Calculator + Python REPL
├── requirements.txt
└── test_agent.py           # Local testing script
```

## Setup

### Environment Variables

Set these as secrets in your HuggingFace Space (or in a local `.env` file):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT-4o for reasoning, vision, and Whisper transcription |
| `TAVILY_API_KEY` | Web search via Tavily |

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_agent.py      # test on a random GAIA question
python app.py             # launch Gradio UI
```

## Usage

1. Open the Gradio interface.
2. Log in with your Hugging Face account.
3. Click **Run Evaluation & Submit All Answers**.
4. The agent processes all 20 GAIA questions and submits answers to the [leaderboard](https://huggingface.co/spaces/agents-course/Students_leaderboard).

## Scoring

The GAIA benchmark uses **exact match** scoring. The agent's system prompts enforce concise answers — a number, a few words, or a comma-separated list — with no articles, abbreviations, or units unless specified.
