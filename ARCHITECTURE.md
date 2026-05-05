# Architecture

## System Overview

```mermaid
graph LR
    User[User via Gradio] --> App[app.py]
    App -->|fetch questions| API[GAIA Scoring API]
    App -->|"run per question (+ file_name)"| Agent[GAIAAgent]
    Agent --> Supervisor["Supervisor (GPT-5-mini)"]
    Supervisor -->|delegate| WebAgent[Web Research Agent]
    Supervisor -->|delegate| CodeAgent[Code Execution Agent]
    Supervisor -->|delegate| FileAgent[File Processing Agent]
    Supervisor -->|delegate| MathAgent[Math Agent]
    Agent -->|"extract FINAL ANSWER"| App
    App -->|submit answers| API
    App -->|save| Log[submission_payload.log]
```

## Supervisor Routing

The supervisor receives each question and routes based on strict rules. The `[IMPORTANT CONTEXT: ...]` marker in the prompt is the only signal for file-based routing.

```mermaid
flowchart TD
    Q[Incoming Question] --> Analyze[Supervisor Analyzes Question]
    Analyze --> HasMarker{"Has [IMPORTANT CONTEXT: file...] marker?"}
    HasMarker -->|Yes| FileAgent[File Processing Agent]
    HasMarker -->|No| HasYT{Contains YouTube URL?}
    HasYT -->|Yes| WebAgent[Web Research Agent]
    HasYT -->|No| Classify{Question Type?}
    Classify -->|Facts / Search| WebAgent
    Classify -->|Code / Algorithm| CodeAgent[Code Execution Agent]
    Classify -->|Math / Calculation| MathAgent[Math Agent]
    FileAgent --> NeedMore{Need further processing?}
    NeedMore -->|Yes| Classify
    NeedMore -->|No| Extract["Extract FINAL ANSWER"]
    WebAgent --> Extract
    CodeAgent --> Extract
    MathAgent --> Extract
    Extract --> Return[Return to App]
```

## Agent-Tool Mapping

Each sub-agent is built with `create_agent` and has access to specific tools.

```mermaid
graph TD
    subgraph web ["Web Research Agent (GPT-5-mini)"]
        Tavily[Tavily Search]
        Wiki[Wikipedia]
        Gemini["Gemini 2.5 Pro Video"]
    end

    subgraph code ["Code Execution Agent (GPT-5-mini)"]
        PythonREPL1[Python REPL]
    end

    subgraph file ["File Processing Agent (GPT-5-mini)"]
        Download["GAIA File Downloader (HF Dataset)"]
        Excel[Excel/CSV Reader]
        Audio[Whisper Transcription]
        Vision["GPT-5-mini Vision (Responses API)"]
        TextFile[Text File Reader]
        PDF[PDF Reader]
        PythonREPL2[Python REPL]
    end

    subgraph math ["Math Agent (GPT-5-mini)"]
        Calc[Calculator]
        PythonREPL3[Python REPL]
    end
```

## Answer Flow

All agents use the GAIA answer format prompt: reason through the problem, then output `FINAL ANSWER: [answer]`. The extraction layer strips the prefix before submission.

```mermaid
flowchart LR
    Prompt["GAIA Answer Format Prompt"] --> Agent["Sub-Agent Reasons"]
    Agent --> FA["FINAL ANSWER: 42"]
    FA --> Extract["_extract_answer()"]
    Extract --> Clean["42"]
    Clean --> Submit["POST /submit"]
```

## Data Flow — Single Question

```mermaid
sequenceDiagram
    participant App as app.py
    participant GA as GAIAAgent
    participant SV as Supervisor
    participant SA as Sub-Agent
    participant Tool as Tool

    App->>GA: question + task_id + file_name
    GA->>GA: Build prompt with file context if file_name present
    GA->>SV: Invoke graph with messages
    SV->>SV: Analyze question, pick agent
    SV->>SA: Delegate with full context
    SA->>Tool: Call tool (search, video, code, file, etc.)
    Tool-->>SA: Tool result
    SA->>SA: Reason and produce FINAL ANSWER
    SA-->>SV: Response with FINAL ANSWER
    SV-->>GA: Relay FINAL ANSWER
    GA->>GA: Extract answer via regex
    GA-->>App: Clean answer string
```

## Submission Flow — Full Evaluation

```mermaid
sequenceDiagram
    participant User
    participant Gradio as Gradio UI
    participant App as app.py
    participant API as GAIA Scoring API
    participant Agent as GAIAAgent

    User->>Gradio: Click "Run Evaluation"
    Gradio->>App: run_and_submit_all(profile)
    App->>API: GET /questions
    API-->>App: 20 questions with file_name metadata

    loop For each question
        App->>Agent: agent(question, task_id, file_name)
        Agent-->>App: concise answer
    end

    App->>App: Save submission_payload.log
    App->>API: POST /submit (username, agent_code, answers)
    API-->>App: score, correct_count, total_attempted
    App-->>Gradio: Display results table + score
    Gradio-->>User: Show results
```

## File Processing Pipeline

The file agent downloads from the HuggingFace GAIA dataset (with API fallback) and handles multiple modalities by extension:

```mermaid
flowchart LR
    Download["Download from HF Dataset"] --> Detect{File Extension?}
    Detect -->|.xlsx .xls .csv| Pandas[Pandas Reader]
    Detect -->|.mp3 .wav .m4a| Whisper[Whisper API]
    Detect -->|.png .jpg .gif .webp| GPT5V["GPT-5-mini Vision"]
    Detect -->|.pdf| PyPDF2[PyPDF2 Reader]
    Detect -->|.txt .py .json .md| TextReader[Text Reader]
    Pandas --> Analyze["Reason + FINAL ANSWER"]
    Whisper --> Analyze
    GPT5V --> Analyze
    PyPDF2 --> Analyze
    TextReader --> Analyze
```

## Video Analysis Pipeline

YouTube video questions are handled by the web research agent using Gemini's native video understanding -- no download required:

```mermaid
flowchart LR
    Question["Question with YouTube URL"] --> WebAgent[Web Research Agent]
    WebAgent --> GeminiTool["analyze_youtube_video tool"]
    GeminiTool -->|"pass URL directly"| Gemini["Gemini 2.5 Pro"]
    Gemini -->|watches video| Response[Video Analysis Result]
    Response --> Answer["Reason + FINAL ANSWER"]
```
