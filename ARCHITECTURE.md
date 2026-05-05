# Architecture

## System Overview

```mermaid
graph LR
    User[User via Gradio] --> App[app.py]
    App -->|fetch questions| API[GAIA Scoring API]
    App -->|run per question| Agent[GAIAAgent]
    Agent --> Supervisor[Supervisor Agent]
    Supervisor -->|delegate| WebAgent[Web Research Agent]
    Supervisor -->|delegate| CodeAgent[Code Execution Agent]
    Supervisor -->|delegate| FileAgent[File Processing Agent]
    Supervisor -->|delegate| MathAgent[Math Agent]
    Agent -->|extract answer| App
    App -->|submit answers| API
```

## Supervisor Routing

The supervisor receives each question, classifies it, and routes to the appropriate specialist. It can call multiple agents sequentially for multi-step questions.

```mermaid
flowchart TD
    Q[Incoming Question] --> Analyze[Supervisor Analyzes Question]
    Analyze --> HasFile{Has associated file?}
    HasFile -->|Yes| FileAgent[File Processing Agent]
    HasFile -->|No| Classify{Question Type?}
    Classify -->|Facts / Search| WebAgent[Web Research Agent]
    Classify -->|Code / Algorithm| CodeAgent[Code Execution Agent]
    Classify -->|Math / Calculation| MathAgent[Math Agent]
    FileAgent --> NeedMore{Need further processing?}
    NeedMore -->|Yes| Classify
    NeedMore -->|No| Extract[Extract Concise Answer]
    WebAgent --> Extract
    CodeAgent --> Extract
    MathAgent --> Extract
    Extract --> Return[Return Answer to App]
```

## Agent-Tool Mapping

Each sub-agent is built with `create_react_agent` and has access to specific tools.

```mermaid
graph TD
    subgraph web [Web Research Agent]
        Tavily[Tavily Search]
        Wiki[Wikipedia]
    end

    subgraph code [Code Execution Agent]
        PythonREPL1[Python REPL]
    end

    subgraph file [File Processing Agent]
        Download[GAIA File Downloader]
        Excel[Excel/CSV Reader]
        Audio[Whisper Transcription]
        Vision[GPT-4o Image Analysis]
        TextFile[Text File Reader]
        PDF[PDF Reader]
        PythonREPL2[Python REPL]
    end

    subgraph math [Math Agent]
        Calc[Calculator]
        PythonREPL3[Python REPL]
    end
```

## Data Flow — Single Question

```mermaid
sequenceDiagram
    participant App as app.py
    participant GA as GAIAAgent
    participant SV as Supervisor
    participant SA as Sub-Agent
    participant Tool as Tool

    App->>GA: question + task_id
    GA->>GA: Check if task has file (HEAD /files/{task_id})
    GA->>SV: Invoke graph with messages
    SV->>SV: Analyze question, pick agent
    SV->>SA: Delegate with full context
    SA->>Tool: Call tool (search, code, file, etc.)
    Tool-->>SA: Tool result
    SA-->>SV: Agent response
    SV->>SV: Decide: done or call another agent?
    SV-->>GA: Final response
    GA->>GA: Extract concise answer (strip prefixes)
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
    API-->>App: 20 questions

    loop For each question
        App->>Agent: agent(question, task_id)
        Agent-->>App: concise answer
    end

    App->>API: POST /submit (username, agent_code, answers)
    API-->>App: score, correct_count, total_attempted
    App-->>Gradio: Display results table + score
    Gradio-->>User: Show results
```

## File Processing Pipeline

The file agent handles multiple modalities depending on the file extension:

```mermaid
flowchart LR
    Download[Download File] --> Detect{File Extension?}
    Detect -->|.xlsx .xls .csv| Pandas[Pandas Reader]
    Detect -->|.mp3 .wav .m4a| Whisper[Whisper API]
    Detect -->|.png .jpg .gif .webp| GPT4V[GPT-4o Vision]
    Detect -->|.pdf| PyPDF2[PyPDF2 Reader]
    Detect -->|.txt .py .json .md| TextReader[Text Reader]
    Pandas --> Analyze[Analyze + Answer]
    Whisper --> Analyze
    GPT4V --> Analyze
    PyPDF2 --> Analyze
    TextReader --> Analyze
```
