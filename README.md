# 🤖 Enhanced Multi-Modal GAIA Agent

A sophisticated AI agent system built with LangGraph and LangChain for the GAIA (General AI Assistant) evaluation benchmark. This agent processes multi-modal questions including text, images, audio, Excel files, and performs web searches with intelligent tool routing.

## 🌟 Features

### 🚀 Core Capabilities
- **Multi-Modal Processing**: Handles text, images, audio, Excel files, and documents
- **Intelligent Tool Routing**: Automatic question classification and appropriate tool selection
- **Async Processing**: Concurrent question processing for improved performance
- **File Download & Processing**: Automatic file retrieval via `/files/{task_id}` endpoint
- **Web Search Integration**: Real-time information retrieval using Tavily Search API
- **Python Code Execution**: Mathematical calculations and data analysis via REPL

### 🔧 Technical Features
- **LangGraph Workflow**: State-based processing with retry mechanisms
- **Comprehensive Error Handling**: Robust error recovery and logging
- **Caching System**: LRU cache for web search results (100 entries)
- **Answer Extraction**: Clean answer formatting for evaluation compliance
- **Enhanced Logging**: Detailed execution tracking and debugging

### 📁 Supported File Types
- 🖼️ **Images** (JPG, PNG, GIF): OCR and image captioning
- 🎵 **Audio** (MP3, WAV): Speech-to-text transcription
- 📊 **Excel** (XLSX, XLS, CSV): Data extraction and analysis
- 📄 **Text** (TXT, MD, JSON): Direct content reading
- 🔍 **Auto-Detection**: Automatic file type identification

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for GPT-4)
- Tavily Search API key
- Hugging Face account (for deployment)

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
SPACE_ID=your_huggingface_space_id  # Optional, for HF Spaces deployment
SPACE_HOST=your_space_host          # Optional, for HF Spaces deployment
```

## 📖 Usage

### Web Interface
1. **Login**: Use the Hugging Face login button
2. **Run Evaluation**: Click "Run Enhanced Evaluation & Submit All Answers"
3. **Monitor Progress**: View real-time status and detailed results
4. **Review Results**: Check the results table for performance metrics

### Question Processing Flow
```
Question Input → File Download → Question Classification → Tool Selection → Processing → Answer Extraction → Submission
```

### Supported Question Types
- **Mathematics**: Calculations, equations, statistical analysis
- **Web Search**: Current events, factual information, research
- **Programming**: Code execution, algorithm implementation
- **Image Analysis**: OCR, image description, visual content analysis
- **Audio Processing**: Speech transcription, audio analysis
- **Data Analysis**: Excel file processing, CSV analysis

---

**Made with ❤️ for the GAIA evaluation benchmark**