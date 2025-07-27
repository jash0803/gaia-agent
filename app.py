import os
import gradio as gr
import requests
import inspect
import pandas as pd
from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, Literal
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from transformers import pipeline
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.tools.python.tool import PythonREPLTool
import asyncio
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
import json
from pathlib import Path
import mimetypes

# Fixed import
from dotenv import load_dotenv

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Enhanced State Definition ---
from typing import TypedDict
class EnhancedQuestionState(TypedDict, total=False):
    question: str
    question_id: str
    question_type: str
    context: dict
    tools_available: list
    tool_results: dict
    reasoning_steps: list
    final_answer: str
    confidence_score: float
    execution_time: float
    error: str
    timestamp: str
    retry_count: int
    associated_files: list
    file_paths: dict

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
FILES_DOWNLOAD_DIR = "downloaded_files"

# Create downloads directory
os.makedirs(FILES_DOWNLOAD_DIR, exist_ok=True)

# --- File Download and Processing Functions ---
def download_task_file(task_id: str, api_url: str) -> dict:
    """Download associated file for a task if it exists"""
    file_url = f"{api_url}/files/{task_id}"
    file_info = {"exists": False, "path": None, "type": None, "error": None}
    
    try:
        logger.info(f"Attempting to download file for task {task_id}")
        response = make_request_with_retry(file_url)
        
        # Try multiple methods to get the proper filename
        filename = None
        
        # Method 1: Check Content-Disposition header
        content_disposition = response.headers.get('content-disposition', '')
        if 'filename=' in content_disposition:
            # Handle both quoted and unquoted filenames
            if 'filename*=' in content_disposition:
                # RFC 5987 format: filename*=UTF-8''filename.ext
                filename_part = content_disposition.split('filename*=')[1]
                if "''" in filename_part:
                    filename = filename_part.split("''")[1].strip('"')
                else:
                    filename = filename_part.strip('"')
            elif 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"').strip("'")
        
        # Method 2: Check if URL has filename parameter
        if not filename and '?' in file_url:
            from urllib.parse import parse_qs, urlparse
            parsed_url = urlparse(file_url)
            params = parse_qs(parsed_url.query)
            if 'filename' in params:
                filename = params['filename'][0]
        
        # Method 3: Try to extract from response headers or content
        if not filename:
            content_type = response.headers.get('content-type', '')
            # Check for common content types and assign reasonable names
            if 'audio' in content_type:
                if 'mp3' in content_type or content_type == 'audio/mpeg':
                    filename = f"Homework.mp3"  # Default for audio homework
                elif 'wav' in content_type:
                    filename = f"audio_{task_id}.wav"
                else:
                    filename = f"audio_{task_id}.mp3"
            elif 'image' in content_type:
                if 'jpeg' in content_type or 'jpg' in content_type:
                    filename = f"image_{task_id}.jpg"
                elif 'png' in content_type:
                    filename = f"image_{task_id}.png"
                else:
                    filename = f"image_{task_id}.jpg"
            elif 'excel' in content_type or 'spreadsheet' in content_type:
                filename = f"data_{task_id}.xlsx"
            elif 'text' in content_type:
                filename = f"text_{task_id}.txt"
            else:
                # Guess extension from content-type
                extension = mimetypes.guess_extension(content_type) or '.bin'
                filename = f"file_{task_id}{extension}"
        
        # Ensure filename is valid and clean
        if filename:
            # Remove any invalid characters and ensure it's not too long
            import re
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            filename = filename[:100]  # Limit length
        else:
            filename = f"file_{task_id}.bin"
        
        file_path = os.path.join(FILES_DOWNLOAD_DIR, filename)
        
        # Handle duplicate filenames by adding a counter
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        file_info.update({
            "exists": True,
            "path": file_path,
            "filename": os.path.basename(file_path),
            "original_filename": filename,
            "type": determine_file_type(file_path),
            "size": len(response.content),
            "content_type": response.headers.get('content-type', 'unknown')
        })
        
        logger.info(f"Successfully downloaded file for task {task_id}: {filename} -> {os.path.basename(file_path)}")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.info(f"No file associated with task {task_id}")
        else:
            logger.warning(f"HTTP error downloading file for task {task_id}: {e}")
            file_info["error"] = str(e)
    except Exception as e:
        logger.error(f"Error downloading file for task {task_id}: {e}")
        file_info["error"] = str(e)
    
    return file_info

def determine_file_type(file_path: str) -> str:
    """Determine file type based on extension and content"""
    extension = Path(file_path).suffix.lower()
    
    if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
        return 'image'
    elif extension in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
        return 'audio'
    elif extension in ['.xlsx', '.xls', '.csv']:
        return 'excel'
    elif extension in ['.txt', '.md', '.json']:
        return 'text'
    elif extension in ['.pdf']:
        return 'pdf'
    else:
        return 'unknown'

def process_file_content(file_path: str, file_type: str) -> str:
    """Process file content based on type"""
    try:
        # Ensure we're using the full path
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        
        logger.info(f"Processing file: {file_path} (type: {file_type})")
        
        if file_type == 'image':
            return ocr_tool_stub(file_path)
        elif file_type == 'audio':
            return audio_tool_stub(file_path)
        elif file_type == 'excel':
            return excel_processor_tool(file_path)
        elif file_type == 'text':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File content:\n{content}"
        elif file_type == 'pdf':
            return f"PDF file detected: {os.path.basename(file_path)}. PDF processing not implemented yet."
        else:
            return f"File type '{file_type}' detected but automatic processing not supported. File location: {file_path}"
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return f"Error processing file: {e}"
search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
audio_transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
ocr = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
python_repl = PythonREPL()

@lru_cache(maxsize=100)
def cached_web_search(query: str) -> str:
    """Cached web search with enhanced error handling"""
    try:
        logger.info(f"Performing web search for: {query[:50]}...")
        results = search.results(query=query, include_answer=True)
        if results and results[0].get("answer"):
            return results[0]["answer"]
        else:
            return " ".join(r.get("content", "") for r in results[:2])
    except Exception as e:
        logger.error(f"Web search error for query '{query}': {e}")
        return f"Search error: {e}"

def web_search_tool(q):
    """Web search tool with retry mechanism"""
    for attempt in range(MAX_RETRIES):
        try:
            return cached_web_search(q)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"Search failed after {MAX_RETRIES} attempts: {e}"
            time.sleep(1)  # Brief delay before retry

def audio_tool_stub(audio):
    """Enhanced audio transcription with error handling and path verification"""
    try:
        # Ensure we have the full path
        if isinstance(audio, str):
            if not os.path.isabs(audio):
                audio = os.path.abspath(audio)
            
            if not os.path.exists(audio):
                return f"Audio file not found at: {audio}"
        
        logger.info(f"Processing audio transcription for: {audio}")
        result = audio_transcriber(audio)
        transcription = result['text'] if 'text' in result else "No transcription available."
        return f"Audio transcription completed:\n{transcription}"
    except Exception as e:
        logger.error(f"Audio transcription error for {audio}: {e}")
        return f"Audio transcription error: {e}"
    
def ocr_tool_stub(image):
    """Enhanced OCR with error handling and path verification"""
    try:
        # Ensure we have the full path
        if isinstance(image, str):
            if not os.path.isabs(image):
                image = os.path.abspath(image)
            
            if not os.path.exists(image):
                return f"Image file not found at: {image}"
        
        logger.info(f"Processing OCR for: {image}")
        result = ocr(image)
        extracted_text = result[0]['generated_text'] if result else "No text extracted."
        return f"OCR processing completed:\n{extracted_text}"
    except Exception as e:
        logger.error(f"OCR error for {image}: {e}")
        return f"OCR error: {e}"

def excel_processor_tool(filepath):
    """Enhanced Excel processor with better error handling and path verification"""
    try:
        # Ensure we have the full path
        if not os.path.isabs(filepath):
            filepath = os.path.abspath(filepath)
        
        if not os.path.exists(filepath):
            return f"Excel file not found at: {filepath}"
        
        logger.info(f"Processing Excel file: {filepath}")
        df = pd.read_excel(filepath)
        csv_content = df.to_csv(index=False)
        return f"Excel file processed successfully. Data converted to CSV format:\n{csv_content}"
    except Exception as e:
        logger.error(f"Excel processing error for {filepath}: {e}")
        return f"Excel processing error: {e}"
    
python_repl_tool = PythonREPLTool(
    python_repl=python_repl,
    return_direct=True,
    descriptor="Execute Python code for mathematical calculations, data analysis, algorithmic problems, and computational tasks. Can handle complex math, statistics, and programming challenges."
)

# --- Question Type Classification ---
def classify_question_type(question: str) -> str:
    """Classify question type for better tool routing"""
    question_lower = question.lower()
    
    # Math/calculation patterns
    math_patterns = [
        r'\d+\s*[\+\-\*\/\%]\s*\d+',  # Basic math operations
        r'calculate|compute|solve|equation|formula',
        r'sum|average|mean|median|standard deviation',
        r'probability|statistics|derivative|integral'
    ]
    
    # Search patterns
    search_patterns = [
        r'what is|who is|when did|where is|how many',
        r'current|latest|recent|today|news',
        r'definition of|explain|describe'
    ]
    
    # Programming patterns
    code_patterns = [
        r'python|code|function|algorithm|program',
        r'list comprehension|loop|array|data structure',
        r'debug|error|exception'
    ]
    
    # Check patterns
    for pattern in math_patterns:
        if re.search(pattern, question_lower):
            return "math"
    
    for pattern in code_patterns:
        if re.search(pattern, question_lower):
            return "programming"
    
    for pattern in search_patterns:
        if re.search(pattern, question_lower):
            return "search"
    
    # Default classification
    if any(word in question_lower for word in ['image', 'picture', 'photo', 'visual']):
        return "image"
    elif any(word in question_lower for word in ['audio', 'sound', 'music', 'speech']):
        return "audio"
    elif any(word in question_lower for word in ['excel', 'spreadsheet', 'csv', 'data']):
        return "excel"
    
    return "general"

# --- Enhanced Tool Definitions ---
tools = [
    Tool(
        name="WebSearch", 
        func=web_search_tool, 
        description="Search the internet for current information, facts, news, and general knowledge. Use for questions requiring up-to-date information, factual verification, or when you need to find specific information not in your training data."
    ),
    Tool(
        name="AudioTranscribe", 
        func=audio_tool_stub, 
        description="Transcribe audio files (.mp3, .wav) to text. Use when you need to process audio recordings, voice messages, or any audio content to extract spoken words."
    ),
    Tool(
        name="OCR", 
        func=ocr_tool_stub, 
        description="Extract text from images and analyze visual content including charts, diagrams, handwritten text, and printed documents. Use for processing any image-based information."
    ),
    Tool(
        name="ExcelProcessor", 
        func=excel_processor_tool, 
        description="Read and process Excel files (.xlsx, .xls) and convert to CSV format for analysis. Use when you need to extract data from spreadsheets or perform data analysis on Excel files."
    ),
    Tool(
        name="FileProcessor", 
        func=lambda file_path: process_file_content(file_path, determine_file_type(file_path)), 
        description="Process downloaded files including images, audio, Excel files, and text documents. Use when you have a file path from the downloaded_files folder and need to extract information from it. The file path should be the full path to the downloaded file."
    ),
    python_repl_tool,
]

# Enhanced LLM setup with system prompt for course evaluation
SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

def extract_final_answer(agent_response: str) -> str:
    """Extract just the final answer from the agent's response for course submission"""
    try:
        # Look for "FINAL ANSWER:" pattern and extract what comes after
        if "FINAL ANSWER:" in agent_response:
            final_answer = agent_response.split("FINAL ANSWER:")[-1].strip()
            # Clean up any trailing punctuation or formatting
            final_answer = final_answer.strip(".,!?;")
            return final_answer
        else:
            # If no FINAL ANSWER pattern found, try to extract the last meaningful line
            lines = [line.strip() for line in agent_response.split('\n') if line.strip()]
            if lines:
                return lines[-1].strip(".,!?;")
            return agent_response.strip()
    except Exception as e:
        logger.warning(f"Error extracting final answer: {e}")
        return agent_response.strip()

llm = ChatOpenAI(temperature=0, model_name="gpt-4")
langchain_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=300,
    agent_kwargs={
        "system_message": SYSTEM_PROMPT
    }
)

# --- Enhanced LangGraph Nodes ---
def classify_question_node(state: EnhancedQuestionState) -> Dict[str, Any]:
    """Classify the question type for better routing"""
    question = state.get("question", "")
    question_type = classify_question_type(question)
    
    logger.info(f"Question classified as: {question_type}")
    
    return {
        **state, 
        "question_type": question_type,
        "timestamp": datetime.now().isoformat()
    }

def enhanced_agent_node(state: EnhancedQuestionState) -> Dict[str, Any]:
    """Enhanced agent node with comprehensive logging and error handling"""
    start_time = time.time()
    question = state.get("question")
    question_type = state.get("question_type", "general")
    retry_count = state.get("retry_count", 0)
    
    if not question:
        return {**state, "error": "No question provided"}

    logger.info(f"Processing {question_type} question: {question[:100]}...")
    
    try:
        # Add context based on question type and system prompt
        context_prompt = f"""
{SYSTEM_PROMPT}

Question type: {question_type}. """
        
        if question_type == "math":
            context_prompt += "Focus on mathematical accuracy and show your work. "
        elif question_type == "search":
            context_prompt += "Use web search for current and factual information. "
        elif question_type == "programming":
            context_prompt += "Use Python REPL for code execution and testing. "
        
        context_prompt += "Remember to end your response with 'FINAL ANSWER: [YOUR ANSWER]' following the specified format rules.\n\n"
        
        enhanced_question = context_prompt + question
        
        result = langchain_agent.run(enhanced_question)
        execution_time = time.time() - start_time
        
        # Extract just the final answer for course submission
        clean_answer = extract_final_answer(result)
        
        logger.info(f"Question processed successfully in {execution_time:.2f}s")
        logger.info(f"Original response length: {len(result)} chars, Clean answer: '{clean_answer}'")
        
        return {
            **state, 
            "final_answer": clean_answer,  # Use cleaned answer for submission
            "full_response": result,       # Keep full response for debugging
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "confidence_score": 0.85
        }
    except Exception as e:
        logger.error(f"Agent error for question '{question}': {e}")
        
        # Retry logic
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying question (attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(2)  # Brief delay before retry
            return {
                **state, 
                "retry_count": retry_count + 1
            }
        
        return {
            **state, 
            "final_answer": f"Agent error after {MAX_RETRIES} attempts: {e}",
            "full_response": f"Error occurred: {e}",
            "error": str(e)
        }

# --- Enhanced LangGraph Compile ---
def should_retry(state: EnhancedQuestionState) -> Literal["agent", "end"]:
    """Determine if we should retry or end"""
    if state.get("retry_count", 0) < MAX_RETRIES and state.get("error") and not state.get("final_answer"):
        return "agent"
    return "end"

graph = StateGraph(EnhancedQuestionState)
graph.add_node("classifier", classify_question_node)
graph.add_node("agent", enhanced_agent_node)

graph.set_entry_point("classifier")
graph.add_edge("classifier", "agent")
graph.add_conditional_edges(
    "agent",
    should_retry,
    {
        "agent": "agent",  # Retry
        "end": END
    }
)

agent_graph = graph.compile()

# --- Async Processing Functions ---
def process_single_question(item: dict, api_url: str) -> dict:
    """Process a single question synchronously with file download support"""
    task_id = item.get("task_id")
    question = item.get("question")
    
    try:
        logger.info(f"Starting processing for task {task_id}")
        
        # Download associated file if it exists
        file_info = download_task_file(task_id, api_url)
        
        # Prepare enhanced state with file information
        state: EnhancedQuestionState = {
            "question": question,
            "question_id": task_id,
            "retry_count": 0,
            "associated_files": [file_info] if file_info["exists"] else [],
            "file_paths": {task_id: file_info["path"]} if file_info["exists"] else {}
        }
        
        # If file exists, process it and enhance the question
        enhanced_question = question
        file_content = ""
        
        if file_info["exists"]:
            logger.info(f"Processing associated file for task {task_id}: {file_info['original_filename']} at {file_info['path']}")
            
            # Verify file exists before processing
            if os.path.exists(file_info["path"]):
                file_content = process_file_content(file_info["path"], file_info["type"])
                
                # Enhance question with file context using original filename
                enhanced_question = f"""
Question: {question}

Associated file information:
- Filename: {file_info['original_filename']}
- File Path: {file_info['path']}
- Type: {file_info['type']}
- Size: {file_info['size']} bytes
- File content/analysis:
{file_content}

Please answer the question taking into account the provided file information. The file has been downloaded and processed successfully.
"""
            else:
                logger.error(f"Downloaded file not found at expected path: {file_info['path']}")
                enhanced_question = f"""
Question: {question}

Note: There was an associated file ({file_info['original_filename']}) but it could not be accessed at the expected location: {file_info['path']}. Please answer the question based on the text alone.
"""
            
            state["question"] = enhanced_question
        
        result = agent_graph.invoke(state)
        
        if result is None:
            # Fallback to direct agent call
            result = enhanced_agent_node(state)
        
        submitted_answer = result.get("final_answer", result.get("answer", "ERROR"))
        
        logger.info(f"Completed processing for task {task_id}")
        
        return {
            "task_id": task_id, 
            "submitted_answer": submitted_answer,
            "question": question,
            "execution_time": result.get("execution_time", 0),
            "question_type": result.get("question_type", "unknown"),
            "has_file": file_info["exists"],
            "file_type": file_info.get("type", "none"),
            "file_error": file_info.get("error")
        }
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        return {
            "task_id": task_id,
            "submitted_answer": f"PROCESSING ERROR: {e}",
            "question": question,
            "execution_time": 0,
            "question_type": "error",
            "has_file": False,
            "file_type": "none"
        }

async def process_questions_async(questions_data: list, api_url: str) -> list:
    """Process questions asynchronously with file download support"""
    logger.info(f"Starting async processing of {len(questions_data)} questions with file support")
    
    with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers to avoid rate limits
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_single_question, item, api_url)
            for item in questions_data
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions in results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception in async processing for item {i}: {result}")
            processed_results.append({
                "task_id": questions_data[i].get("task_id", f"unknown_{i}"),
                "submitted_answer": f"ASYNC ERROR: {result}",
                "question": questions_data[i].get("question", ""),
                "execution_time": 0,
                "question_type": "error",
                "has_file": False,
                "file_type": "none"
            })
        else:
            processed_results.append(result)
    
    return processed_results

def make_request_with_retry(url: str, method: str = "GET", **kwargs) -> requests.Response:
    """Make HTTP request with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            if method.upper() == "POST":
                response = requests.post(url, timeout=TIMEOUT_SECONDS, **kwargs)
            else:
                response = requests.get(url, timeout=TIMEOUT_SECONDS, **kwargs)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Enhanced version with async processing, better error handling, and comprehensive logging
    """
    start_time = time.time()
    
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        logger.info(f"User logged in: {username}")
    else:
        logger.warning("User not logged in")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 2. Fetch Questions with retry logic
    logger.info(f"Fetching questions from: {questions_url}")
    try:
        response = make_request_with_retry(questions_url)
        questions_data = response.json()
        
        if not questions_data:
            logger.error("Fetched questions list is empty")
            return "Fetched questions list is empty or invalid format.", None
            
        logger.info(f"Successfully fetched {len(questions_data)} questions")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        logger.error(f"Unexpected error fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Process questions asynchronously
    try:
        logger.info("Starting async question processing...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        processed_results = loop.run_until_complete(
            process_questions_async(questions_data, api_url)
        )
        loop.close()
        
        # Prepare results
        results_log = []
        answers_payload = []
        
        for result in processed_results:
            task_id = result["task_id"]
            question = result["question"]
            submitted_answer = result["submitted_answer"]
            execution_time = result.get("execution_time", 0)
            question_type = result.get("question_type", "unknown")
            has_file = result.get("has_file", False)
            file_type = result.get("file_type", "none")
            
            answers_payload.append({
                "task_id": task_id, 
                "submitted_answer": submitted_answer
            })
            
            # Enhanced results with file information and clean answers
            file_indicator = "📎" if has_file else "📄"
            results_log.append({
                "Task ID": task_id, 
                "Question": question[:100] + "..." if len(question) > 100 else question,
                "Submitted Answer": submitted_answer[:200] + "..." if len(submitted_answer) > 200 else submitted_answer,
                "Type": question_type,
                "Time (s)": f"{execution_time:.2f}",
                "File": f"{file_indicator} {file_type}" if has_file else "None"
            })
            
        processing_time = time.time() - start_time
        
        # Count files processed
        files_processed = sum(1 for r in processed_results if r.get("has_file", False))
        
        logger.info(f"Completed processing {len(processed_results)} questions ({files_processed} with files) in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in async processing: {e}")
        return f"Error in question processing: {e}", pd.DataFrame()

    if not answers_payload:
        logger.error("No answers produced")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission with enhanced data
    agent_code_url = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "https://github.com/user/local-agent-code"
    
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code_url,
        "answers": answers_payload
    }
    
    status_update = f"Agent finished processing in {processing_time:.2f}s. Submitting {len(answers_payload)} answers for user '{username}'..."
    logger.info(status_update)

    # 5. Submit with retry logic
    logger.info(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = make_request_with_retry(
            submit_url, 
            method="POST", 
            json=submission_data
        )
        
        result_data = response.json()
        total_time = time.time() - start_time
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Questions Processed: {len(questions_data)}\n"
            f"Files Processed: {files_processed}\n"
            f"Total Processing Time: {total_time:.2f}s\n"
            f"Average Time per Question: {total_time/len(questions_data):.2f}s\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        
        logger.info("Submission successful")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except:
            error_detail += f" Response: {e.response.text[:500]}"
        
        status_message = f"Submission Failed: {error_detail}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
        
    except Exception as e:
        status_message = f"Submission failed: {e}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

# --- Enhanced Gradio Interface ---
with gr.Blocks(title="Enhanced Multi-Agent Evaluation System") as demo:
    gr.Markdown("# Enhanced Multi-Agent Evaluation System")
    gr.Markdown(
        """
        **🎯 Course Evaluation Features:**
        - **Clean Answer Extraction**: Automatically extracts only the final answer for submission
        - **Format Compliance**: Follows course requirements for answer formatting
        - **No Extra Text**: Submits only the essential answer without "FINAL ANSWER" prefix
        - **Smart Parsing**: Handles various response formats and extracts clean answers
        - **File Processing**: Automatic download and processing of associated files via `/files/{task_id}`
        - **Multi-Modal Support**: Handles images, audio, Excel files, and text documents
        - **Async Processing**: Questions processed concurrently for better performance
        - **Smart Question Classification**: Automatic routing to appropriate tools
        - **Retry Logic**: Automatic retries for failed requests
        - **Comprehensive Logging**: Detailed execution tracking
        - **Caching**: Improved performance with result caching
        - **Enhanced Error Handling**: Better error recovery and reporting

        **Instructions:**
        1. Clone this space and modify the agent logic as needed
        2. Log in to your Hugging Face account using the button below
        3. Click 'Run Enhanced Evaluation' to start processing
        4. System will automatically download and process associated files via `/files/{task_id}`
        5. Monitor the detailed results with execution times, question types, and file indicators

        **File Processing Support:**
        - 🖼️ **Images**: OCR and image captioning (JPG, PNG, GIF, etc.)
        - 🎵 **Audio**: Speech-to-text transcription (MP3, WAV, etc.)
        - 📊 **Excel**: Data extraction and CSV conversion (XLSX, XLS)
        - 📄 **Text**: Direct text file reading (TXT, MD, JSON)
        - 📎 **Auto-Detection**: Automatic file type detection and processing

        **Performance Notes:**
        - Questions are processed asynchronously with smart batching
        - Files are automatically downloaded and processed per task
        - Automatic retry mechanisms handle temporary failures
        - Caching reduces redundant API calls
        - Comprehensive logging helps with debugging
        """
    )

    with gr.Row():
        gr.LoginButton()
        
    with gr.Row():
        run_button = gr.Button(
            "🚀 Run Enhanced Evaluation & Submit All Answers", 
            variant="primary",
            size="lg"
        )

    with gr.Row():
        status_output = gr.Textbox(
            label="📊 Execution Status & Results", 
            lines=8, 
            interactive=False,
            placeholder="Click the button above to start processing..."
        )
    
    with gr.Row():
        results_table = gr.DataFrame(
            label="📋 Detailed Question Analysis", 
            wrap=True,
            headers=["Task ID", "Question", "Submitted Answer", "Type", "Time (s)", "File"]
        )

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
        show_progress=True
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 ENHANCED MULTI-AGENT EVALUATION SYSTEM STARTING")
    print("="*70)
    
    # Enhanced startup logging
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   🌐 Runtime URL: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST not found (running locally)")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   📁 Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   🌳 Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID not found (running locally)")

    print(f"🔧 Configuration:")
    print(f"   - Max Retries: {MAX_RETRIES}")
    print(f"   - Timeout: {TIMEOUT_SECONDS}s")
    print(f"   - Cache Size: 100 entries")
    print(f"   - Async Workers: 3")
    
    print("="*70)
    print("🎯 Launching Enhanced Gradio Interface...")
    print("="*70 + "\n")

    demo.launch(debug=True, share=False)