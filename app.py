import os
import gradio as gr
import requests
import inspect
import pandas as pd
from langgraph.graph import StateGraph, END, START
from typing import Dict, Any
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from transformers import pipeline
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.tools.python.tool import PythonREPLTool

import load_dotenv

# Load environment variables
load_dotenv.load_dotenv()

# --- State Definition ---
from typing import TypedDict
class QuestionState(TypedDict, total=False):
    question: str
    answer: str
    error: str

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Tool Wrappers ---
search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
audio_transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
ocr = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
python_repl = PythonREPL()

# --- Tool Wrappers ---
def web_search_tool(q):
    try:
        results = search.results(query=q, include_answer=True)
        if results and results[0].get("answer"):
            return results[0]["answer"]
        else:
            return " ".join(r.get("content", "") for r in results[:2])
    except Exception as e:
        return f"Search error: {e}"

def audio_tool_stub(audio):
    try:
        result = audio_transcriber(audio)
        return result['text'] if 'text' in result else "No transcription available."
    except Exception as e:
        return f"Audio transcription error: {e}"
    
def ocr_tool_stub(image):
    try:
        result = ocr(image)
        return result[0]['generated_text'] if result else "No text extracted."
    except Exception as e:
        return f"OCR error: {e}"
    
python_repl_tool = PythonREPLTool(
    python_repl=python_repl,
    return_direct=True,
    descriptor="Executes Python code in a REPL environment for solving programming and math tasks."
)

# --- Tool-using LLM Agent ---
tools = [
    Tool(name="WebSearch", func=web_search_tool, description="Useful for answering questions using Wikipedia or current facts from the internet."),
    Tool(name="AudioTranscribe", func=audio_tool_stub, description="Use for processing .mp3 or audio recordings for answers."),
    Tool(name="OCR", func=ocr_tool_stub, description="Use to analyze text or chess boards in image input."),
    Tool(name="ExcelProcessor", func=lambda f: pd.read_excel(f).to_csv(index=False), description="Reads and extracts data from an Excel file and returns as CSV text."),
    python_repl_tool,
]

llm = ChatOpenAI(temperature=0, model_name="gpt-4")
langchain_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=300
)

# --- LangGraph Nodes ---
def agent_node(state: QuestionState) -> Dict[str, Any]:
    question = state.get("question")
    if not question:
        return {**state, "error": "No question provided"}

    try:
        result = langchain_agent.run(question)
        return {**state, "answer": result}
    except Exception as e:
        return {**state, "answer": f"Agent error: {e}"}


# --- LangGraph Compile ---
graph = StateGraph(QuestionState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
agent_graph = graph.compile()

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        # Initialize state for each question
        state: QuestionState = {"question": None}
        
        task_id = item.get("task_id")
        question = item.get("question")
        try:
            state = {"question": question}
            result = agent_graph.invoke(state)
            if result is None:
                # Fallback to direct agent call
                result = agent_node(state)
            submitted_answer = result.get("answer", "ERROR")
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {
        "username": username.strip(),
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers_payload
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)