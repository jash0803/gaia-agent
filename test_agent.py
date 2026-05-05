"""Quick test: run agent on one non-file question and one video question."""
import requests
from dotenv import load_dotenv

load_dotenv()

from agent import GAIAAgent

agent = GAIAAgent()

# Fetch all questions
resp = requests.get("https://agents-course-unit4-scoring.hf.space/questions", timeout=15)
questions = resp.json()

# Test 1: Non-file question (should NOT go to file_agent)
non_file = [q for q in questions if not q.get("file_name") and "youtube" not in q.get("question", "").lower()]
if non_file:
    q = non_file[0]
    print(f"=== TEST 1: Non-file question ===")
    print(f"Question: {q['question'][:150]}...")
    answer = agent(q["question"], task_id=q["task_id"], file_name=q.get("file_name", ""))
    print(f">>> ANSWER: {answer}\n")

# Test 2: YouTube question (should use Gemini video)
video_qs = [q for q in questions if "youtube" in q.get("question", "").lower()]
if video_qs:
    q = video_qs[0]
    print(f"=== TEST 2: YouTube question ===")
    print(f"Question: {q['question'][:150]}...")
    answer = agent(q["question"], task_id=q["task_id"], file_name=q.get("file_name", ""))
    print(f">>> ANSWER: {answer}\n")
