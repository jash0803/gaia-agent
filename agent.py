import re
import logging
from datetime import datetime
from agents.supervisor import build_supervisor_graph

logger = logging.getLogger("gaia_agent")
_log_handler = logging.FileHandler("gaia_agent.log", mode="a")
_log_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)

INTERNAL_ROUTING_PATTERNS = re.compile(
    r"^transfer_to_\w+$|^handoff_to_\w+$|^route_to_\w+$", re.IGNORECASE
)


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    if INTERNAL_ROUTING_PATTERNS.match(text.strip()):
        return ""

    # Look for "FINAL ANSWER: ..." pattern anywhere in the text
    fa_match = re.search(r"(?i)FINAL\s*ANSWER\s*:\s*(.+)", text)
    if fa_match:
        return fa_match.group(1).strip()

    # Fallback: strip common prefixes from the last non-empty line
    prefixes_to_strip = [
        r"(?i)^the\s+answer\s+is\s*:\s*",
        r"(?i)^answer\s*:\s*",
    ]
    cleaned = text.strip()
    for pattern in prefixes_to_strip:
        cleaned = re.sub(pattern, "", cleaned).strip()

    lines = cleaned.strip().split("\n")
    if lines:
        last_non_empty = ""
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not INTERNAL_ROUTING_PATTERNS.match(stripped):
                last_non_empty = stripped
                break
        for pattern in prefixes_to_strip:
            last_non_empty = re.sub(pattern, "", last_non_empty).strip()
        if last_non_empty:
            cleaned = last_non_empty

    return cleaned.strip()


def _extract_trace(messages) -> tuple[list[str], list[str]]:
    """Walk the message list and collect which agents and tools were invoked."""
    agents_used = []
    tools_used = []
    for msg in messages:
        msg_type = type(msg).__name__
        name = getattr(msg, "name", None)
        if msg_type == "AIMessage" and name and name != "supervisor":
            if name not in agents_used:
                agents_used.append(name)
        if msg_type == "ToolMessage" and name:
            if name not in tools_used:
                tools_used.append(name)
    return agents_used, tools_used


class GAIAAgent:
    def __init__(self):
        print("Initializing GAIAAgent with multi-agent supervisor...")
        self.graph = build_supervisor_graph()
        logger.info("--- Session started ---")
        print("GAIAAgent initialized successfully.")

    def __call__(self, question: str, task_id: str | None = None, file_name: str = "") -> str:
        print(f"\n{'='*60}")
        print(f"Question (first 100 chars): {question[:100]}...")
        print(f"Task ID: {task_id}")

        has_file = bool(file_name)
        print(f"Associated file: {'yes (' + file_name + ')' if has_file else 'no'}")

        prompt = question
        if has_file and task_id:
            prompt = (
                f"{question}\n\n"
                f"[IMPORTANT CONTEXT: This question has an associated file named '{file_name}'. "
                f"You MUST use the download_gaia_file tool with task_id='{task_id}' and "
                f"file_name='{file_name}' to download and process this file before answering.]"
            )
        elif task_id:
            prompt = (
                f"{question}\n\n"
                f"[Context: Task ID is '{task_id}'. If you need to download an associated file, "
                f"use the download_gaia_file tool with this task_id.]"
            )

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.graph.invoke(
                {"messages": messages},
                config={"recursion_limit": 50},
            )

            response_messages = result.get("messages", [])

            agents_used, tools_used = _extract_trace(response_messages)

            if response_messages:
                final_msg = response_messages[-1]
                raw_answer = (
                    final_msg.content
                    if hasattr(final_msg, "content")
                    else str(final_msg)
                )
            else:
                raw_answer = str(result)

            answer = _extract_answer(raw_answer)

            logger.info(
                f"Q: {question[:80]}... | "
                f"file={'yes' if has_file else 'no'} | "
                f"agents: {', '.join(agents_used) or 'none'} | "
                f"tools: {', '.join(tools_used) or 'none'} | "
                f"answer: {answer[:80]}"
            )

            print(f"Agents used: {agents_used}")
            print(f"Tools used: {tools_used}")
            print(f"Final answer: {answer}")
            print(f"{'='*60}\n")
            return answer

        except Exception as e:
            print(f"Error running agent: {e}")
            logger.info(f"Q: {question[:80]}... | ERROR: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {e}"
