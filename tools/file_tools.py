import os
import base64
import tempfile
import mimetypes
import requests
import pandas as pd
from langchain_core.tools import tool
from openai import OpenAI

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

GAIA_HF_REPO = "gaia-benchmark/GAIA"
GAIA_VALIDATION_PATH = "2023/validation"


def _download_from_api(task_id: str) -> str | None:
    """Try downloading from the scoring API. Returns local path or None."""
    url = f"{DEFAULT_API_URL}/files/{task_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception:
        return None

    content_disposition = resp.headers.get("content-disposition", "")
    filename = "downloaded_file"
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip('"').strip("'")

    ext = os.path.splitext(filename)[1]
    if not ext:
        content_type = resp.headers.get("content-type", "")
        ext = mimetypes.guess_extension(content_type) or ""

    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, filename if filename != "downloaded_file" else f"file{ext}")
    with open(file_path, "wb") as f:
        f.write(resp.content)
    return file_path


def _download_from_hf(task_id: str, file_name: str) -> str | None:
    """Download the file from the HuggingFace GAIA dataset."""
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")
    hf_path = f"{GAIA_VALIDATION_PATH}/{file_name}"

    try:
        local_path = hf_hub_download(
            repo_id=GAIA_HF_REPO,
            repo_type="dataset",
            filename=hf_path,
            token=hf_token,
        )
        return local_path
    except Exception as e:
        print(f"HF download failed for {hf_path}: {e}")
        return None


@tool
def download_gaia_file(task_id: str, file_name: str = "") -> str:
    """Download the file associated with a GAIA task ID.
    Args:
        task_id: The GAIA task ID.
        file_name: The known file name (e.g. 'abc123.png'). If provided, speeds up download.
    Returns the local file path where the file was saved,
    or an error message if no file exists for this task.
    """
    # Try the scoring API first
    path = _download_from_api(task_id)
    if path:
        return path

    # Fall back to HuggingFace dataset
    if file_name:
        path = _download_from_hf(task_id, file_name)
        if path:
            return path

    if not file_name:
        return f"No file associated with task {task_id}."

    return f"Could not download file for task {task_id}."


@tool
def read_excel_or_csv(file_path: str, query: str = "") -> str:
    """Read an Excel (.xlsx/.xls) or CSV file and return its contents or answer a query about it.
    Args:
        file_path: Path to the Excel or CSV file.
        query: Optional description of what to look for in the data.
    Returns a string summary of the data.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)

        info_parts = [
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            f"Data types:\n{df.dtypes.to_string()}",
            f"\nFirst 20 rows:\n{df.head(20).to_string()}",
        ]

        if df.shape[0] <= 100:
            info_parts.append(f"\nFull data:\n{df.to_string()}")

        info_parts.append(f"\nBasic statistics:\n{df.describe(include='all').to_string()}")

        return "\n".join(info_parts)

    except Exception as e:
        return f"Error reading file: {e}"


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file (MP3, WAV, M4A, etc.) to text using OpenAI Whisper.
    Args:
        file_path: Path to the audio file.
    Returns the transcribed text.
    """
    try:
        client = OpenAI()
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text
    except Exception as e:
        return f"Error transcribing audio: {e}"


@tool
def analyze_image(file_path: str, question: str = "Describe this image in detail.") -> str:
    """Analyze an image file using gpt-4o vision via the Responses API.
    Args:
        file_path: Path to the image file (PNG, JPG, JPEG, GIF, WEBP).
        question: What to analyze or look for in the image.
    Returns the analysis result as text.
    """
    try:
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
        mime_type = mime_map.get(ext, "png")

        client = OpenAI()
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/{mime_type};base64,{image_data}",
                        },
                    ],
                }
            ],
        )
        return response.output_text
    except Exception as e:
        return f"Error analyzing image: {e}"


@tool
def read_text_file(file_path: str) -> str:
    """Read a plain text file (.txt, .py, .md, .json, .xml, .html, etc.) and return its contents.
    Args:
        file_path: Path to the text file.
    Returns the file contents as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 50000:
            return content[:50000] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def read_pdf(file_path: str) -> str:
    """Read a PDF file and extract its text content.
    Args:
        file_path: Path to the PDF file.
    Returns the extracted text.
    """
    try:
        import PyPDF2
        text_parts = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        result = "\n".join(text_parts)
        if len(result) > 50000:
            return result[:50000] + "\n... (truncated)"
        return result if result.strip() else "PDF appears to contain no extractable text (may be image-based)."
    except Exception as e:
        return f"Error reading PDF: {e}"


def get_file_tools():
    return [
        download_gaia_file,
        read_excel_or_csv,
        transcribe_audio,
        analyze_image,
        read_text_file,
        read_pdf,
    ]
