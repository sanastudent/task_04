import json
import os
import tempfile
import traceback # Added for detailed error reporting
from agents import function_tool
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, ModelSettings
import tiktoken # Added for token counting and truncation

load_dotenv() # Load environment variables from .env file

USER_PROFILE_PATH = "user_profile.json"
MAX_INPUT_TOKENS = 10000 # Define a maximum input token limit for LLM calls

# Internal helper function, not exposed as a tool
def _read_user_profile_logic() -> dict:
    if not os.path.exists(USER_PROFILE_PATH):
        return {}
    try:
        with open(USER_PROFILE_PATH, "r") as f:
            content = f.read()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

# Tool definition for user profile
@function_tool
def read_user_profile() -> dict:
    """
    Reads and returns user profile data from 'user_profile.json'.
    Returns an empty dictionary if the file does not exist or is invalid.
    """
    return _read_user_profile_logic()

@function_tool
def update_user_profile(key: str, value: str):
    """
    Updates a specific key-value pair in the user profile and saves it to 'user_profile.json'.
    If the file does not exist, it will be created.
    """
    profile = _read_user_profile_logic()
    profile[key] = value
    with open(USER_PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=4)

@function_tool
def extract_text_from_pdf(pdf_file_path: str) -> str:
    """
    Extracts text content from a PDF file given its path.
    The file is expected to be accessible at the given path.
    """
    try:
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF at {pdf_file_path}: {e}\n{traceback.format_exc()}"

@function_tool
async def summarize_document(document_text: str) -> str:
    """
    Summarizes the provided document text using the Gemini LLM.
    """
    try:
        # Initialize AsyncOpenAI client for Gemini within the tool
        openai_client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        # Truncate document_text if it's too long
        encoding = tiktoken.encoding_for_model("gpt-4") # Using gpt-4 tokenizer as a proxy
        encoded_text = encoding.encode(document_text)
        if len(encoded_text) > MAX_INPUT_TOKENS:
            truncated_text = encoding.decode(encoded_text[:MAX_INPUT_TOKENS])
            document_text = truncated_text
            # print(f"Warning: Document text truncated to {MAX_INPUT_TOKENS} tokens for summarization.")

        # Craft a prompt for summarization
        prompt_string = f"Please provide a concise summary of the following document:\n\n{document_text}"
        
        # Directly use the openai_client to call the LLM
        response = await openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt_string}],
            max_tokens=500
        )
        
        # Extract the summary from the response
        summary = response.choices[0].message.content or "No summary was generated."
        return summary
    except Exception as e:
        return f"Error summarizing document: {e}\n{traceback.format_exc()}"

@function_tool
async def generate_quiz(document_text: str) -> str:
    """
    Generates a quiz (MCQs and mixed-style questions) from the provided document text using the Gemini LLM.
    """
    try:
        # Initialize AsyncOpenAI client for Gemini within the tool
        openai_client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Truncate document_text if it's too long
        encoding = tiktoken.encoding_for_model("gpt-4") # Using gpt-4 tokenizer as a proxy
        encoded_text = encoding.encode(document_text)
        if len(encoded_text) > MAX_INPUT_TOKENS:
            truncated_text = encoding.decode(encoded_text[:MAX_INPUT_TOKENS])
            document_text = truncated_text
            # print(f"Warning: Document text truncated to {MAX_INPUT_TOKENS} tokens for quiz generation.")

        # Craft a prompt for quiz generation
        prompt_string = (
            f"Generate a quiz from the following document. "
            f"Include 3-5 Multiple Choice Questions (MCQs) and 2-3 mixed-style questions (e.g., true/false, short answer).\n\n"
            f"Document:\n{document_text}\n\n"
            f"Please format the quiz clearly with questions and answer options/spaces for answers."
        )
        
        # Directly use the openai_client to call the LLM
        response = await openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt_string}],
            max_tokens=1000
        )
        
        # Extract the quiz from the response
        quiz = response.choices[0].message.content or "No quiz was generated."
        return quiz
    except Exception as e:
        return f"Error generating quiz: {e}\n{traceback.format_exc()}"
