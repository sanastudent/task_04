import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel
from tools import (
    read_user_profile, 
    update_user_profile, 
    extract_text_from_pdf,
    summarize_document,
    generate_quiz
)

load_dotenv() # Load environment variables from .env file

# Initialize AsyncOpenAI client for Gemini
openai_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize OpenAIChatCompletionsModel for Gemini-2.0-flash
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=openai_client
)

# Configure the agent
agent = Agent(
    name="PDF Assistant", # Changed name to reflect new capabilities
    instructions=(
        "You are an expert assistant for PDF processing, summarization, and quiz generation. "
        "Your primary goal is to help users process PDF documents, summarize text, and create quizzes. "
        "Always use the `extract_text_from_pdf` tool when a user asks you to process a PDF file path. "
        "Always use the `summarize_document` tool when a user asks for a summary of provided text. "
        "Always use the `generate_quiz` tool when a user asks for a quiz from provided text. "
        "You also greet users by name if known and detect when users share personal info to save it using `read_user_profile` and `update_user_profile` tools. "
        "When using a tool, you MUST respond with the exact output of the tool, even if it contains an error message or a traceback. "
        "Do NOT try to interpret, rephrase, or apologize for tool outputs, especially errors. Simply provide the raw tool output."
    ),
    model=gemini_model,
    tools=[
        read_user_profile, 
        update_user_profile,
        extract_text_from_pdf,
        summarize_document,
        generate_quiz
    ]
)
