import streamlit as st
import asyncio
import os
import tempfile
from agent import agent
from agents import Runner

# Asynchronous function to run the agent and display tool outputs
async def run_agent_and_display_tools(prompt: str) -> str:
    result = await Runner.run(agent, prompt)
    final_output_from_agent = result.final_output # This is what the agent's LLM decided to say

    tool_call_outputs = [item for item in result.new_items if item.__class__.__name__ == 'ToolCallOutputItem']
    
    # Prioritize actual tool output over the agent's conversational response if a tool was called
    if tool_call_outputs:
        # If multiple tool outputs, combine them. For quiz/summary, usually one is expected.
        combined_tool_output = "\n".join([getattr(item, 'output', 'No output attribute found') for item in tool_call_outputs])
        
        # Display tool outputs in a collapsible expander for debugging
        with st.expander("Agent Tool Outputs (for Debugging)"):
            st.info(combined_tool_output)
        
        # Return the raw tool output. This is what the Streamlit app expects for summary/quiz content.
        return combined_tool_output
    else:
        # If no tool was called, or if the agent just had a conversational response, return that.
        return final_output_from_agent


# Streamlit UI
st.set_page_config(page_title="PDF Assistant", layout="wide")
st.title("📄 PDF Summarizer & Quiz Generator")

# Initialize chat history and extracted text in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "quiz" not in st.session_state:
    st.session_state.quiz = None

# --- Sidebar for Chat with Agent (Memory) ---
with st.sidebar:
    st.header("Chat with Agent (Memory)")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Say something to the agent...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        with st.spinner("Agent thinking..."):
            agent_response = asyncio.run(run_agent_and_display_tools(prompt_input))
        
        st.session_state.messages.append({"role": "assistant", "content": agent_response})
        with st.chat_message("assistant"):
            st.markdown(agent_response)


# --- Main Content Area for PDF Processing ---
st.header("Upload PDF for Summary & Quiz")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    # Use uploaded_file.name as part of the key for extracted_text to ensure unique state per file
    # Check if a new file is uploaded or if extracted_text for this file is not in session state
    if st.session_state.uploaded_file_name != uploaded_file.name or \
       f"extracted_text_{uploaded_file.name}" not in st.session_state or \
       st.session_state.get(f"extracted_text_{uploaded_file.name}") is None: # Use .get for safety
        
        st.write("Extracting text from PDF...")
        pdf_bytes = uploaded_file.getvalue()
        
        # Save to a temporary file, as the agent tool expects a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_file_path = temp_pdf.name
        
        with st.spinner("Extracting text from PDF via agent..."):
            prompt_for_extraction = f"Please extract all text from the PDF file located at: {temp_file_path}"
            extracted_text_result = asyncio.run(run_agent_and_display_tools(prompt_for_extraction))
            
            # The agent returns the text or an error string. Now remove the temporary file.
            try:
                os.remove(temp_file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary PDF file: {e}")

            if extracted_text_result.startswith("Error"):
                st.error(extracted_text_result)
                st.session_state[f"extracted_text_{uploaded_file.name}"] = "" # Store empty string on error
            else:
                st.session_state[f"extracted_text_{uploaded_file.name}"] = extracted_text_result
            
            st.session_state.uploaded_file_name = uploaded_file.name # Update stored file name
            st.session_state.summary = None # Clear previous summary for new file
            st.session_state.quiz = None    # Clear previous quiz for new file

            if not extracted_text_result.startswith("Error"):
                st.success("Text extracted!")
            else:
                st.error("Text extraction failed.")
    
    extracted_text_for_display = st.session_state.get(f"extracted_text_{uploaded_file.name}", "") # Use .get for safety and consistent key

    # Only show controls if text was successfully extracted and is not empty
    if extracted_text_for_display and not extracted_text_for_display.startswith("Error"):
        with st.expander("View Extracted Text"):
            st.text_area("Extracted Text", extracted_text_for_display, height=300, disabled=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate Summary"):
                st.session_state.summary = None # Clear previous summary
                with st.spinner("Generating summary..."):
                    summary_prompt = f"Summarize the following document: {extracted_text_for_display}"
                    summary_output = asyncio.run(run_agent_and_display_tools(summary_prompt))
                    
                    if summary_output.startswith("Error"):
                        st.error(summary_output)
                        st.session_state.summary = "Error: " + summary_output
                    else:
                        st.session_state.summary = summary_output
                
                if st.session_state.summary and not st.session_state.summary.startswith("Error"):
                    st.success("Summary generated!")
        
        with col2:
            if st.button("Generate Quiz"):
                st.session_state.quiz = None # Clear previous quiz
                with st.spinner("Generating quiz..."):
                    quiz_prompt = f"Generate a quiz from the following document: {extracted_text_for_display}"
                    quiz_output = asyncio.run(run_agent_and_display_tools(quiz_prompt))
                    
                    if quiz_output.startswith("Error"):
                        st.error(quiz_output)
                        st.session_state.quiz = "Error: " + quiz_output
                    else:
                        st.session_state.quiz = quiz_output
                
                if st.session_state.quiz and not st.session_state.quiz.startswith("Error"):
                    st.success("Quiz generated!")

        if st.session_state.summary and not st.session_state.summary.startswith("Error"):
            st.subheader("Document Summary")
            st.markdown(st.session_state.summary)
        elif st.session_state.summary and st.session_state.summary.startswith("Error"):
            st.subheader("Document Summary")
            st.error(st.session_state.summary)

        if st.session_state.quiz and not st.session_state.quiz.startswith("Error"):
            st.subheader("Generated Quiz")
            st.markdown(st.session_state.quiz)
        elif st.session_state.quiz and st.session_state.quiz.startswith("Error"):
            st.subheader("Generated Quiz")
            st.error(st.session_state.quiz)