import streamlit as st
import pandas as pd
import os
import re
from groq import Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import tempfile
from langchain.callbacks import StreamlitCallbackHandler

# Set page title and layout
st.set_page_config(page_title="CSV Data Analysis Agent", layout="wide")

# Add header
st.title("CSV Data Analysis Agent")
st.markdown("Upload a CSV file and ask questions about your data")

# Using the hardcoded API key from the original file
GROQ_API_KEY = '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'

# Sidebar for model options only
with st.sidebar:
    st.header("Configuration")
    model_option = st.selectbox(
        "Select Groq Model",
        ["llama3-70b-8192", "llama3-8b-8192", "llama3-70b-4096"],
        index=0  # Default to llama3-70b-8192
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# Function to simplify complex queries
def simplify_complex_query(query):
    # Check complexity criteria
    def is_complex_query(text):
        complexity_markers = [
            r'\b(detailed|comprehensive|analysis|breakdown)\b',
            r'\w+\s+wise',
            r'\(.*\)',
            r'multiple\s+dimensions',
            r'cross-reference'
        ]

        complexity_score = sum(
            1 for marker in complexity_markers
            if re.search(marker, text, re.IGNORECASE)
        )

        return (
            complexity_score > 1 or
            len(text.split()) > 15 or
            any(keyword in text.lower() for keyword in [
                'include', 'alongside', 'detailed', 'comprehensive'
            ])
        )

    # If not complex, return original query
    if not is_complex_query(query):
        return query

    # Use Groq with Llama 3 to simplify complex queries
    try:
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""Simplify this complex query into a concise 3-4 line analytical breakdown
        with clear, numbered steps. Focus on the core analytical objectives:

        Original Query: {query}

        Provide a straightforward, actionable summary of the analysis steps."""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Simplify complex queries into clear, concise steps."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=200
        )

        simplified_query = chat_completion.choices[0].message.content.strip()
        return simplified_query

    except Exception as e:
        st.error(f"Error in query simplification: {e}")
        return query

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_csv_path = tmp_file.name
    
    try:
        # Get user query
        prompt = st.chat_input("Ask a question about your data")
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response with a spinner while processing
            with st.chat_message("assistant"):
                # Create a container for the thinking process
                thinking_container = st.container()
                
                with st.spinner("Analyzing your data..."):
                    # Create LLM
                    llm = ChatGroq(
                        temperature=temperature,
                        model_name=model_option,
                        groq_api_key=GROQ_API_KEY
                    )
                    
                    # Create a Streamlit callback handler to show the thinking process
                    st_callback = StreamlitCallbackHandler(thinking_container)
                    
                    # Create CSV Agent
                    agent_executor = create_csv_agent(
                        llm,
                        temp_csv_path,
                        verbose=True,
                        allow_dangerous_code=True,
                        allowed_tools=['python_repl_ast']
                    )
                    
                    # Get simplified query
                    simplified_query = simplify_complex_query(prompt)
                    
                    # If query was simplified, show the simplified version
                    if simplified_query != prompt:
                        st.info(f"Simplified query: {simplified_query}")
                    
                    # Get response from agent with callback to show thinking
                    response = agent_executor.invoke(
                        simplified_query, 
                        callbacks=[st_callback]
                    )
                    
                    # Extract the output
                    output = response.get("output", "No response generated.")
                    
                    # Separate container for final answer
                    answer_container = st.container()
                    with answer_container:
                        st.markdown(output)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": output})
        
    except Exception as e:
        st.error(f"Error processing the file: {e}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)

else:
    st.info("Please upload a CSV file to begin.")

# Removed the usage instructions footer