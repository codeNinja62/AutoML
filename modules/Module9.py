"""Module 9: Chat with Dataset using Gemini API."""

from __future__ import annotations

import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChatInterface:
    def __init__(self, api_key: str | None = None):
        # Priority: 1. st.secrets (Streamlit Cloud), 2. .env (local), 3. Explicit arg
        import streamlit as st
        
        self.api_key = None
        
        # 1. Try st.secrets first (for Streamlit Cloud)
        try:
            secret_key = st.secrets.get("GEMINI_API_KEY")
            if secret_key:
                self.api_key = secret_key.strip().strip('"').strip("'")
        except Exception:
            pass
        
        # 2. Try .env (for local development)
        if not self.api_key:
            env_key = os.getenv("GEMINI_API_KEY")
            if env_key:
                self.api_key = env_key.strip().strip('"').strip("'")
        
        # 3. Use explicit arg as fallback
        if not self.api_key and api_key:
            self.api_key = api_key.strip().strip('"').strip("'")
        
        self.model = None
        self.chat_session = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")

    def is_configured(self) -> bool:
        return self.model is not None

    def start_new_chat(self, df: pd.DataFrame, training_results: str | None = None) -> None:
        """Initialize a new chat session with dataset and training context."""
        if not self.model:
            return

        # Create a context prompt about the dataset
        schema = str(df.dtypes)
        summary = str(df.describe())
        head = str(df.head(5))
        
        training_context = f"\nMODEL TRAINING RESULTS:\n{training_results}" if training_results else ""

        system_prompt = f"""
You are an expert data science assistant. You help students understand their AutoML results.
Here is the context of the dataset and the training results:

DATASET SCHEMA:
{schema}

SUMMARY STATISTICS:
{summary}

DATA PREVIEW:
{head}
{training_context}

Please answer the user's questions based on this specific information. 
If asked which model is best, refer to the training results.
If asked about data patterns, refer to the schema and summary.
Be highly specific, task-oriented, and educational. Avoid generic apologies.
"""
        history = [
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["I have analyzed your dataset and the training results. Ask me anything about the data, the models, or the insights found."]}
        ]
        self.chat_session = self.model.start_chat(history=history)

    def send_message(self, message: str) -> str:
        """Send a message to the chat session and get response."""
        if not self.chat_session:
            return "Chat session not initialized. Please configure API key and load a dataset."
        
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {e}"
